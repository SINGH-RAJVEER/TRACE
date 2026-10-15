#Requires -Version 7.2
# Shared helpers for the launcher scripts. Dot-source this file; do not execute it.
#
# Runtime selection, in order:
#   1. uv   uv on PATH. Missing system tools are reported before anything starts.
#   2. pip  No uv. A Python 3.14 interpreter and pip build .venv-pip.
# Set DISTILL_RUNTIME=uv|pip to force one.

Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$script:LogDir = Join-Path $script:RepoRoot 'artifacts/logs'
$script:DefaultTeacherModel = 'Qwen/Qwen3.5-9B'
$script:DefaultTeacherRevision = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
$script:Runtime = ''
$script:AppBin = ''
$script:Vllm = ''
$script:Background = [System.Collections.Generic.List[object]]::new()

function Write-Log([string]$Message) {
	Write-Host "[$script:ScriptName] $Message"
}

function Stop-WithError([string]$Message) {
	Write-Host "[$script:ScriptName] error: $Message" -ForegroundColor Red
	exit 1
}

function Get-EnvValue([string]$Name, [string]$Default = '') {
	$value = [Environment]::GetEnvironmentVariable($Name)
	if ([string]::IsNullOrEmpty($value)) { return $Default }
	return $value
}

# Exports .env values that are not already set in the environment.
function Import-DotEnv {
	$file = Join-Path $script:RepoRoot '.env'
	if (-not (Test-Path $file)) { return }
	foreach ($line in Get-Content $file) {
		if ($line -match '^\s*(#|$)' -or $line -notmatch '=') { continue }
		$key, $value = $line -split '=', 2
		$key = ($key -replace '^\s*export\s+', '').Trim()
		if ($value -match '^"(.*)"$' -or $value -match "^'(.*)'$") { $value = $Matches[1] }
		if ($key -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') { continue }
		if ($null -eq [Environment]::GetEnvironmentVariable($key)) {
			[Environment]::SetEnvironmentVariable($key, $value)
		}
	}
}

function Get-VenvExecutable([string]$Venv, [string]$Name) {
	if ($IsWindows) { return Join-Path $Venv "Scripts/$Name.exe" }
	return Join-Path $Venv "bin/$Name"
}

function Test-Command([string]$Name) {
	return [bool](Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue)
}

# Returns the path of the first interpreter that is at least the given version, or $null.
# Each candidate is a command plus optional launcher arguments, such as @('py', '-3.14').
function Find-Python([version]$Minimum, [object[]]$Candidates) {
	$check = "import sys; print(sys.executable); sys.exit(sys.version_info[:2] < ($($Minimum.Major), $($Minimum.Minor)))"
	foreach ($candidate in $Candidates) {
		$command = @($candidate)
		if (-not (Test-Command $command[0])) { continue }
		$launcherArgs = @($command | Select-Object -Skip 1)
		$path = & $command[0] @launcherArgs -c $check 2>$null
		if ($LASTEXITCODE -eq 0 -and $path) { return ($path | Select-Object -Last 1).Trim() }
	}
	return $null
}

function Find-WorkspacePython {
	return Find-Python ([version]'3.14') @(@('python3.14'), @('py', '-3.14'), @('python3'), @('python'))
}

function Invoke-Checked([string]$FilePath, [string[]]$Arguments) {
	& $FilePath @Arguments
	if ($LASTEXITCODE -ne 0) { Stop-WithError "$FilePath exited with code $LASTEXITCODE" }
}

function Find-VllmPython {
	$python = Get-EnvValue 'VLLM_PYTHON' '3.12'
	if ($python -match '^[0-9.]+$') { $python = "python$python" }
	return Find-Python ([version]'3.10') @(@($python), @('python3.12'), @('python3.13'), @('python3.11'), @('python3'))
}

function Test-VllmInstalled {
	return [bool](Get-EnvValue 'VLLM_BIN') -or (Test-Command 'vllm') -or
		(Test-Path (Get-VenvExecutable (Join-Path $script:RepoRoot '.venv-vllm') 'vllm'))
}

# Picks the runtime and reports missing tools.
function Select-Runtime([switch]$NeedsGpu, [switch]$NeedsVllm) {
	$missing = [System.Collections.Generic.List[string]]::new()
	$warnings = [System.Collections.Generic.List[string]]::new()
	$script:Runtime = Get-EnvValue 'DISTILL_RUNTIME' 'auto'
	if ($script:Runtime -eq 'auto') {
		$script:Runtime = if (Test-Command 'uv') { 'uv' } else { 'pip' }
	}
	switch ($script:Runtime) {
		'uv' {
			if (-not (Test-Command 'uv')) { $missing.Add('uv (https://docs.astral.sh/uv/)') }
		}
		'pip' {
			$warnings.Add('using pip, which does not read uv.lock; install uv for locked dependencies')
			$python = Find-WorkspacePython
			if (-not $python) { $missing.Add('Python 3.14 (python3.14 or py -3.14)') }
			else {
				& $python -c 'import ensurepip, venv' 2>$null
				if ($LASTEXITCODE -ne 0) { $missing.Add("venv and ensurepip for $python (python3.14-venv on Debian and Ubuntu)") }
			}
			if ($NeedsVllm -and -not $IsWindows -and -not (Test-VllmInstalled) -and -not (Find-VllmPython)) {
				$missing.Add('Python 3.10 to 3.13 for vLLM (set VLLM_PYTHON)')
			}
		}
		default { Stop-WithError "unknown DISTILL_RUNTIME '$script:Runtime'; use uv or pip" }
	}
	if ($NeedsVllm -and $IsWindows) {
		$missing.Add('vLLM, which does not run natively on Windows; start an OpenAI-compatible server and pass -UpstreamUrl')
	}
	if ($NeedsGpu -and -not (Test-Command 'nvidia-smi')) {
		$warnings.Add('nvidia-smi not found; vLLM and QLoRA training need an NVIDIA GPU and driver')
	}
	if ($NeedsGpu -and -not $IsWindows -and -not (Test-Command 'cc') -and -not (Test-Command 'gcc')) {
		$warnings.Add('no C compiler (cc or gcc); Triton kernels in vLLM and training compile with one')
	}

	Write-Log "runtime: $script:Runtime"
	foreach ($item in $warnings) { Write-Log "warning: $item" }
	if ($missing.Count -gt 0) {
		foreach ($item in $missing) { Write-Log "missing: $item" }
		Stop-WithError 'install the missing tools above and run again'
	}
}

# Installs the workspace packages and sets AppBin.
function Initialize-AppEnvironment([switch]$Train) {
	Set-Location $script:RepoRoot
	if ($script:Runtime -eq 'pip') {
		$venv = Join-Path $script:RepoRoot '.venv-pip'
		$python = Get-VenvExecutable $venv 'python'
		if (-not (Test-Path $python)) { Invoke-Checked (Find-WorkspacePython) @('-m', 'venv', $venv) }
		$distillation = if ($Train) { './packages/distillation[train]' } else { './packages/distillation' }
		Write-Log 'installing workspace packages with pip'
		Invoke-Checked $python @('-m', 'pip', 'install', '--quiet', '--upgrade', 'pip')
		Invoke-Checked $python @('-m', 'pip', 'install', '--quiet', '-e', $distillation, '-e', './apps/teacher-api')
		$script:AppBin = $venv
	}
	else {
		$arguments = @('sync', '--all-packages', '--inexact')
		if ($Train) { $arguments += @('--extra', 'train') }
		Write-Log 'installing workspace packages with uv'
		Invoke-Checked 'uv' $arguments
		$script:AppBin = Join-Path $script:RepoRoot '.venv'
	}
}

function Get-AppExecutable([string]$Name) {
	return Get-VenvExecutable $script:AppBin $Name
}

# Locates vLLM or installs it into .venv-vllm, separate from the Python 3.14 workspace.
function Initialize-Vllm {
	if ($IsWindows) {
		Stop-WithError ('vLLM does not run natively on Windows. Start an OpenAI-compatible server ' +
			'(vLLM under WSL2, llama.cpp, or LM Studio) and pass -UpstreamUrl')
	}
	$explicit = Get-EnvValue 'VLLM_BIN'
	if ($explicit) { $script:Vllm = $explicit }
	elseif (Test-Command 'vllm') { $script:Vllm = (Get-Command 'vllm' -CommandType Application).Source }
	else {
		$venv = Join-Path $script:RepoRoot '.venv-vllm'
		$spec = Get-EnvValue 'VLLM_SPEC' 'vllm'
		$python = Get-EnvValue 'VLLM_PYTHON' '3.12'
		$script:Vllm = Get-VenvExecutable $venv 'vllm'
		if (-not (Test-Path $script:Vllm)) {
			Write-Log "installing $spec into .venv-vllm (several GB)"
			if ($script:Runtime -eq 'pip') {
				$interpreter = Find-VllmPython
				if (-not $interpreter) { Stop-WithError 'vLLM needs a Python interpreter between 3.10 and 3.13; set VLLM_PYTHON' }
				Invoke-Checked $interpreter @('-m', 'venv', $venv)
				$venvPython = Get-VenvExecutable $venv 'python'
				Invoke-Checked $venvPython @('-m', 'pip', 'install', '--quiet', '--upgrade', 'pip')
				Invoke-Checked $venvPython @('-m', 'pip', 'install', $spec)
			}
			else {
				Invoke-Checked 'uv' @('venv', '--python', $python, $venv)
				Invoke-Checked 'uv' @('pip', 'install', '--python', (Get-VenvExecutable $venv 'python'), $spec)
			}
		}
	}
	if (-not (Test-Path $script:Vllm)) { Stop-WithError "vLLM executable not found at $script:Vllm" }
}

# Quotes arguments with the rules both Windows and .NET on Unix use to split a command line.
function ConvertTo-CommandLine([string[]]$Arguments) {
	$quoted = foreach ($argument in $Arguments) {
		if ($argument -ne '' -and $argument -notmatch '[\s"]') { $argument; continue }
		'"' + (($argument -replace '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1') + '"'
	}
	return $quoted -join ' '
}

# Starts a process. LogFile receives stdout, LogFile with .err.log receives stderr.
function Start-Background([string]$Name, [string]$LogFile, [string]$FilePath, [string[]]$Arguments = @()) {
	$parameters = @{ FilePath = $FilePath; PassThru = $true; NoNewWindow = $true }
	if ($Arguments.Count -gt 0) { $parameters.ArgumentList = ConvertTo-CommandLine $Arguments }
	if ($LogFile) {
		New-Item -ItemType Directory -Force -Path $script:LogDir | Out-Null
		$parameters.RedirectStandardOutput = $LogFile
		$parameters.RedirectStandardError = [IO.Path]::ChangeExtension($LogFile, '.err.log')
	}
	$process = Start-Process @parameters
	$script:Background.Add([pscustomobject]@{ Name = $Name; Process = $process })
	Write-Log "started $Name (pid $($process.Id))"
	return $process
}

# Stops background services, newest first, with SIGTERM where available, then a tree kill.
function Stop-Background {
	$kill = Get-Command 'kill' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
	for ($index = $script:Background.Count - 1; $index -ge 0; $index--) {
		$entry = $script:Background[$index]
		if ($entry.Process.HasExited) { continue }
		Write-Log "stopping $($entry.Name) (pid $($entry.Process.Id))"
		if (-not $IsWindows -and $kill) { & $kill.Source -TERM $entry.Process.Id 2>$null }
		if (-not $entry.Process.WaitForExit(30000)) { $entry.Process.Kill($true) }
	}
	$script:Background.Clear()
}

function Show-LogTail([string]$LogFile) {
	if (-not $LogFile) { return }
	foreach ($path in @($LogFile, [IO.Path]::ChangeExtension($LogFile, '.err.log'))) {
		if (Test-Path $path) { Get-Content $path -Tail 40 | Write-Host }
	}
}

# Waits until a URL returns 2xx and its body contains the expected text.
function Wait-Http([string]$Url, [string]$Expected, [int]$TimeoutSeconds, $Process = $null, [string]$LogFile = '') {
	Write-Log "waiting for $Url"
	$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
	while ((Get-Date) -lt $deadline) {
		try {
			$response = Invoke-WebRequest -Uri $Url -TimeoutSec 5 -SkipHttpErrorCheck
			$body = if ($response.Content -is [byte[]]) { [Text.Encoding]::UTF8.GetString($response.Content) } else { [string]$response.Content }
			if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300 -and $body.Contains($Expected)) { return }
		}
		catch { }
		if ($Process -and $Process.HasExited) {
			Show-LogTail $LogFile
			Stop-WithError "process for $Url exited during startup"
		}
		Start-Sleep -Seconds 3
	}
	Stop-WithError "timed out after $($TimeoutSeconds)s waiting for $Url"
}

# Sends one small generation through the teacher API to check auth and non-thinking mode.
function Test-TeacherSmoke([string]$TeacherUrl, [string]$Token) {
	$headers = @{}
	if ($Token) { $headers.Authorization = "Bearer $Token" }
	$body = @{
		messages = @(@{ role = 'user'; content = 'Reply with the single word: ready' })
		temperature = 0
		max_output_tokens = 32
	} | ConvertTo-Json -Depth 4
	try {
		$response = Invoke-WebRequest -Uri "$($TeacherUrl.TrimEnd('/'))/v1/teacher/generate" -Method Post `
			-ContentType 'application/json' -Headers $headers -Body $body -TimeoutSec 300 -SkipHttpErrorCheck
	}
	catch { Stop-WithError "teacher smoke test failed: $($_.Exception.Message)" }
	if ($response.StatusCode -ne 200) {
		Write-Host $response.Content
		Stop-WithError "teacher smoke test failed with HTTP $($response.StatusCode)"
	}
	if ($response.Content -match '<think>') {
		Stop-WithError 'teacher response contains a <think> trace; disable Qwen thinking mode upstream'
	}
	Write-Log 'teacher smoke test passed'
}

# Starts vLLM (unless an upstream URL is given) and the teacher API, then waits for both.
# The teacher API logs to ApiLog, or to the console when it is empty.
function Start-TeacherStack([string]$BindHost, [int]$Port, [string]$UpstreamUrl, [string]$ApiLog) {
	$env:TEACHER_UPSTREAM_MODEL = Get-EnvValue 'TEACHER_UPSTREAM_MODEL' $script:DefaultTeacherModel
	if (-not $UpstreamUrl) {
		Initialize-Vllm
		$vllmPort = Get-EnvValue 'VLLM_PORT' '8001'
		$vllmLog = Join-Path $script:LogDir 'vllm.log'
		$extra = @((Get-EnvValue 'VLLM_EXTRA_ARGS') -split '\s+' | Where-Object { $_ })
		# Qwen3.5 checkpoints include a vision encoder; text-only serving skips it and its
		# profiling, leaving that memory for the KV cache. No reasoning parser: it would move
		# <think> text out of the content that the smoke test and collector check.
		$arguments = @(
			'serve', $env:TEACHER_UPSTREAM_MODEL,
			'--revision', (Get-EnvValue 'TEACHER_MODEL_REVISION' $script:DefaultTeacherRevision),
			'--served-model-name', $env:TEACHER_UPSTREAM_MODEL,
			'--host', '127.0.0.1',
			'--port', $vllmPort,
			'--max-model-len', (Get-EnvValue 'VLLM_MAX_MODEL_LEN' '8192'),
			'--gpu-memory-utilization', (Get-EnvValue 'VLLM_GPU_MEMORY_UTILIZATION' '0.90'),
			'--default-chat-template-kwargs', '{"enable_thinking": false}',
			'--language-model-only'
		) + $extra
		$vllmProcess = Start-Background 'vllm' $vllmLog $script:Vllm $arguments
		Write-Log "vLLM log: $vllmLog"
		Wait-Http "http://127.0.0.1:$vllmPort/health" '' ([int](Get-EnvValue 'VLLM_STARTUP_TIMEOUT' '1800')) $vllmProcess $vllmLog
		$UpstreamUrl = "http://127.0.0.1:$vllmPort/v1"
		$env:TEACHER_PROVIDER_NAME = Get-EnvValue 'TEACHER_PROVIDER_NAME' 'self-hosted-vllm'
	}
	$env:TEACHER_UPSTREAM_BASE_URL = $UpstreamUrl
	$env:TEACHER_HOST = $BindHost
	$env:TEACHER_PORT = "$Port"
	$apiProcess = Start-Background 'teacher-api' $ApiLog (Get-AppExecutable 'teacher-api')
	Wait-Http "http://127.0.0.1:$Port/healthz" '"ready"' 120 $apiProcess $ApiLog
	Test-TeacherSmoke "http://127.0.0.1:$Port" (Get-EnvValue 'TEACHER_API_TOKEN')
}

# Blocks until a background service exits, then fails.
function Wait-Background {
	while ($true) {
		foreach ($entry in $script:Background) {
			if ($entry.Process.HasExited) { Stop-WithError "$($entry.Name) exited unexpectedly" }
		}
		Start-Sleep -Seconds 5
	}
}

# Validates options shared by student-train.ps1 and distill-local.ps1 and returns the
# resolved terms review ID.
function Assert-PipelineOptions([hashtable]$Options) {
	if (-not $Options.TermsReviewId) { $Options.TermsReviewId = Get-EnvValue 'TEACHER_TERMS_REVIEW_ID' }
	if (-not $Options.TermsReviewId) {
		Stop-WithError 'set -TermsReviewId or TEACHER_TERMS_REVIEW_ID (see docs/distillation-pipeline.md)'
	}
	$buildCorpus = [bool]$Options.Corpus
	if ($buildCorpus) {
		if ($Options.Corpus -notin @('quick', 'standard', 'full')) { Stop-WithError '-Corpus must be quick, standard, or full' }
		if (-not $Options.References) {
			$Options.References = Join-Path (Split-Path $Options.Prompts -Parent) 'references.jsonl'
		}
		if ($Options.Config -eq 'configs/train.example.toml') { $Options.Config = 'configs/train.distill.toml' }
	}
	if (-not $Options.Reviews -and -not $Options.References -and -not $Options.AcceptPending) {
		Stop-WithError 'pass -References PATH or -Reviews PATH, or -AcceptPending for a plumbing test'
	}
	if (-not $buildCorpus -and $Options.References -and -not (Test-Path $Options.References -PathType Leaf)) {
		Stop-WithError "references file not found: $($Options.References)"
	}
	if ($Options.Reviews -and -not (Test-Path $Options.Reviews -PathType Leaf)) {
		Stop-WithError "reviews file not found: $($Options.Reviews)"
	}
	if (-not $buildCorpus -and -not (Test-Path $Options.Prompts -PathType Leaf)) { Stop-WithError "prompts file not found: $($Options.Prompts); pass -Corpus PROFILE or run just corpus" }
	if (-not (Test-Path $Options.Config -PathType Leaf)) { Stop-WithError "training config not found: $($Options.Config)" }
}

# Builds the prompt corpus for -Corpus unless the same profile is already in place. The build
# is deterministic, so a prompt keeps its example_id and text across rebuilds and profiles.
function Invoke-CorpusBuild([hashtable]$Options) {
	if (-not $Options.Corpus) { return }
	$directory = Split-Path $Options.Prompts -Parent
	$manifest = Join-Path $directory 'manifest.json'
	if ((Test-Path $manifest) -and (Test-Path $Options.Prompts) -and (Test-Path $Options.References)) {
		if ((Get-Content $manifest -Raw | ConvertFrom-Json).profile -eq $Options.Corpus) {
			Write-Log "using the existing $($Options.Corpus) prompt corpus in $directory"
			return
		}
	}
	Write-Log "building the $($Options.Corpus) prompt corpus into $directory (downloads public datasets)"
	Invoke-Checked (Get-AppExecutable 'distill') @('corpus', 'build', '--profile', $Options.Corpus, '--output-dir', $directory)
	if (-not ((Test-Path $Options.Prompts) -and (Test-Path $Options.References))) {
		Stop-WithError "corpus build did not write $($Options.Prompts) and $($Options.References)"
	}
}

function Invoke-Collection([hashtable]$Options, [string]$TeacherUrl, [string]$Token) {
	$arguments = @(
		'collect', '--input', $Options.Prompts, '--output', $Options.Ledger,
		'--teacher-url', $TeacherUrl, '--terms-review-id', $Options.TermsReviewId,
		'--concurrency', "$($Options.Concurrency)", '--max-output-tokens', "$($Options.MaxOutputTokens)", '--no-thinking'
	)
	if ($null -ne $Options.Seed) { $arguments += @('--seed', "$($Options.Seed)") }
	Write-Log "collecting teacher responses into $($Options.Ledger)"
	$previous = $env:TEACHER_API_TOKEN
	$env:TEACHER_API_TOKEN = $Token
	try { Invoke-Checked (Get-AppExecutable 'distill') $arguments }
	finally { $env:TEACHER_API_TOKEN = $previous }
}

# Validates the ledger against the answer key. Points Reviews at the automatic decisions.
function Invoke-Review([hashtable]$Options) {
	if (-not $Options.References) { return }
	$output = [System.IO.Path]::ChangeExtension($Options.Ledger, '.reviews.jsonl')
	$arguments = @(
		'review', '--ledger', $Options.Ledger, '--references', $Options.References,
		'--output', $output, '--config', $Options.Config
	)
	if ($Options.Reviews) { $arguments += @('--overrides', $Options.Reviews) }
	$image = Get-EnvValue 'REVIEW_CODE_IMAGE'
	if ($image) { $arguments += @('--code-image', $image) }
	Write-Log "reviewing teacher responses into $output"
	Invoke-Checked (Get-AppExecutable 'distill') $arguments
	$Options.Reviews = $output
}

function Invoke-PrepareAndTrain([hashtable]$Options) {
	Invoke-Review $Options
	$read = "import sys, tomllib; print(tomllib.load(open(sys.argv[1], 'rb'))['dataset_path'])"
	$dataset = (& (Get-AppExecutable 'python') -c $read $Options.Config)
	if ($LASTEXITCODE -ne 0) { Stop-WithError "could not read dataset_path from $($Options.Config)" }
	$arguments = @('prepare', '--input', $Options.Ledger, '--output', $dataset, '--split', 'train', '--prompts', $Options.Prompts)
	if ($Options.Reviews) { $arguments += @('--reviews', $Options.Reviews) }
	if ($Options.AcceptPending) { $arguments += '--accept-pending' }
	Write-Log "preparing training data into $dataset"
	Invoke-Checked (Get-AppExecutable 'distill') $arguments
	$read = "import sys, tomllib; print(tomllib.load(open(sys.argv[1], 'rb')).get('validation_dataset_path', ''))"
	$validationDataset = (& (Get-AppExecutable 'python') -c $read $Options.Config)
	if ($LASTEXITCODE -ne 0) { Stop-WithError "could not read validation_dataset_path from $($Options.Config)" }
	if ($validationDataset) {
		$arguments = @('prepare', '--input', $Options.Ledger, '--output', $validationDataset, '--split', 'validation', '--prompts', $Options.Prompts)
		if ($Options.Reviews) { $arguments += @('--reviews', $Options.Reviews) }
		if ($Options.AcceptPending) { $arguments += '--accept-pending' }
		Write-Log "preparing validation data into $validationDataset"
		Invoke-Checked (Get-AppExecutable 'distill') $arguments
	}
	Write-Log "training the student with $($Options.Config)"
	Invoke-Checked (Get-AppExecutable 'distill') @('train', '--config', $Options.Config)
}
