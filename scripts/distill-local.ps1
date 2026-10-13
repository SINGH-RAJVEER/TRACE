#Requires -Version 7.2
<#
.SYNOPSIS
Scenario 2, single machine: start the teacher, collect, stop the teacher to free GPU memory,
then prepare the dataset and train the student.

.DESCRIPTION
Runs the whole distillation on this machine. The teacher API binds to 127.0.0.1. Paths are
relative to the repository root. Pass -Reviews or, for a plumbing test only, -AcceptPending.
vLLM does not run natively on Windows; pass -UpstreamUrl there. vLLM settings: VLLM_PORT,
VLLM_MAX_MODEL_LEN, VLLM_GPU_MEMORY_UTILIZATION, VLLM_EXTRA_ARGS, VLLM_SPEC, VLLM_PYTHON,
VLLM_BIN, VLLM_STARTUP_TIMEOUT. See docs/run-scripts.md.

.PARAMETER Port
Teacher API port. Defaults to TEACHER_PORT or 8000.

.PARAMETER UpstreamUrl
Use an existing OpenAI-compatible server instead of starting vLLM.

.PARAMETER TermsReviewId
Approval reference written into every ledger record. Defaults to TEACHER_TERMS_REVIEW_ID.
#>
[CmdletBinding()]
param(
	[int]$Port = 0,
	[string]$UpstreamUrl = '',
	[string]$TermsReviewId = '',
	[string]$Reviews = '',
	[switch]$AcceptPending,
	[string]$Prompts = 'data/examples/prompts.jsonl',
	[string]$Ledger = 'data/teacher/ledger.jsonl',
	[string]$Config = 'configs/train.example.toml',
	[int]$Concurrency = 4,
	[int]$MaxOutputTokens = 1024,
	[Nullable[int]]$Seed = $null
)

$script:ScriptName = 'distill-local'
. (Join-Path $PSScriptRoot 'lib/Common.ps1')
Set-Location $RepoRoot
Import-DotEnv
if ($Port -eq 0) { $Port = [int](Get-EnvValue 'TEACHER_PORT' '8000') }

$options = @{
	TermsReviewId = $TermsReviewId
	Reviews = $Reviews
	AcceptPending = [bool]$AcceptPending
	Prompts = $Prompts
	Ledger = $Ledger
	Config = $Config
	Concurrency = $Concurrency
	MaxOutputTokens = $MaxOutputTokens
	Seed = $Seed
}
Assert-PipelineOptions $options
# The teacher API refuses requests until it has an approval reference.
$env:TEACHER_TERMS_REVIEW_ID = $options.TermsReviewId

Select-Runtime -NeedsGpu -NeedsVllm:(-not $UpstreamUrl)
Initialize-AppEnvironment -Train
try {
	Start-TeacherStack '127.0.0.1' $Port $UpstreamUrl (Join-Path $LogDir 'teacher-api.log')
	Invoke-Collection $options "http://127.0.0.1:$Port" (Get-EnvValue 'TEACHER_API_TOKEN')
	Write-Log 'stopping the teacher to free GPU memory for training'
}
finally {
	Stop-Background
}
Invoke-PrepareAndTrain $options
Write-Log "done; the adapter and training manifest are in the config's output_dir"
