#Requires -Version 7.2
<#
.SYNOPSIS
Scenario 1, teacher machine: serve Qwen3.5-9B with vLLM on loopback and expose the teacher API
on the network for a student machine to collect from.

.DESCRIPTION
Starts vLLM and the teacher API, then runs until interrupted. Requires TEACHER_TERMS_REVIEW_ID,
and TEACHER_API_TOKEN when binding beyond loopback. vLLM does not run natively on Windows; pass
-UpstreamUrl there. vLLM settings: VLLM_PORT, VLLM_MAX_MODEL_LEN, VLLM_GPU_MEMORY_UTILIZATION,
VLLM_EXTRA_ARGS, VLLM_SPEC, VLLM_PYTHON, VLLM_BIN, VLLM_STARTUP_TIMEOUT. See docs/run-scripts.md.

.PARAMETER BindHost
Teacher API bind address. Defaults to 0.0.0.0.

.PARAMETER Port
Teacher API port. Defaults to TEACHER_PORT or 8000.

.PARAMETER UpstreamUrl
Use an existing OpenAI-compatible server instead of starting vLLM.
#>
[CmdletBinding()]
param(
	[string]$BindHost = '0.0.0.0',
	[int]$Port = 0,
	[string]$UpstreamUrl = ''
)

$script:ScriptName = 'teacher-host'
. (Join-Path $PSScriptRoot 'lib/Common.ps1')
Set-Location $RepoRoot
Import-DotEnv
if ($Port -eq 0) { $Port = [int](Get-EnvValue 'TEACHER_PORT' '8000') }

if (-not (Get-EnvValue 'TEACHER_TERMS_REVIEW_ID')) {
	Stop-WithError 'set TEACHER_TERMS_REVIEW_ID in .env; the teacher API refuses requests without it'
}
if ($BindHost -notin @('127.0.0.1', 'localhost') -and -not (Get-EnvValue 'TEACHER_API_TOKEN')) {
	Stop-WithError ("set TEACHER_API_TOKEN before binding to $BindHost (generate one with: " +
		'[Convert]::ToHexString([Security.Cryptography.RandomNumberGenerator]::GetBytes(32)))')
}

Select-Runtime -NeedsGpu:(-not $UpstreamUrl) -NeedsVllm:(-not $UpstreamUrl)
Initialize-AppEnvironment
try {
	Start-TeacherStack $BindHost $Port $UpstreamUrl ''
	$address = [Net.Dns]::GetHostAddresses([Net.Dns]::GetHostName()) |
		Where-Object { $_.AddressFamily -eq 'InterNetwork' -and -not [Net.IPAddress]::IsLoopback($_) } |
		Select-Object -First 1
	$display = if ($address) { $address.ToString() } else { '<this-host>' }
	Write-Log "teacher API ready on $($BindHost):$Port"
	Write-Log 'on the student machine run:'
	Write-Log "  scripts/student-train.ps1 -TeacherUrl http://$($display):$Port -Reviews <file>"
	Write-Log 'with the same TEACHER_API_TOKEN set on that machine'
	Write-Log 'press Ctrl+C to stop'
	Wait-Background
}
finally {
	Stop-Background
}
