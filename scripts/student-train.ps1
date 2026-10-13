#Requires -Version 7.2
<#
.SYNOPSIS
Scenario 1, student machine: collect from a remote teacher API, prepare the dataset, and train
the Qwen3-4B QLoRA adapter.

.DESCRIPTION
Runs collection against a teacher API on another machine, then preparation and training.
Paths are relative to the repository root. Pass -Reviews or, for a plumbing test only,
-AcceptPending.

.PARAMETER TeacherUrl
Teacher API base URL, for example http://10.0.0.5:8000.

.PARAMETER Token
Teacher API bearer token. Defaults to TEACHER_API_TOKEN.

.PARAMETER TermsReviewId
Approval reference written into every ledger record. Defaults to TEACHER_TERMS_REVIEW_ID.
#>
[CmdletBinding()]
param(
	[Parameter(Mandatory)][string]$TeacherUrl,
	[string]$Token = '',
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

$script:ScriptName = 'student-train'
. (Join-Path $PSScriptRoot 'lib/Common.ps1')
Set-Location $RepoRoot
Import-DotEnv

$TeacherUrl = $TeacherUrl.TrimEnd('/')
if (-not $Token) { $Token = Get-EnvValue 'TEACHER_API_TOKEN' }
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

Select-Runtime -NeedsGpu
Initialize-AppEnvironment -Train
Wait-Http "$TeacherUrl/healthz" '"ready"' 60
Test-TeacherSmoke $TeacherUrl $Token
Invoke-Collection $options $TeacherUrl $Token
Invoke-PrepareAndTrain $options
Write-Log "done; the adapter and training manifest are in the config's output_dir"
