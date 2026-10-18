[CmdletBinding()]
param(
    [ValidateRange(1, 1000)]
    [int]$Runs = 1,
    [string]$OutputDir,
    [ValidateRange(1, 86400)]
    [int]$TimeoutSeconds = 3600
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$inputFile = Join-Path $repoRoot 'TestResourse\Training set\彼方のアストラ\1話.cbz'
if (-not (Test-Path -LiteralPath $inputFile -PathType Leaf)) {
    throw "Input file does not exist: $inputFile"
}
if (-not $OutputDir) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fffffff'
    $OutputDir = Join-Path $repoRoot ".cache\bench\astra-1-$stamp"
}

$benchArgs = @(
    (Join-Path $PSScriptRoot 'bench.py'),
    $inputFile,
    '--runs', [string]$Runs,
    '--limit', '0',
    '--timeout', [string]$TimeoutSeconds,
    '--output-dir', $OutputDir
)

Write-Host "Input: $inputFile"
Write-Host "Processing all images, $Runs run(s). Report directory: $OutputDir"
& python @benchArgs
exit $LASTEXITCODE
