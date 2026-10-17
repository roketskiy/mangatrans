# Run the full 11-page test book and write a stage-timing JSON next to the output.
param(
    [string]$In = "E:\edge_download\少年ジャンプ＋ (Shonen Jump +)\11歳になってしまう\11歳になってしまう.cbz",
    [string]$Out = ".cache\out\bench.cbz"
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$engine = Join-Path $root 'bin\mangatrans.exe'
if (-not (Test-Path $engine)) {
    Write-Error "engine not found: $engine (run build.ps1 first)"
}

$outPath = Join-Path $root $Out
$benchPath = [System.IO.Path]::ChangeExtension($outPath, '.json')
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outPath) | Out-Null

& $engine -in $In -out $outPath -bench-json $benchPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output "benchmark json: $benchPath"
