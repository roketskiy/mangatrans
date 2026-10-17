# Build the MyGo desktop shell with the Go 1.27 toolchain it requires.
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$go = Join-Path $root '.cache\go1.27\go\bin\go.exe'
$src = Join-Path $root 'desktop'

if (-not (Test-Path $go)) {
    Write-Error "Go 1.27 not found at $go"
}
if (-not (Test-Path (Join-Path $root 'bin\mangatrans.exe'))) {
    Write-Warning 'bin\mangatrans.exe is missing; build the engine first with build.ps1'
}

$env:CGO_ENABLED = '0'
Push-Location $src
try {
    & $go build -o (Join-Path $root 'bin\mangatrans-ui.exe') .
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Output ('built ' + (Join-Path $root 'bin\mangatrans-ui.exe'))
} finally {
    Pop-Location
}
