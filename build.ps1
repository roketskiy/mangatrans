# Build mangatrans with the mingw-w64 toolchain required by cgo on Windows.
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$mingw = Join-Path $root 'runtime\toolchain\mingw64\mingw64\bin'

if (-not (Test-Path $mingw)) {
    Write-Error "mingw-w64 not found at $mingw"
}

$env:CGO_ENABLED = '1'
$env:CC = Join-Path $mingw 'gcc.exe'
$env:PATH = $mingw + ';' + $env:PATH

Push-Location $root
try {
    go build -o (Join-Path $root 'bin\mangatrans.exe') .\cmd\mangatrans
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Output ('built ' + (Join-Path $root 'bin\mangatrans.exe'))
} finally {
    Pop-Location
}
