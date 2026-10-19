$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$dir = Join-Path $root '.cache/settings/engine-check'
New-Item -ItemType Directory -Path $dir -Force | Out-Null
$env:CGO_ENABLED = '1'
$env:CC = Join-Path $root 'runtime/toolchain/mingw64/mingw64/bin/gcc.exe'
$env:PATH = (Split-Path -Parent $env:CC) + [IO.Path]::PathSeparator + $env:PATH
$env:GOTMPDIR = $dir
$env:HTTP_PROXY = 'http://127.0.0.1:7897'
$env:HTTPS_PROXY = 'http://127.0.0.1:7897'
Push-Location $root
try {
    & go build -o (Join-Path $dir 'mangatrans.exe') ./cmd/mangatrans
    if ($LASTEXITCODE -ne 0) { throw 'Engine build failed' }
    & go vet ./cmd/mangatrans ./internal/render ./internal/detect ./internal/inpaint ./internal/translate
    if ($LASTEXITCODE -ne 0) { throw 'Engine static checks failed' }
    Write-Output '[OK] Engine build completed'
} finally {
    Pop-Location
}
