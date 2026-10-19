param([int]$CommandTimeoutSeconds = 120)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$artifacts = Join-Path $root '.cache/settings/checks'
New-Item -ItemType Directory -Force -Path $artifacts | Out-Null
$go = (Get-Command go -ErrorAction Stop).Source
$desktopGo = Join-Path $root '.cache/go1.27/go/bin/go.exe'
$gcc = Join-Path $root 'runtime/toolchain/mingw64/mingw64/bin/gcc.exe'
$checks = [Collections.Generic.List[object]]::new()

function Invoke-Check {
    param([string]$Name, [string]$Executable, [string]$Directory, [string[]]$Arguments)
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Executable
    $info.WorkingDirectory = $Directory
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($argument in $Arguments) { $info.ArgumentList.Add($argument) }
    $info.Environment['HTTP_PROXY'] = 'http://127.0.0.1:7897'
    $info.Environment['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
    $info.Environment['CGO_ENABLED'] = '1'
    $info.Environment['CC'] = $gcc
    $info.Environment['PATH'] = (Split-Path -Parent $gcc) + [IO.Path]::PathSeparator + $env:PATH
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    if (!$process.Start()) { throw "Could not start $Name" }
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    if (!$process.WaitForExit($CommandTimeoutSeconds * 1000)) {
        $process.Kill($true)
        $process.WaitForExit()
        throw "$Name exceeded its time limit"
    }
    $text = $stdout.GetAwaiter().GetResult() + $stderr.GetAwaiter().GetResult()
    Set-Content -LiteralPath (Join-Path $artifacts ($Name + '.log')) -Value $text -Encoding utf8
    $checks.Add([ordered]@{ name = $Name; exitCode = $process.ExitCode; passed = ($process.ExitCode -eq 0) })
    Write-Output "[$Name] exit $($process.ExitCode)"
    if ($process.ExitCode -ne 0) { Write-Output $text }
    $process.Dispose()
}

if (!(Test-Path -LiteralPath $desktopGo)) { throw 'The documented desktop Go toolchain is missing.' }
if (!(Test-Path -LiteralPath $gcc)) { throw 'The documented engine compiler is missing.' }

Invoke-Check 'settings' $go $root @('run', '-mod=readonly', './cmd/verify-settings')
Invoke-Check 'codecs' $go $root @('run', '-mod=readonly', './cmd/verify-pageimage')
Invoke-Check 'engine-build' $go $root @('build', '-mod=readonly', '-o', (Join-Path $artifacts 'mangatrans.exe'), './cmd/mangatrans')
Invoke-Check 'desktop-build' $desktopGo (Join-Path $root 'desktop') @('build', '-mod=readonly', '-ldflags=-H=windowsgui', '-o', (Join-Path $artifacts 'mangatrans-ui.exe'), '.')
Invoke-Check 'engine-vet' $go $root @('vet', './cmd/mangatrans', './internal/render', './internal/inpaint', './internal/detect', './internal/translate', './settings', './internal/pageimage')
Invoke-Check 'desktop-vet' $desktopGo (Join-Path $root 'desktop') @('vet', '.')

$failed = @($checks | Where-Object { !$_.passed })
$report = [ordered]@{ time = (Get-Date).ToString('o'); passed = ($failed.Count -eq 0); checks = $checks; uiLaunched = $false; goTestRun = $false }
$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $artifacts 'report.json') -Encoding utf8
if ($failed.Count -gt 0) { exit 1 }
