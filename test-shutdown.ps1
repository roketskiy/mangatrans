# Start the engine hidden, wait for its llama-server child, then run the same
# taskkill /T /F command the desktop shell uses on Stop and window close.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$engine = Join-Path $root 'bin\mangatrans.exe'
$inPath = 'E:\edge_download\少年ジャンプ＋ (Shonen Jump +)\11歳になってしまう\11歳になってしまう.cbz'
$outPath = Join-Path $root '.cache\out\shutdown-test.cbz'
$resultPath = Join-Path $root '.cache\out\shutdown-test.json'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outPath) | Out-Null

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $engine
$psi.WorkingDirectory = Join-Path $root 'bin'
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
foreach ($a in @('-in', $inPath, '-out', $outPath)) {
    $psi.ArgumentList.Add($a)
}

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
[void]$proc.Start()
$enginePid = $proc.Id
$errTask = $proc.StandardError.ReadToEndAsync()
$outTask = $proc.StandardOutput.ReadToEndAsync()

$childPids = @()
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 500
    if ($proc.HasExited) { break }
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$enginePid" -ErrorAction SilentlyContinue)
    $llama = @($children | Where-Object { $_.Name -like 'llama-server*' })
    if ($llama.Count -gt 0) {
        $childPids = @($llama | ForEach-Object { $_.ProcessId })
        break
    }
}

$foundChild = $childPids.Count -gt 0
$tkPsi = New-Object System.Diagnostics.ProcessStartInfo
$tkPsi.FileName = 'taskkill'
$tkPsi.UseShellExecute = $false
$tkPsi.CreateNoWindow = $true
foreach ($a in @('/PID', [string]$enginePid, '/T', '/F')) {
    $tkPsi.ArgumentList.Add($a)
}
$tk = New-Object System.Diagnostics.Process
$tk.StartInfo = $tkPsi
[void]$tk.Start()
$tk.WaitForExit()

Start-Sleep -Seconds 2

$engineAlive = $null -ne (Get-Process -Id $enginePid -ErrorAction SilentlyContinue)
$childrenAlive = @()
foreach ($pidValue in $childPids) {
    if (Get-Process -Id $pidValue -ErrorAction SilentlyContinue) {
        $childrenAlive += $pidValue
    }
}

$result = [ordered]@{
    engine_pid             = $enginePid
    child_pids             = $childPids
    found_llama_child      = $foundChild
    taskkill_exit          = $tk.ExitCode
    engine_alive_after     = $engineAlive
    children_alive_after   = $childrenAlive
    passed                 = $foundChild -and (-not $engineAlive) -and ($childrenAlive.Count -eq 0)
}
$result | ConvertTo-Json -Depth 5 | Set-Content -Path $resultPath -Encoding UTF8
$result | Format-List

if (-not $proc.HasExited) {
    [void]$proc.WaitForExit(3000)
}
$null = $errTask.Result
$null = $outTask.Result
Remove-Item -LiteralPath $outPath -ErrorAction SilentlyContinue
Write-Output "result json: $resultPath"
