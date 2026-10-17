# Run the 11-page test book without any console window, and record stage
# timings, peak working set, and peak GPU memory/utilisation.
param(
    [string]$In = "E:\edge_download\少年ジャンプ＋ (Shonen Jump +)\11歳になってしまう\11歳になってしまう.cbz",
    [string]$Out = ".cache\out\bench-resource.cbz"
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$engine = Join-Path $root 'bin\mangatrans.exe'
if (-not (Test-Path $engine)) {
    Write-Error "engine not found: $engine (run build.ps1 first)"
}

$outPath = Join-Path $root $Out
$benchPath = [System.IO.Path]::ChangeExtension($outPath, '.json')
$resPath = [System.IO.Path]::ChangeExtension($outPath, '.resources.json')
$logPath = [System.IO.Path]::ChangeExtension($outPath, '.log')
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outPath) | Out-Null

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $engine
$psi.WorkingDirectory = Join-Path $root 'bin'
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
foreach ($a in @('-in', $In, '-out', $outPath, '-bench-json', $benchPath)) {
    $psi.ArgumentList.Add($a)
}

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
[void]$proc.Start()
$enginePid = $proc.Id
$stdoutTask = $proc.StandardOutput.ReadToEndAsync()
$stderrTask = $proc.StandardError.ReadToEndAsync()

$peakEngineMB = 0.0
$peakLlamaMB = 0.0
$peakTotalMB = 0.0
$peakGpuMB = 0.0
$peakGpuUtil = 0.0
$lastEngineCPU = 0.0
$lastLlamaCPU = 0.0
$samples = 0
$start = Get-Date

while (-not $proc.HasExited) {
    $engineMB = 0.0
    $llamaMB = 0.0

    $ep = Get-Process -Id $enginePid -ErrorAction SilentlyContinue
    if ($ep) {
        $engineMB = $ep.WorkingSet64 / 1MB
        $lastEngineCPU = $ep.CPU
    }

    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$enginePid" -ErrorAction SilentlyContinue)
    $llamaCPU = 0.0
    foreach ($child in $children) {
        if ($child.Name -like 'llama-server*') {
            $cp = Get-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
            if ($cp) {
                $llamaMB += $cp.WorkingSet64 / 1MB
                $llamaCPU += $cp.CPU
            }
        }
    }
    if ($llamaCPU -gt $lastLlamaCPU) { $lastLlamaCPU = $llamaCPU }

    $totalMB = $engineMB + $llamaMB
    if ($engineMB -gt $peakEngineMB) { $peakEngineMB = $engineMB }
    if ($llamaMB -gt $peakLlamaMB) { $peakLlamaMB = $llamaMB }
    if ($totalMB -gt $peakTotalMB) { $peakTotalMB = $totalMB }

    $gpuLine = & nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($gpuLine) {
        $parts = $gpuLine -split ','
        if ($parts.Count -ge 2) {
            $gpuMB = [double]$parts[0].Trim()
            $gpuUtil = [double]$parts[1].Trim()
            if ($gpuMB -gt $peakGpuMB) { $peakGpuMB = $gpuMB }
            if ($gpuUtil -gt $peakGpuUtil) { $peakGpuUtil = $gpuUtil }
        }
    }

    $samples++
    Start-Sleep -Milliseconds 1000
}

$proc.WaitForExit()
$wall = ((Get-Date) - $start).TotalSeconds
Set-Content -Path $logPath -Value $stderrTask.Result -Encoding UTF8
$null = $stdoutTask.Result

$bench = $null
if (Test-Path $benchPath) {
    $bench = Get-Content $benchPath -Raw | ConvertFrom-Json
}

$result = [ordered]@{
    pages               = if ($bench) { $bench.pages } else { 0 }
    blocks              = if ($bench) { $bench.blocks } else { 0 }
    chars               = if ($bench) { $bench.chars } else { 0 }
    total_ms            = if ($bench) { $bench.total_ms } else { 0 }
    detect_ms           = if ($bench) { $bench.detect_ms } else { 0 }
    ocr_ms              = if ($bench) { $bench.ocr_ms } else { 0 }
    translate_ms        = if ($bench) { $bench.translate_ms } else { 0 }
    inpaint_ms          = if ($bench) { $bench.inpaint_ms } else { 0 }
    render_ms           = if ($bench) { $bench.render_ms } else { 0 }
    encode_ms           = if ($bench) { $bench.encode_ms } else { 0 }
    validate_ms         = if ($bench) { $bench.validate_ms } else { 0 }
    load_ocr_ms         = if ($bench) { $bench.load_ocr_ms } else { 0 }
    load_det_ms         = if ($bench) { $bench.load_det_ms } else { 0 }
    load_mt_ms          = if ($bench) { $bench.load_mt_ms } else { 0 }
    load_lama_ms        = if ($bench) { $bench.load_lama_ms } else { 0 }
    wall_s              = [math]::Round($wall, 1)
    peak_engine_ws_mb   = [math]::Round($peakEngineMB, 1)
    peak_llama_ws_mb    = [math]::Round($peakLlamaMB, 1)
    peak_total_ws_mb    = [math]::Round($peakTotalMB, 1)
    peak_gpu_mem_mb     = [math]::Round($peakGpuMB, 1)
    peak_gpu_util_pct   = [math]::Round($peakGpuUtil, 1)
    cpu_total_s         = [math]::Round($lastEngineCPU + $lastLlamaCPU, 1)
    cpu_avg_cores       = [math]::Round(($lastEngineCPU + $lastLlamaCPU) / [math]::Max($wall, 0.1), 2)
    samples             = $samples
    exit_code           = $proc.ExitCode
}

$result | ConvertTo-Json -Depth 5 | Set-Content -Path $resPath -Encoding UTF8
$result | Format-List
Write-Output "resources json: $resPath"
Write-Output "benchmark json: $benchPath"
