$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$artifacts = Join-Path $root '.cache/settings'
$paths = @(
    (Join-Path $artifacts 'engine-check/go-build1276611263'),
    'C:/Users/Public/Temp/go-build558447868',
    (Join-Path $artifacts 'security-db')
)
$results = @()
foreach ($path in $paths) {
    $status = 'already absent'
    if (Test-Path -LiteralPath $path) {
        $item = Get-Item -LiteralPath $path -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'Refusing to remove a reparse point.'
        }
        Remove-Item -LiteralPath $path -Recurse -Force
        $status = 'removed'
    }
    $results += [ordered]@{ path = $path; status = $status; absent = !(Test-Path -LiteralPath $path) }
}
$running = @(Get-Process -Name 'translate.test' -ErrorAction SilentlyContinue)
$report = [ordered]@{
    time = (Get-Date).ToString('o')
    paths = $results
    runningTestProcessCount = $running.Count
    quarantine = 'Not restored, modified, or emptied. Managed by antivirus.'
}
$json = $report | ConvertTo-Json -Depth 10
Set-Content -LiteralPath (Join-Path $artifacts 'cleanup-report.json') -Value $json -Encoding utf8
Write-Output $json
