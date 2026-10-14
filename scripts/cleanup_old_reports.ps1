$ErrorActionPreference = 'Stop'

# Scope: old experiment media and archives only; retain metrics, logs and models.
$projectDirectory = Split-Path -Parent $PSScriptRoot
$reportsDirectory = (Resolve-Path -LiteralPath (Join-Path $projectDirectory 'reports')).ProviderPath
$oldNames = @(
    'chair_warmup_20261008', 'chair_rl_20261008', 'chair_official_20261008',
    'chair_asg_20261008', 'still2_warmup_20261008', 'chair_sg_v2_20261008',
    'chair_reference_20261008', 'llnerf_random3_20261008', 'chair_sg_20261008'
)
$pattern = '(?i)\.(png|jpg|jpeg|webp|gif|mp4|avi|zip|7z|rar|tar|gz|tgz|bz2|tbz2|xz|txz|zst)$'
$targets = @(
    foreach ($name in $oldNames) {
        $directory = Join-Path $reportsDirectory $name
        if (Test-Path -LiteralPath $directory -PathType Container) {
            $resolvedDirectory = (Resolve-Path -LiteralPath $directory).ProviderPath
            if ($resolvedDirectory -ne $directory) { throw "Unexpected directory: $directory" }
            $directoryItems = @(Get-ChildItem -LiteralPath $resolvedDirectory -Recurse -Force)
            if ($directoryItems | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) {
                throw "Refuse to delete through a linked path: $directory"
            }
            $directoryItems | Where-Object { -not $_.PSIsContainer -and $_.Name -match $pattern }
        }
    }
)

$deleted = 0
$freedBytes = 0L
foreach ($target in $targets) {
    $resolvedFile = (Resolve-Path -LiteralPath $target.FullName).ProviderPath
    if (-not $resolvedFile.StartsWith($reportsDirectory + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target outside reports: $resolvedFile"
    }
    $size = $target.Length
    Remove-Item -LiteralPath $resolvedFile -Force
    $deleted++
    $freedBytes += $size
}
Write-Host ('Deleted {0} old media/archive files; released {1:N2} GiB.' -f $deleted, ($freedBytes / 1GB))
