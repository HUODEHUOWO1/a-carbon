param(
  [string]$OutDir = ".\\data\\vision_proxy"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$tgz = Join-Path $OutDir "imagenette2-160.tgz"
$dst = Join-Path $OutDir "imagenette2-160"
Invoke-WebRequest -Uri "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz" -OutFile $tgz
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
# Requires tar on Windows 10+/PowerShell environment
& tar -xzf $tgz -C $OutDir
Write-Host "Extracted to $OutDir"
