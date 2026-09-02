$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$excludeArgs = @(
    "--exclude-module", "pytest",
    "--exclude-module", "IPython",
    "--exclude-module", "dash",
    "--exclude-module", "jupyter",
    "--exclude-module", "nbformat"
)
& $python -m PyInstaller --noconfirm --clean --onedir --windowed --collect-data gs360studio --add-data "gui/assets;gui/assets" @excludeArgs --name "360GS Studio" gs360studio/gui_entry.py
& $python -m PyInstaller --noconfirm --clean --onedir --console --collect-data gs360studio @excludeArgs --name "360gs-studio" gs360studio/__main__.py
Copy-Item -LiteralPath (Join-Path $root "dist\360gs-studio\360gs-studio.exe") -Destination (Join-Path $root "dist\360GS Studio\360gs-studio.exe") -Force
& $python scripts/build_sbom.py --output dist/sbom.cdx.json
& (Join-Path $root "dist\360GS Studio\360gs-studio.exe") --version

Write-Host "Built Windows folders under dist/. Use Inno Setup with packaging/360GS-Studio.iss for the installer."
