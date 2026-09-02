$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}
$python = (Get-Command $python -ErrorAction Stop).Source

# Prevent unrelated DLL directories from the developer's PATH from being
# collected into the frozen application. PyInstaller adds package-specific
# DLL directories itself while analyzing Qt, OpenCV, and Open3D.
$env:PATH = @(
    (Split-Path -Parent $python),
    (Join-Path $env:SystemRoot "System32"),
    $env:SystemRoot,
    (Join-Path $env:SystemRoot "System32\Wbem"),
    (Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0")
) -join ";"

$excludeArgs = @(
    "--exclude-module", "pytest",
    "--exclude-module", "IPython",
    "--exclude-module", "dash",
    "--exclude-module", "jupyter",
    "--exclude-module", "nbformat"
)
& $python scripts/run_pyinstaller_clean.py --noconfirm --clean --onedir --windowed --icon "gui/assets/app_icon.ico" --collect-data gs360studio --add-data "gui/assets;gui/assets" @excludeArgs --name "360GS Studio" gs360studio/gui_entry.py
if ($LASTEXITCODE -ne 0) { throw "GUI packaging failed with exit code $LASTEXITCODE." }
& $python scripts/run_pyinstaller_clean.py --noconfirm --clean --onedir --console --collect-data gs360studio @excludeArgs --name "360gs-studio" gs360studio/__main__.py
if ($LASTEXITCODE -ne 0) { throw "CLI packaging failed with exit code $LASTEXITCODE." }
Copy-Item -Path (Join-Path $root "dist\360gs-studio\_internal\*") -Destination (Join-Path $root "dist\360GS Studio\_internal") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $root "dist\360gs-studio\360gs-studio.exe") -Destination (Join-Path $root "dist\360GS Studio\360gs-studio.exe") -Force
foreach ($notice in @("LICENSE", "NOTICE.md", "THIRD_PARTY_LICENSES.md", "CHANGELOG.md")) {
    Copy-Item -LiteralPath (Join-Path $root $notice) -Destination (Join-Path $root "dist\360GS Studio\$notice") -Force
}
& $python scripts/build_sbom.py --output dist/sbom.cdx.json
& (Join-Path $root "dist\360GS Studio\360gs-studio.exe") --version
if ($LASTEXITCODE -ne 0) { throw "Packaged CLI smoke test failed with exit code $LASTEXITCODE." }
$previousQtPlatform = $env:QT_QPA_PLATFORM
$env:QT_QPA_PLATFORM = "offscreen"
$guiProcess = Start-Process -FilePath (Join-Path $root "dist\360GS Studio\360GS Studio.exe") -ArgumentList "--smoke-test" -PassThru
if (-not $guiProcess.WaitForExit(30000)) {
    Stop-Process -Id $guiProcess.Id -Force
    $env:QT_QPA_PLATFORM = $previousQtPlatform
    throw "Packaged GUI smoke test timed out."
}
$env:QT_QPA_PLATFORM = $previousQtPlatform
if ($guiProcess.ExitCode -ne 0) {
    throw "Packaged GUI smoke test failed with exit code $($guiProcess.ExitCode)."
}

Write-Host "Built Windows folders under dist/. Use Inno Setup with packaging/360GS-Studio.iss for the installer."
