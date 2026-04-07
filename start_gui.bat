@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    call setup_windows.bat
)

call ".venv\Scripts\activate.bat"
python -m gui.app %*
