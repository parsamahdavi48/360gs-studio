@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] .venv not found. Running setup_windows.bat...
    call setup_windows.bat
    if errorlevel 1 (
        echo [ERROR] setup_windows.bat failed.
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate .venv
    exit /b 1
)

echo [INFO] Launching extract_frames_gui.py
python extract_frames_gui.py %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo [ERROR] GUI exited with code %RC%
)
exit /b %RC%
