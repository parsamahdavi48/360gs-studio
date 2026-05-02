@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] .venv was not found. Running setup_windows.bat first...
    call setup_windows.bat
    if errorlevel 1 (
        echo [ERROR] Setup failed. The GUI cannot be started.
        echo.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -m gui.app %*
