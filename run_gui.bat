@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

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

call :add_winget_ffmpeg_to_path

".venv\Scripts\python.exe" -m gui.app %*
exit /b %ERRORLEVEL%

:add_winget_ffmpeg_to_path
where ffmpeg >nul 2>&1
set "FFMPEG_FOUND=!errorlevel!"
where ffprobe >nul 2>&1
set "FFPROBE_FOUND=!errorlevel!"
if "!FFMPEG_FOUND!"=="0" if "!FFPROBE_FOUND!"=="0" exit /b 0

if exist "%LocalAppData%\Microsoft\WinGet\Links" (
    set "PATH=%LocalAppData%\Microsoft\WinGet\Links;%PATH%"
)
set "FFMPEG_BIN="
if exist "%LocalAppData%\Microsoft\WinGet\Packages" (
    for /f "delims=" %%F in ('where /R "%LocalAppData%\Microsoft\WinGet\Packages" ffmpeg.exe 2^>nul') do (
        if not defined FFMPEG_BIN set "FFMPEG_BIN=%%~dpF"
    )
)
if defined FFMPEG_BIN set "PATH=!FFMPEG_BIN!;%PATH%"
exit /b 0
