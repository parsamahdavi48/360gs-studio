@echo off
setlocal

cd /d "%~dp0"
if errorlevel 1 exit /b 1

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\dev_scene_preview_viewer.py %*
) else (
    python scripts\dev_scene_preview_viewer.py %*
)
exit /b %ERRORLEVEL%
