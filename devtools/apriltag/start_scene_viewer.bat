@echo off
setlocal

set "ROOT=%~dp0..\.."
pushd "%ROOT%" >nul
if errorlevel 1 exit /b 1

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\dev_apriltag_scene_viewer.py %*
) else (
    python scripts\dev_apriltag_scene_viewer.py %*
)
set "EXITCODE=%ERRORLEVEL%"

popd >nul
exit /b %EXITCODE%
