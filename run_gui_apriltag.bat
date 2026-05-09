@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "STECHDRIVE_ENABLE_APRILTAG=1"

call run_gui.bat --enable-apriltag %*
