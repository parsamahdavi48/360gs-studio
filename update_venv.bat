@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "UPDATER_PY="
set "UPDATER_ARGS="
set "PAUSE_ON_EXIT=1"

:parse_args
if "%~1"=="" goto detect_python
if /I "%~1"=="--no-pause" (
    set "PAUSE_ON_EXIT=0"
    shift
    goto parse_args
)
if /I "%~1"=="--pause" (
    set "PAUSE_ON_EXIT=1"
    shift
    goto parse_args
)
set "UPDATER_ARGS=!UPDATER_ARGS! ^"%~1^""
shift
goto parse_args

:detect_python
where py >nul 2>&1
if not errorlevel 1 (
    for %%V in (3.14 3.13 3.12 3.11) do (
        if not defined UPDATER_PY (
            py -%%V -c "import sys" >nul 2>&1
            if !errorlevel! equ 0 set "UPDATER_PY=py -%%V"
        )
    )
)

if not defined UPDATER_PY if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    set "UPDATER_PY="%LocalAppData%\Programs\Python\Python312\python.exe""
)

if not defined UPDATER_PY (
    where python >nul 2>&1
    if not errorlevel 1 set "UPDATER_PY=python"
)

if not defined UPDATER_PY (
    echo [ERROR] No Python command was found to run scripts\update_venv.py.
    echo [INFO] Install Python 3.12 or newer, then retry.
    set "EXIT_CODE=1"
    goto finish
)

%UPDATER_PY% "%~dp0scripts\update_venv.py" %UPDATER_ARGS%
set "EXIT_CODE=%errorlevel%"

:finish
echo.
if "%EXIT_CODE%"=="0" (
    echo [DONE] update_venv.bat completed successfully.
) else (
    echo [ERROR] update_venv.bat failed with exit code %EXIT_CODE%.
)
echo [INFO] Log file: "%~dp0.cache\update_venv.log"
if "%PAUSE_ON_EXIT%"=="1" (
    echo.
    pause
)
exit /b %EXIT_CODE%
