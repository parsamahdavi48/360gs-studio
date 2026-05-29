@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PAUSE_ON_EXIT=1"
set "DO_APP=1"
set "DO_DEPS=1"
set "DRY_RUN=0"
set "SKIP_DEP_CHECK=0"
set "APP_ARGS="
set "SYNC_ARGS="
set "CHECK_ARGS=--locked"
set "PREFLIGHT_ARGS=--dry-run --locked --candidates 3.12 --no-install-python"
set "PYTHON_CMD="
set "EXIT_CODE=0"
set "UPDATE_RESULT="

:parse_args
if "%~1"=="" goto main
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
if /I "%~1"=="--app-only" (
    set "DO_DEPS=0"
    shift
    goto parse_args
)
if /I "%~1"=="--deps-only" (
    set "DO_APP=0"
    shift
    goto parse_args
)
if /I "%~1"=="--latest-deps" (
    set "SYNC_ARGS=!SYNC_ARGS! --latest"
    set "SKIP_DEP_CHECK=1"
    shift
    goto parse_args
)
if /I "%~1"=="--allow-cpu-torch" (
    set "SYNC_ARGS=!SYNC_ARGS! --allow-cpu-torch"
    set "CHECK_ARGS=!CHECK_ARGS! --allow-cpu-torch"
    set "PREFLIGHT_ARGS=!PREFLIGHT_ARGS! --allow-cpu-torch"
    shift
    goto parse_args
)
if /I "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    set "APP_ARGS=!APP_ARGS! --dry-run"
    set "SYNC_ARGS=!SYNC_ARGS! --dry-run"
    shift
    goto parse_args
)
if /I "%~1"=="--force" (
    set "APP_ARGS=!APP_ARGS! --force"
    shift
    goto parse_args
)
if /I "%~1"=="--require-sha256" (
    set "APP_ARGS=!APP_ARGS! --require-sha256"
    shift
    goto parse_args
)
if /I "%~1"=="--allow-dev-checkout" (
    set "APP_ARGS=!APP_ARGS! --allow-dev-checkout"
    shift
    goto parse_args
)
if /I "%~1"=="--version" (
    if "%~2"=="" (
        echo [ERROR] --version requires a value.
        set "EXIT_CODE=1"
        goto finish
    )
    set "APP_ARGS=!APP_ARGS! --version ^"%~2^""
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--zip" (
    if "%~2"=="" (
        echo [ERROR] --zip requires a value.
        set "EXIT_CODE=1"
        goto finish
    )
    set "APP_ARGS=!APP_ARGS! --zip ^"%~2^""
    shift
    shift
    goto parse_args
)
echo [WARN] Ignoring unknown update option: %~1
shift
goto parse_args

:main
if "%DO_APP%"=="1" (
    call :detect_python
    if not defined PYTHON_CMD (
        echo [ERROR] Python 3.12 was not found. Run setup_windows.bat first.
        set "EXIT_CODE=1"
        set "UPDATE_RESULT=failed; Python 3.12 was not found"
        goto finish
    )

    echo [INFO] Updating application files...
    %PYTHON_CMD% "%~dp0scripts\update_app.py" %APP_ARGS%
    if errorlevel 1 (
        set "EXIT_CODE=%errorlevel%"
        set "UPDATE_RESULT=failed; application update did not complete"
        goto finish
    )
)

if "%DO_DEPS%"=="1" (
    if not exist ".venv\Scripts\python.exe" (
        if "%DRY_RUN%"=="1" (
            if not defined PYTHON_CMD call :detect_python
            if not defined PYTHON_CMD (
                echo [ERROR] Python 3.12 was not found. Run setup_windows.bat first.
                set "EXIT_CODE=1"
                set "UPDATE_RESULT=failed; Python 3.12 was not found"
                goto finish
            )
            echo [INFO] .venv was not found. Checking whether the recommended dependencies can be installed...
            %PYTHON_CMD% "%~dp0scripts\update_venv.py" !PREFLIGHT_ARGS!
            if errorlevel 1 (
                set "EXIT_CODE=%errorlevel%"
                set "UPDATE_RESULT=failed; dependency preflight did not complete"
                goto finish
            )
            goto deps_done
        )

        echo [INFO] .venv was not found. Running setup_windows.bat first...
        call setup_windows.bat --no-pause
        if errorlevel 1 (
            set "EXIT_CODE=%errorlevel%"
            set "UPDATE_RESULT=failed; setup_windows.bat did not complete"
            goto finish
        )
    )

    if "%SKIP_DEP_CHECK%"=="0" (
        echo [INFO] Checking recommended dependencies in the existing .venv...
        ".venv\Scripts\python.exe" "%~dp0scripts\check_venv.py" !CHECK_ARGS!
        if not errorlevel 1 (
            echo [INFO] Existing .venv already matches the recommended dependencies.
            goto deps_done
        )
    )

    echo [INFO] Updating recommended dependencies in the existing .venv...
    ".venv\Scripts\python.exe" "%~dp0scripts\sync_venv.py" !SYNC_ARGS!
    if errorlevel 1 (
        set "EXIT_CODE=%errorlevel%"
        set "UPDATE_RESULT=failed; dependency update did not complete"
        goto finish
    )
)

:deps_done
set "UPDATE_RESULT=update completed"
set "EXIT_CODE=0"
goto finish

:finish
echo.
echo ========== Update Summary ==========
if defined UPDATE_RESULT (
    echo Result: %UPDATE_RESULT%
) else (
    echo Result: update did not run
)
if "%DO_APP%"=="1" echo Application files: checked
if "%DO_DEPS%"=="1" echo Dependencies: checked
if exist ".cache\updater\installed_manifest.json" echo Manifest: "%~dp0.cache\updater\installed_manifest.json"
echo ===================================
if "%PAUSE_ON_EXIT%"=="1" (
    echo.
    pause
)
exit /b %EXIT_CODE%

:detect_python
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=".venv\Scripts\python.exe""
    exit /b 0
)

where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=py -3.12"
        exit /b 0
    )
)

if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD="%LocalAppData%\Programs\Python\Python312\python.exe""
    exit /b 0
)

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=python"
        exit /b 0
    )
)
exit /b 0
