@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

echo [INFO] setup_windows.bat

set "PYTHON_CMD="
set "PY_VER="
set "PY_VER_FULL="
set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PY312_LOCAL=%LocalAppData%\Programs\Python\Python312\python.exe"
set "PY312_PROGRAMFILES=%ProgramFiles%\Python312\python.exe"
set "PY312_PROGRAMFILES_X86=%ProgramFiles(x86)%\Python312\python.exe"

call :detect_py312

if not defined PYTHON_CMD (
    echo [INFO] Python 3.12 was not found. Trying winget install of Python 3.12...
    where winget >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] winget is not available and Python 3.12 is missing.
        echo [ERROR] Install Python 3.12 manually, then run setup_windows.bat again.
        exit /b 1
    )

    winget install --id Python.Python.3.12 --source winget --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [ERROR] winget failed to install Python 3.12.
        echo [ERROR] Run manually: winget install --id Python.Python.3.12 --source winget
        exit /b 1
    )

    call :detect_py312
)

if not defined PYTHON_CMD (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python command was not found after installation attempt.
        exit /b 1
    )

    for /f "usebackq delims=" %%V in (`python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"` ) do set "PY_VER=%%V"
    if not "%PY_VER%"=="3.12" (
        echo [ERROR] Python 3.12 is required for this repository.
        echo [ERROR] Found python version: %PY_VER%
        exit /b 1
    )

    set "PYTHON_CMD=python"
)

for /f "usebackq delims=" %%V in (`%PYTHON_CMD% -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"` ) do set "PY_VER=%%V"
for /f "usebackq delims=" %%V in (`%PYTHON_CMD% -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')"` ) do set "PY_VER_FULL=%%V"
echo [INFO] Python command: %PYTHON_CMD%
echo [INFO] Python version: %PY_VER_FULL%

if not "%PY_VER_FULL%"=="3.12.10" (
    echo [WARN] This repository is confirmed with Python 3.12.10.
    echo [WARN] Continuing with Python %PY_VER_FULL%.
)

echo [INFO] Creating verified Python 3.12 environment
%PYTHON_CMD% "%~dp0scripts\update_venv.py" --candidates 3.12 --skip-pytest
if errorlevel 1 (
    echo [ERROR] venv setup failed
    exit /b 1
)

echo [DONE] venv setup completed.
exit /b 0

:detect_py312
where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=py -3.12"
        exit /b 0
    )
)

if not defined PYTHON_CMD if exist "!PY312_LOCAL!" (
    "!PY312_LOCAL!" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD="!PY312_LOCAL!""
    )
)
if defined PYTHON_CMD exit /b 0

if not defined PYTHON_CMD if exist "!PY312_PROGRAMFILES!" (
    "!PY312_PROGRAMFILES!" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD="!PY312_PROGRAMFILES!""
    )
)
if defined PYTHON_CMD exit /b 0

if not defined PYTHON_CMD if not "!PY312_PROGRAMFILES_X86!"=="\Python312\python.exe" if exist "!PY312_PROGRAMFILES_X86!" (
    "!PY312_PROGRAMFILES_X86!" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD="!PY312_PROGRAMFILES_X86!""
    )
)
if defined PYTHON_CMD exit /b 0

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=python"
        exit /b 0
    )
)
exit /b 0
