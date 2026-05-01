@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

echo [INFO] setup_windows.bat

set "PYTHON_CMD="
set "PY_VER="
set "PY_VER_FULL="
set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PY311_LOCAL=%LocalAppData%\Programs\Python\Python311\python.exe"
set "PY311_PROGRAMFILES=%ProgramFiles%\Python311\python.exe"
set "PY311_PROGRAMFILES_X86=%ProgramFiles(x86)%\Python311\python.exe"

call :detect_py311

if not defined PYTHON_CMD (
    echo [INFO] Python 3.11 was not found. Trying winget install of 3.11.8...
    where winget >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] winget is not available and Python 3.11 is missing.
        echo [ERROR] Install Python 3.11 manually, then run setup_windows.bat again.
        exit /b 1
    )

    winget install --id Python.Python.3.11 --version 3.11.8 --source winget --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [ERROR] winget failed to install Python 3.11.8.
        echo [ERROR] Run manually: winget install --id Python.Python.3.11 --version 3.11.8 --source winget
        exit /b 1
    )

    call :detect_py311
)

if not defined PYTHON_CMD (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python command was not found after installation attempt.
        exit /b 1
    )

    for /f "usebackq delims=" %%V in (`python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"` ) do set "PY_VER=%%V"
    if not "%PY_VER%"=="3.11" (
        echo [ERROR] Python 3.11 is required for this repository.
        echo [ERROR] Found python version: %PY_VER%
        exit /b 1
    )

    set "PYTHON_CMD=python"
)

for /f "usebackq delims=" %%V in (`%PYTHON_CMD% -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"` ) do set "PY_VER=%%V"
for /f "usebackq delims=" %%V in (`%PYTHON_CMD% -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')"` ) do set "PY_VER_FULL=%%V"
echo [INFO] Python command: %PYTHON_CMD%
echo [INFO] Python version: %PY_VER_FULL%

if not "%PY_VER_FULL%"=="3.11.8" (
    echo [WARN] This repository is confirmed with Python 3.11.8.
    echo [WARN] Continuing with Python %PY_VER_FULL%.
)

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating venv: .venv
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv
        exit /b 1
    )
) else (
    echo [INFO] Reusing existing venv: .venv
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate .venv
    exit /b 1
)

echo [INFO] Upgrading pip
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] pip upgrade failed
    exit /b 1
)

echo [INFO] Installing PyTorch (CUDA 12.8 wheels)
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [ERROR] PyTorch install failed
    exit /b 1
)

echo [INFO] Installing project dependencies
python -m pip install numpy opencv-python Pillow open3d ultralytics tqdm PySide6
if errorlevel 1 (
    echo [ERROR] Dependency install failed
    exit /b 1
)

echo [DONE] venv setup completed.
exit /b 0

:detect_py311
where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -c "import sys" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=py -3.11"
        exit /b 0
    )
)

if not defined PYTHON_CMD if exist "!PY311_LOCAL!" (
    "!PY311_LOCAL!" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD="!PY311_LOCAL!""
    )
)
if defined PYTHON_CMD exit /b 0

if not defined PYTHON_CMD if exist "!PY311_PROGRAMFILES!" (
    "!PY311_PROGRAMFILES!" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD="!PY311_PROGRAMFILES!""
    )
)
if defined PYTHON_CMD exit /b 0

if not defined PYTHON_CMD if not "!PY311_PROGRAMFILES_X86!"=="\Python311\python.exe" if exist "!PY311_PROGRAMFILES_X86!" (
    "!PY311_PROGRAMFILES_X86!" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD="!PY311_PROGRAMFILES_X86!""
    )
)
if defined PYTHON_CMD exit /b 0

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=python"
        exit /b 0
    )
)
exit /b 0
