# Installation

## Current development build

1. Install Python 3.12 and Git.
2. Clone the repository.
3. Run `setup_windows.bat` from the repository folder.
4. Launch with `run_gui.bat`.

The setup script creates an isolated `.venv` and checks the main media and GPU dependencies. A system Python installation is not required for packaged releases.

## Language

Choose **Language → English / 日本語 / فارسی** in the application. The selected language is applied at the next start. Persian launches with a right-to-left interface and falls back to English for specialist strings that have not been translated yet.

From the command line:

```powershell
360gs-studio gui --language fa
```

## Diagnostics

Run:

```powershell
360gs-studio doctor
```

Core CPU conversion remains usable when optional NVIDIA, CUDA, COLMAP, or trainer capabilities are unavailable.
