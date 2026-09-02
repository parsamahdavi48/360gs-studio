# Installation

## Packaged development preview

Open the [GitHub Releases page](https://github.com/parsamahdavi48/360gs-studio/releases) and select the newest prerelease.

- **Installer:** download `360GS-Studio-...-setup.exe` and follow the setup wizard.
- **Portable:** download the `...-windows-x64-portable.zip`, extract the complete archive, and run `360GS Studio.exe`. Keep the `_internal` folder beside the executable.

The packaged application includes its Python, Qt, OpenCV, and core CPU runtime. It does not bundle FFmpeg, COLMAP, model weights, CUDA/PyTorch ML components, or external trainers; diagnostics and the component manager report what is available.

Development previews may be unsigned. Download `SHA256SUMS.txt` from the same release and verify the installer or ZIP before running it:

```powershell
Get-FileHash .\360GS-Studio-*-windows-x64-portable.zip -Algorithm SHA256
```

Compare the displayed value with the matching line in `SHA256SUMS.txt`.

## Run from source

1. Install Python 3.12 and Git.
2. Clone the repository.
3. Run `setup_windows.bat` from the repository folder.
4. Launch with `run_gui.bat`.

The setup script creates an isolated `.venv` and checks the main media and GPU dependencies. This is intended for contributors and advanced testing; a system Python installation is not required for packaged releases.

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
