# Developer Guide

## Architecture

- `gs360studio/domain` — versioned contracts
- `gs360studio/engine` — projection and export logic
- `gs360studio/pipeline` — resumable workflow orchestration
- `gs360studio/adapters` — external tool boundaries
- `gs360studio/platform` — storage, diagnostics, components, and updates
- `gui` — PySide6 workstation and inherited workflow widgets

## Quality gates

Before opening a pull request:

```powershell
python -m ruff check .
python -m pytest -q
```

Use Conventional Commits, preserve the MIT and third-party notices, and add tests for contract, migration, projection, or UI changes. Architecture decisions live under `doc/adr/`.
