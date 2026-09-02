# Contributing

Use Python 3.12. Create a virtual environment, install
`requirements/core.txt` and `requirements/test.txt`, then run `pytest` and
`ruff check .` before opening a pull request.

Commits follow Conventional Commits. `main` is release-ready: changes arrive
through pull requests with tests and a changelog label. Keep external tool
adapters capability-driven, preserve `_stechdrive` metadata, and never commit
commercial applications, model weights, or LichtFeld source/artwork.

The upstream remote is:

```text
upstream https://github.com/stechdrive/stechdrive-3dgs-utils.git
```

Fetch upstream, create a dedicated sync branch, and cherry-pick or merge only
reviewed changes. Record conflicts and behavior changes with the
`Upstream impact` project field.
