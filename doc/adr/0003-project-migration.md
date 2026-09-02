# ADR-0003: Project migration

Status: accepted. Legacy `_stechdrive/project.json` is read but never edited.
The first save creates `_360gs/project.json` plus a migration report. Existing
top-level `images`, `masks`, and `output` directories remain authoritative.
