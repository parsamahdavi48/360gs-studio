# 360GS Studio Wiki

360GS Studio is a Windows workstation for preparing 360° captures, generating perspective views and masks, solving camera poses, creating 3D Gaussian Splatting datasets, and launching compatible external trainers.

## Start here

- [Installation](Installation)
- [Your first project](First-Project)
- [Perspective Export](Perspective-Export)
- [Workflow reference](Workflow-Reference)
- [External tools](External-Tools)
- [Troubleshooting](Troubleshooting)
- [Developer guide](Developer-Guide)
- [راهنمای فارسی](Persian-Guide)

## Current release status

The project is in the v0.2 workstation-shell stage. Treat builds as prerelease software and keep original capture files outside the project output folders.

## Supported project layout

```text
scene/
├── images/          extracted or imported source frames
├── masks/           SfM and training masks
├── output/          datasets and generated media
├── _stechdrive/     preserved legacy metadata
└── _360gs/          360GS Studio manifests, jobs, profiles, and logs
```

360GS Studio never needs to modify `_stechdrive/project.json` when migrating an existing scene.
