# Third-party licenses

The application source is MIT licensed. Distributions must also include the
license texts for bundled dependencies and managed components.

| Component | Role | License handling |
|---|---|---|
| stechdrive-3dgs-utils | Primary source foundation | MIT notice retained in `LICENSE` and `NOTICE.md` |
| Insta360Convert-GUI | Feature reference | MIT notice acknowledged; implementation is new |
| PySide6 / Qt | Desktop UI | LGPL/commercial terms; include Qt notices and relinking information in builds |
| FFmpeg / FFprobe | Media processing | Build-dependent LGPL/GPL notices shown before installation |
| OpenCV, NumPy, Pillow | Image processing | Licenses collected into release bundle |
| COLMAP | SfM integration | Separate executable; license shown by component manager |
| Inter | Typography | OFL-1.1; bundle only the unmodified upstream font and license |
| LichtFeld Studio | External trainer | GPLv3 application detected or selected by the user; never bundled |
| Postshot | External trainer | Proprietary application selected by the user; never bundled |
| Brush / gsplat | External trainer | Separate installations; never silently downloaded |

`scripts/build_sbom.py` is intended to generate the exact dependency inventory
for a release. Component manifests must contain a license identifier and SHA-256
before a managed download can be enabled.

## YOLO and SAM mask features

Optional masking may use the `ultralytics` package and YOLO model weights.
Ultralytics YOLO code and trained models use AGPL-3.0 by default, with a
separate enterprise license available from Ultralytics. Optional SAM2 / SAM2.1
code and checkpoints are provided by Meta under Apache-2.0. Model weights are
not included in this repository or its release archives.

References:

- [Ultralytics license](https://www.ultralytics.com/license)
- [Meta SAM2](https://github.com/facebookresearch/sam2)

## Semantic and prompt mask features

Optional semantic masking may use YOLO26 semantic weights and Cityscapes class
definitions. Prompt masking may use Meta SAM3.1 code and checkpoints, governed
by Meta's SAM License. Users must obtain gated weights and accept their terms;
weights are not bundled. Local Ultralytics weights belong under
`models/ultralytics/`, and local SAM3.1 checkpoints under `models/sam3.1/`.

References:

- [Cityscapes terms](https://www.cityscapes-dataset.com/license/)
- [Meta SAM3.1 model](https://huggingface.co/facebook/sam3.1)
- [Meta SAM3 repository](https://github.com/facebookresearch/sam3)

Users are responsible for ensuring that their use, redistribution, deployment,
or commercial integration complies with every applicable third-party license,
model license, and dataset term.
