# Third-Party Licenses and Model Terms

This repository's own source code is licensed under the MIT License.

Some optional features download or use third-party libraries and model weights
that are governed by separate license terms.

## YOLO/SAM Mask Feature

The YOLO/SAM mask feature may use the `ultralytics` Python package and
Ultralytics YOLO model weights such as `yolo26m.pt`, `yolo26l.pt`, and
`yolo26x.pt`. Ultralytics YOLO code and trained models are licensed under
AGPL-3.0 by default, with an Ultralytics Enterprise License available for
proprietary commercial use.

The same feature may use Meta Segment Anything Model 2 / 2.1 checkpoints such
as `sam2.1_l.pt`. SAM2 model checkpoints and code are licensed under the Apache
License 2.0 by Meta.

Model weights are not included in this repository or release archives. YOLO and
SAM2/SAM2.1 weights may be downloaded by the user's environment on first use, or
provided by the user as local files. Local YOLO/SAM `.pt` files should be placed under
`models/ultralytics/`; legacy `.pt` files in the repository root are still
detected for compatibility.

Users are responsible for ensuring that their use, redistribution, deployment,
or commercial integration complies with the applicable third-party license
terms.

References:

- Ultralytics License: https://www.ultralytics.com/license
- Meta SAM2 repository: https://github.com/facebookresearch/sam2

## Sky Mask Feature

The semantic/prompt mask feature may use Mask2Former ADE20K
semantic-segmentation model weights, such as
`facebook/mask2former-swin-large-ade-semantic`, together with the Hugging Face
`transformers` and `safetensors` libraries. It may also use Meta's `sam3`
Python package and a Meta SAM3.1 checkpoint such as `sam3.1_multiplex.pt`.

Mask2Former source code is licensed under the MIT License. The `transformers`
and `safetensors` libraries are licensed under the Apache License 2.0. ADE20K
dataset materials are governed by the dataset provider's terms. Meta SAM3.1
materials, including model code and trained weights, are governed by the SAM
License provided by Meta.

Model weights are not included in this repository or release archives.
Mask2Former may be downloaded by the user's environment on first use, or
provided by the user as local files under
`models/mask2former-swin-large-ade-semantic/`. SAM3.1 checkpoints may be
downloaded after the user has access to `facebook/sam3.1` and has accepted the
SAM License, or provided as local files under `models/sam3.1/`.

Users are responsible for ensuring that their use, redistribution, deployment,
or commercial integration complies with the applicable third-party license and
dataset terms.

References:

- Mask2Former repository: https://github.com/facebookresearch/Mask2Former
- Mask2Former ADE20K model: https://huggingface.co/facebook/mask2former-swin-large-ade-semantic
- Meta SAM3.1 model: https://huggingface.co/facebook/sam3.1
- Meta SAM3 repository: https://github.com/facebookresearch/sam3
- Transformers: https://github.com/huggingface/transformers
- safetensors: https://github.com/huggingface/safetensors
- ADE20K terms: https://ade20k.csail.mit.edu/terms/
