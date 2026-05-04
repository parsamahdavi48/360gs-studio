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

Model weights are not included in this repository or release archives. They may
be downloaded by the user's environment on first use, or provided by the user as
local files.

Users are responsible for ensuring that their use, redistribution, deployment,
or commercial integration complies with the applicable third-party license
terms.

References:

- Ultralytics License: https://www.ultralytics.com/license
- Meta SAM2 repository: https://github.com/facebookresearch/sam2
