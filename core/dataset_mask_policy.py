from __future__ import annotations

DATASET_MASK_REUSE_EXISTING = "reuse_existing"
DATASET_MASK_CONVERT_SFM = "convert_sfm_masks"
DATASET_MASK_GENERATE_TRAINING = "generate_training"
DATASET_MASK_NONE = "none"

DATASET_MASK_MODES = {
    DATASET_MASK_REUSE_EXISTING,
    DATASET_MASK_CONVERT_SFM,
    DATASET_MASK_GENERATE_TRAINING,
    DATASET_MASK_NONE,
}


def normalize_dataset_mask_mode(value: object, *, default: str = DATASET_MASK_CONVERT_SFM) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "": default,
        "reuse": DATASET_MASK_REUSE_EXISTING,
        "existing": DATASET_MASK_REUSE_EXISTING,
        "keep": DATASET_MASK_REUSE_EXISTING,
        "convert": DATASET_MASK_CONVERT_SFM,
        "sfm": DATASET_MASK_CONVERT_SFM,
        "sfm_masks": DATASET_MASK_CONVERT_SFM,
        "training": DATASET_MASK_GENERATE_TRAINING,
        "generate": DATASET_MASK_GENERATE_TRAINING,
        "regenerate": DATASET_MASK_GENERATE_TRAINING,
        "off": DATASET_MASK_NONE,
        "no": DATASET_MASK_NONE,
        "false": DATASET_MASK_NONE,
    }
    mode = aliases.get(text, text)
    if mode not in DATASET_MASK_MODES:
        raise ValueError(f"Unsupported dataset mask mode: {value}")
    return mode


def dataset_mask_mode_from_legacy_write_masks(write_masks: object) -> str:
    return DATASET_MASK_CONVERT_SFM if bool(write_masks) else DATASET_MASK_REUSE_EXISTING


def dataset_mask_mode_writes_converted_sfm_masks(mode: object) -> bool:
    return normalize_dataset_mask_mode(mode) == DATASET_MASK_CONVERT_SFM


def dataset_mask_mode_generates_training_masks(mode: object) -> bool:
    return normalize_dataset_mask_mode(mode) == DATASET_MASK_GENERATE_TRAINING
