"""Validate the canonical preprocessing contract from central configuration."""

from pathlib import PurePosixPath
from typing import Any


def preprocessing_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Reject unsupported settings rather than silently changing experiments."""
    scope = {"dataset": "CBIS-DDSM", "subset": "mass", "image_type": "full_mammogram",
             "positive_class": "MALIGNANT", "task": "binary_classification"}
    if any(config.get(k) != v for k, v in scope.items()):
        raise ValueError("Preprocessing requires CBIS-DDSM mass full-mammogram configuration")
    settings = dict(config.get("preprocessing", {}))
    expected = {"version", "crop_background", "crop_margin_fraction", "cache_directory", "cache_dtype"}
    if set(settings) != expected:
        raise ValueError(f"Expected preprocessing settings: {sorted(expected)}")
    if settings["version"] != "canonical_v1" or settings["cache_dtype"] != "float32":
        raise ValueError("This implementation supports canonical_v1 / float32 only")
    if type(settings["crop_background"]) is not bool:
        raise ValueError("crop_background must be boolean")
    margin = settings["crop_margin_fraction"]
    if type(margin) not in (int, float) or not 0 <= margin <= 0.5:
        raise ValueError("crop_margin_fraction must be finite and in [0, 0.5]")
    size = config.get("input_size")
    if type(size) is not int or size < 2:
        raise ValueError("input_size must be an integer >= 2")
    directory = settings["cache_directory"]
    if not isinstance(directory, str) or "\\" in directory or ":" in directory:
        raise ValueError("cache_directory must be a portable relative path")
    path = PurePosixPath(directory)
    if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("data", "processed") or len(path.parts) < 3:
        raise ValueError("cache_directory must be beneath data/processed")
    return dict(settings, input_size=size)
