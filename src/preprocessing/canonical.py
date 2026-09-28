"""Model-independent whole-mammogram normalization, border crop and letterbox."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.preprocessing.dicom import decode_dicom, finite_image
from src.preprocessing.settings import preprocessing_settings

# Conservative fallback guards, part of the versioned algorithm, not tunable
# thresholds for segmentation: all non-minimum pixels are always preserved.
MIN_FOREGROUND_FRACTION = 0.01
MIN_FOREGROUND_SPAN_FRACTION = 0.1


@dataclass
class PreprocessingResult:
    image: np.ndarray
    metadata: dict[str, Any]
    decoded: np.ndarray | None = None
    cropped: np.ndarray | None = None


def crop_background(image: np.ndarray, enabled: bool, margin_fraction: float) -> tuple[np.ndarray, dict]:
    """Remove only entirely constant-minimum outer borders; retain every other pixel.

    No segmentation, connected-component deletion, lesion or ROI information.
    Uniform images or implausibly sparse foreground keep their original extent.
    """
    image = finite_image(image)
    if type(enabled) is not bool or not 0 <= margin_fraction <= 0.5:
        raise ValueError("Invalid crop settings")
    height, width = image.shape
    box, reason = [0, height, 0, width], "disabled"
    if enabled:
        foreground = image > image.min()
        ys = np.flatnonzero(foreground.any(axis=1))
        xs = np.flatnonzero(foreground.any(axis=0))
        reliable = (len(ys) and len(xs) and foreground.mean() >= MIN_FOREGROUND_FRACTION
                    and ys[-1] - ys[0] + 1 >= height * MIN_FOREGROUND_SPAN_FRACTION
                    and xs[-1] - xs[0] + 1 >= width * MIN_FOREGROUND_SPAN_FRACTION)
        reason = "unreliable_foreground_preserved"
        if reliable:
            my, mx = math.ceil(height * margin_fraction), math.ceil(width * margin_fraction)
            box = [max(0, int(ys[0]) - my), min(height, int(ys[-1]) + 1 + my),
                   max(0, int(xs[0]) - mx), min(width, int(xs[-1]) + 1 + mx)]
            reason = "constant_minimum_borders" if box != [0, height, 0, width] else "no_empty_borders"
    y0, y1, x0, x1 = box
    return image[y0:y1, x0:x1].copy(), {
        "crop_applied": box != [0, height, 0, width], "crop_reason": reason,
        "crop_box_y0_y1_x0_x1": box, "crop_height": y1 - y0, "crop_width": x1 - x0,
        "crop_margin_fraction": margin_fraction}


def resize_and_pad(image: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    """Fit longest axis; center with extra odd padding on bottom/right, value zero."""
    image = finite_image(image)
    if type(size) is not int or size < 2:
        raise ValueError("Output size must be an integer >= 2")
    height, width = image.shape
    scale = size / max(height, width)
    new_h = max(1, min(size, math.floor(height * scale + 0.5)))
    new_w = max(1, min(size, math.floor(width * scale + 0.5)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas = np.zeros((size, size), dtype=np.float32)
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, {"resize_scale": scale, "resized_height": new_h, "resized_width": new_w,
        "effective_scale_y": new_h / height, "effective_scale_x": new_w / width,
        "interpolation": "INTER_AREA" if scale < 1 else "INTER_LINEAR",
        "padding_top": top, "padding_bottom": size - new_h - top,
        "padding_left": left, "padding_right": size - new_w - left, "padding_value": 0}


def preprocess_dicom(path: Path, config: dict[str, Any], keep_stages: bool = False) -> PreprocessingResult:
    """Apply the same label-independent algorithm to every official partition."""
    settings = preprocessing_settings(config)
    decoded = decode_dicom(path)
    low, high = decoded.metadata["observed_valid_min"], decoded.metadata["observed_valid_max"]
    normalized = (decoded.pixels - low) / (high - low)
    normalized[~decoded.valid_mask] = 0
    cropped, crop_info = crop_background(normalized, settings["crop_background"], settings["crop_margin_fraction"])
    image, geometry = resize_and_pad(cropped, settings["input_size"])
    # Bound only floating-point interpolation round-off, not source intensities.
    image = np.clip(image, 0, 1).astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite canonical output")
    metadata = dict(decoded.metadata, **crop_info, **geometry,
        preprocessing_version=settings["version"], cache_dtype=settings["cache_dtype"],
        input_size=settings["input_size"], normalization="per_image_valid_observed_minmax_before_crop",
        numerical_range=[0, 1], clahe=False, reorientation="none", opencv_version=cv2.__version__)
    return PreprocessingResult(image, metadata, normalized if keep_stages else None,
                               cropped if keep_stages else None)
