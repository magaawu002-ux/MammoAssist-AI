"""Optional post-canonical experiment; deliberately absent from cache building."""

import cv2
import numpy as np

from src.preprocessing.model_inputs import canonical_array


def optional_clahe(image: np.ndarray, *, enabled: bool = False,
                   clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Return an independent array; enabled CLAHE uses 16-bit input and [0,1] output.

    Apply only in a separately recorded ablation. The transform covers the entire
    canonical canvas, including padding, and does not write the canonical cache.
    """
    array = canonical_array(image)
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if not enabled:
        return array
    if not np.isfinite(clip_limit) or clip_limit <= 0 or len(tile_grid_size) != 2 or any(
            type(v) is not int or v <= 0 for v in tile_grid_size):
        raise ValueError("Invalid CLAHE settings")
    encoded = np.rint(array * 65535).astype(np.uint16)
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(encoded).astype(np.float32) / 65535
