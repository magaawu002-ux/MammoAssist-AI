"""Display-only Grad-CAM overlay; no model explanation is computed here."""

import numpy as np


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Return HxWx3 RGB float32 [0,1]; zero heat leaves the grayscale base intact.

    Alpha is maximum opacity; heat intensity controls local opacity. Colour is
    orange-red. This function neither generates nor validates a model explanation.
    """
    if (not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0
            or not isinstance(heatmap, np.ndarray) or heatmap.shape != image.shape):
        raise ValueError("Image and heatmap must be matching nonempty 2D arrays")
    for array in (image, heatmap):
        if array.dtype.kind not in "fui" or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError("Overlay arrays must be finite and within [0,1]")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and within [0,1]")
    base = np.repeat(image.astype(np.float32)[..., None], 3, axis=-1)
    opacity = np.float32(alpha) * heatmap.astype(np.float32)[..., None]
    colour = np.array([1, 0.25, 0], dtype=np.float32)
    return np.clip(base * (1 - opacity) + colour * opacity, 0, 1).astype(np.float32)
