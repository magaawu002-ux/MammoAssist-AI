"""Prespecified localisation metrics; no optimisation or classifier evaluation."""

import numpy as np


def localisation_metrics(heatmap: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> dict:
    """Pointing uses the row-major first maximum; zero maps have no valid point."""
    if (heatmap.ndim != 2 or heatmap.shape != mask.shape or mask.dtype != np.bool_ or not mask.any()
            or not np.isfinite(heatmap).all() or np.any((heatmap < 0) | (heatmap > 1))):
        raise ValueError("Expected finite [0,1] heatmap and aligned nonempty binary annotation")
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Prespecified normalized threshold must be in (0,1]")
    maximum = float(heatmap.max())
    point = np.unravel_index(np.argmax(heatmap), heatmap.shape) if maximum > 0 else None
    predicted = heatmap >= threshold
    intersection = int((predicted & mask).sum())
    union = int((predicted | mask).sum())
    total = float(heatmap.sum(dtype=np.float64))
    return {"pointing_game": bool(mask[point]) if point else False,
            "peak_row": int(point[0]) if point else None, "peak_column": int(point[1]) if point else None,
            "max_tie_count": int(np.sum(heatmap == maximum)) if point else 0,
            "lesion_hit": intersection > 0, "iou": intersection/union,
            "dice": 2*intersection/(int(predicted.sum())+int(mask.sum())),
            "activation_mass_inside": float(heatmap[mask].sum(dtype=np.float64)/total) if total > 0 else None,
            "zero_heatmap": maximum == 0, "threshold": threshold,
            "predicted_pixels": int(predicted.sum()), "annotation_pixels": int(mask.sum())}
