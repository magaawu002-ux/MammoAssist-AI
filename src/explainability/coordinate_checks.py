"""Annotation coordinate diagnostics only; these candidates do not authorise scoring."""

import math
import cv2
import numpy as np


def require_development(partition: str) -> None:
    if partition not in ('train', 'validation'):
        raise ValueError('Coordinate-rule development requires train/validation; test is forbidden')


def grid_relation(full_shape, mask_shape) -> dict:
    h, w = full_shape
    mh, mw = mask_shape
    if any(type(v) is not int or v <= 0 for v in (h, w, mh, mw)):
        raise ValueError('Invalid dimensions')
    sy, sx = h / mh, w / mw
    equal = (h, w) == (mh, mw)
    # Descriptive near-proportional category: at most one pixel uncertainty per axis.
    # This tolerance is not evidence of source coordinate correspondence.
    proportional = abs(sy-sx) <= 1/mh + 1/mw
    return dict(row_scale_ratio=sy, column_scale_ratio=sx,
                aspect_ratio_difference=abs(w/h-mw/mh),
                relative_aspect_ratio_difference=abs((w/h)/(mw/mh)-1),
                dimensions_equal=equal,
                grid_relation='equal' if equal else ('proportional_with_integer_rounding' if proportional else 'non_proportional'))


def mask_statistics(mask: np.ndarray) -> dict:
    if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.any() or mask.all():
        raise ValueError('Expected nonempty, nonconstant boolean mask')
    rows, cols = mask.sum(axis=1), mask.sum(axis=0)
    ys, xs = np.flatnonzero(rows), np.flatnonzero(cols)
    n = int(rows.sum())
    return dict(bbox_y0=int(ys[0]), bbox_y1=int(ys[-1])+1,
                bbox_x0=int(xs[0]), bbox_x1=int(xs[-1])+1,
                centroid_y=float(rows @ np.arange(len(rows)) / n),
                centroid_x=float(cols @ np.arange(len(cols)) / n),
                mask_pixels=n, mask_area_fraction=n/mask.size)


def candidate_mask(mask: np.ndarray, full_shape: tuple[int, int]) -> np.ndarray:
    """Return an explicitly UNPROVEN diagnostic candidate, never a scoring mask."""
    mask_statistics(mask)
    relation = grid_relation(full_shape, mask.shape)['grid_relation']
    if relation == 'equal':
        return mask.copy()
    if relation != 'proportional_with_integer_rounding':
        raise ValueError('Aspect-ratio mismatch: candidate scaling rejected')
    result = cv2.resize(mask.astype(np.uint8), (full_shape[1], full_shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not result.any():
        raise ValueError('Candidate loses lesion support')
    return result


def ncc(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation: affine-intensity invariant; constant patches are undefined."""
    if a.shape != b.shape or a.size < 16 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    x -= x.mean(); y -= y.mean()
    denom = math.sqrt(float(x@x)*float(y@y))
    return float(np.clip((x@y)/denom, -1, 1)) if denom > 1e-12 else None


def crop_consistency(full: np.ndarray, crop: np.ndarray, support: np.ndarray) -> dict:
    """Check bbox-centred origin, a bounded translation search, and displaced controls.

    Crops stay at native pixel resolution. The supplied crop is never a model input.
    Search offsets are diagnostics, not transformations applied to the ROI.
    """
    s = mask_statistics(support)
    h, w = crop.shape
    if h > full.shape[0] or w > full.shape[1] or min(h, w) < 4:
        raise ValueError('Crop larger than full image or too small')
    y = int(np.clip(math.floor((s['bbox_y0']+s['bbox_y1']-h)/2), 0, full.shape[0]-h))
    x = int(np.clip(math.floor((s['bbox_x0']+s['bbox_x1']-w)/2), 0, full.shape[1]-w))
    patch = full[y:y+h, x:x+w]
    controls = []
    for dy, dx in ((-h,0),(h,0),(0,-w),(0,w),(-h,-w),(-h,w),(h,-w),(h,w)):
        yy, xx = y+dy, x+dx
        if 0 <= yy <= full.shape[0]-h and 0 <= xx <= full.shape[1]-w:
            value = ncc(full[yy:yy+h, xx:xx+w], crop)
            if value is not None:
                controls.append(value)
    direct = ncc(patch, crop)
    # Normalise values before OpenCV correlation to avoid cancellation on 16-bit data.
    lo, hi = float(full.min()), float(full.max())
    cy, cx = max(0, y-32), max(0, x-32)
    region = full[cy:min(full.shape[0],y+h+32), cx:min(full.shape[1],x+w+32)]
    a = ((region.astype(np.float32)-lo)/max(hi-lo,1)).astype(np.float32)
    t = ((crop.astype(np.float32)-float(crop.mean()))/max(float(crop.std()),1)).astype(np.float32)
    response = cv2.matchTemplate(a, t, cv2.TM_CCOEFF_NORMED)
    _, _, _, location = cv2.minMaxLoc(response)
    by, bx = cy+location[1], cx+location[0]
    best = ncc(full[by:by+h,bx:bx+w], crop)  # Recompute stably, rejecting constants.
    return dict(crop_rows=h, crop_columns=w, proposed_crop_y=y, proposed_crop_x=x,
                crop_ncc_at_proposed_origin=direct, crop_ncc_local_best=best,
                local_search_dy=by-y, local_search_dx=bx-x,
                local_search_radius_pixels=32, negative_control_count=len(controls),
                negative_control_max_ncc=max(controls) if controls else None,
                crop_contains_mask_bbox=(y <= s['bbox_y0'] and x <= s['bbox_x0'] and
                                         y+h >= s['bbox_y1'] and x+w >= s['bbox_x1']))


def foreground_checks(full: np.ndarray, valid: np.ndarray, support: np.ndarray) -> dict:
    """Conservative nonbackground support and a separate Otsu-component tissue proxy.

    Neither statistic certifies tissue or alignment on its own. Otsu uses a fixed
    1024-pixel longest-axis image for speed; no ROI or label tunes its threshold.
    """
    if full.shape != support.shape or valid.shape != full.shape:
        raise ValueError('Foreground/support shape mismatch')
    n = int(support.sum())
    low, high = float(full[valid].min()), float(full[valid].max())
    nonbackground = valid & (full > low)
    scale = min(1.0, 1024/max(full.shape))
    shape = (max(1,round(full.shape[1]*scale)), max(1,round(full.shape[0]*scale)))
    small = cv2.resize(full.astype(np.float32), shape, interpolation=cv2.INTER_AREA)
    small = np.clip((small-low)/max(high-low,1)*255,0,255).astype(np.uint8)
    threshold, binary = cv2.threshold(small,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    tissue = labels == (1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))) if count>1 else np.zeros_like(binary,bool)
    # Test foreground at the source mask's actual support coordinates, no mask resizing.
    ys, xs = np.where(support)
    mapped_y = np.minimum((ys*shape[1]/full.shape[0]).astype(int), shape[1]-1)
    mapped_x = np.minimum((xs*shape[0]/full.shape[1]).astype(int), shape[0]-1)
    return dict(nonbackground_fraction=float(nonbackground[support].sum()/n),
                declared_valid_fraction=float(valid[support].sum()/n),
                otsu_largest_component_fraction=float(tissue[mapped_y,mapped_x].mean()),
                otsu_threshold_8bit=float(threshold))


def evidence_screen(check: dict) -> tuple[bool, str]:
    """Strict diagnostic screen fixed before cohort processing; NOT alignment approval."""
    reasons = []
    if check['nonbackground_fraction'] < .99:
        reasons.append('background_support')
    value = check['crop_ncc_at_proposed_origin']
    if value is None or value < .995:
        reasons.append('crop_origin_not_confirmed')
    if not check['crop_contains_mask_bbox']:
        reasons.append('crop_does_not_cover_mask_bbox')
    if check['negative_control_count'] < 3 or check['negative_control_max_ncc'] is None or check['negative_control_max_ncc'] >= .98:
        reasons.append('insufficient_control_separation')
    return not reasons, ';'.join(reasons)
