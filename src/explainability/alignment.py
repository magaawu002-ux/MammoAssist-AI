"""Prove source-grid identity, then replay saved canonical spatial geometry."""

import math

import cv2
import numpy as np
import pydicom


def prove_full_grid(mask_header: pydicom.dataset.Dataset, full_header: pydicom.dataset.Dataset,
                    review: dict | None = None) -> str:
    """Equal dimensions alone are insufficient; absent geometry requires explicit review."""
    if (int(mask_header.Rows), int(mask_header.Columns)) != (int(full_header.Rows), int(full_header.Columns)):
        raise ValueError("Mask is not on the full-image grid; cropped/offset/resampled masks need verified registration")
    fields = {"ImagePositionPatient": 3, "ImageOrientationPatient": 6, "PixelSpacing": 2}
    complete = all(hasattr(ds, field) for ds in (mask_header, full_header) for field in fields)
    for field in fields:
        if hasattr(mask_header, field) and hasattr(full_header, field):
            a, b = np.asarray(getattr(mask_header, field), float), np.asarray(getattr(full_header, field), float)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,atol=1e-6,rtol=0):
                raise ValueError(f"Conflicting full-image/mask geometry: {field}")
    a_frame, b_frame = str(getattr(mask_header,"FrameOfReferenceUID","")), str(getattr(full_header,"FrameOfReferenceUID",""))
    if a_frame and b_frame and a_frame != b_frame:
        raise ValueError("Conflicting FrameOfReferenceUID")
    if complete:
        for field, length in fields.items():
            a, b = np.asarray(getattr(mask_header, field), float), np.asarray(getattr(full_header, field), float)
            if a.shape != (length,) or b.shape != (length,) or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Malformed spatial geometry")
            if not np.allclose(a, b, atol=1e-6, rtol=0):
                raise ValueError(f"Conflicting full-image/mask geometry: {field}")
        spacing = np.asarray(full_header.PixelSpacing, float)
        orientation = np.asarray(full_header.ImageOrientationPatient, float).reshape(2, 3)
        if np.any(spacing <= 0) or not np.allclose(orientation @ orientation.T, np.eye(2), atol=1e-6):
            raise ValueError("Invalid spacing or direction cosines")
        frame = str(getattr(full_header, "FrameOfReferenceUID", ""))
        if frame and frame == str(getattr(mask_header, "FrameOfReferenceUID", "")):
            return "matching_DICOM_frame_origin_orientation_spacing_and_dimensions"
    if review and (review.get("method") == "reviewed_full_grid_identity"
                   and review.get("mask_sop_uid") == str(mask_header.SOPInstanceUID)
                   and review.get("full_sop_uid") == str(full_header.SOPInstanceUID)
                   and all(review.get(k, "").strip() for k in ("reviewer", "reviewed_at", "evidence"))):
        return "reviewed_full_grid_identity"
    raise ValueError("UNVERIFIED_ALIGNMENT: missing shared DICOM geometry; explicit hash-bound review required")


def transform_mask(mask: np.ndarray, geometry: dict) -> np.ndarray:
    """Replay canonical_v1 exclusive crop, rounded resize and centred zero padding."""
    if mask.dtype != np.bool_ or mask.ndim != 2 or not mask.any():
        raise ValueError("Expected a nonempty binary mask on a verified full-image grid")
    if geometry.get("preprocessing_version") != "canonical_v1" or geometry.get("reorientation") != "none":
        raise ValueError("Unsupported preprocessing geometry version/orientation")
    h, w = geometry["original_height"], geometry["original_width"]
    if mask.shape != (h, w):
        raise ValueError("Original mask and full-mammogram dimensions differ")
    y0, y1, x0, x1 = geometry["crop_box_y0_y1_x0_x1"]
    size = geometry["input_size"]
    if any(type(v) is not int for v in (h, w, size, y0, y1, x0, x1)) or not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
        raise ValueError("Invalid saved crop box")
    ch, cw = y1-y0, x1-x0
    scale = size / max(ch, cw)
    rh, rw = max(1, math.floor(ch*scale+0.5)), max(1, math.floor(cw*scale+0.5))
    top, left = (size-rh)//2, (size-rw)//2
    expected = {"crop_height": ch, "crop_width": cw, "resized_height": rh, "resized_width": rw,
                "padding_top": top, "padding_bottom": size-rh-top, "padding_left": left, "padding_right": size-rw-left,
                "padding_value": 0}
    if any(geometry.get(k) != v for k, v in expected.items()) or not np.isclose(geometry["resize_scale"], scale, atol=1e-12):
        raise ValueError("Saved resize/padding contradicts canonical geometry")
    cropped = mask[y0:y1, x0:x1]
    if int(cropped.sum()) != int(mask.sum()):
        raise ValueError("Canonical crop removes annotated pixels; do not silently score a clipped lesion")
    resized = cv2.resize(cropped.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not resized.any():
        raise ValueError("Lesion vanished at canonical resolution; resolution loss requires reporting")
    canvas = np.zeros((size, size), bool)
    canvas[top:top+rh, left:left+rw] = resized
    return canvas


def union_masks(masks: list[np.ndarray]) -> np.ndarray:
    if not masks or any(m.dtype != np.bool_ or m.shape != masks[0].shape or not m.any() for m in masks):
        raise ValueError("Union requires nonempty aligned binary masks of equal shape")
    return np.logical_or.reduce(masks)
