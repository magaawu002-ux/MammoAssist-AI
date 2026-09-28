"""Binary annotation decoding, separate from mammogram intensity preprocessing."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom

from src.explainability.roi_mapping import annotation_role


@dataclass
class DecodedMask:
    mask: np.ndarray
    header: pydicom.dataset.Dataset
    metadata: dict


def decode_mask(path: Path, expected_sop: str) -> DecodedMask:
    """Require explicitly identified, two-level masks; never threshold a crop image."""
    try:
        ds = pydicom.dcmread(path)
        if annotation_role(ds) != "mask" or str(getattr(ds, "SOPInstanceUID", "")) != expected_sop:
            raise ValueError("DICOM is not the uniquely mapped ROI mask")
        photo = str(getattr(ds, "PhotometricInterpretation", ""))
        if photo not in ("MONOCHROME1", "MONOCHROME2") or int(getattr(ds, "NumberOfFrames", 1)) != 1:
            raise ValueError("Mask must be single-frame MONOCHROME1/2")
        pixels = ds.pixel_array
        if (pixels.ndim != 2 or not pixels.size or pixels.shape != (int(ds.Rows), int(ds.Columns))
                or not np.isfinite(pixels).all()):
            raise ValueError("Invalid ROI mask pixel geometry/values")
        levels = np.unique(pixels)
        if len(levels) != 2:
            raise ValueError("Mask must contain exactly two levels; empty, constant or grayscale data rejected")
        # Display-bright annotation; do not assume 255/65535 and do not use majority-pixel guessing.
        background = levels[-1] if photo == "MONOCHROME1" else levels[0]
        mask = pixels != background
        if not mask.any() or mask.all():
            raise ValueError("Empty or all-foreground annotation rejected")
        geometry = {k: [float(v) for v in getattr(ds, k)] if hasattr(ds, k) else None for k in
                    ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing")}
        geometry.update(SOPInstanceUID=expected_sop, FrameOfReferenceUID=str(getattr(ds, "FrameOfReferenceUID", "")),
                        Rows=int(ds.Rows), Columns=int(ds.Columns), photometric=photo,
                        stored_levels=[float(v) for v in levels], background=float(background))
        return DecodedMask(mask, ds, geometry)
    except (OSError, AttributeError, ValueError, RuntimeError, EOFError, pydicom.errors.InvalidDicomError) as exc:
        raise ValueError(f"ROI mask decode failed: {exc}") from exc
