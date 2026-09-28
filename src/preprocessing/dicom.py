"""Decode stored mammogram pixels without modifying or reorienting the source."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


class DicomDecodeError(ValueError):
    """An unsupported or undecodable mammogram, with an actionable reason."""


@dataclass
class DecodedMammogram:
    pixels: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any]


def finite_image(pixels: np.ndarray) -> np.ndarray:
    """Require a nonempty finite real 2D array; convert without changing the input."""
    array = np.asarray(pixels)
    if array.ndim != 2 or array.size == 0 or min(array.shape) == 0:
        raise DicomDecodeError("Expected a nonempty 2D single-frame mammogram")
    if array.dtype.kind not in "uif":
        raise DicomDecodeError("Pixels must be real numeric values")
    result = array.astype(np.float64, copy=True)
    if not np.isfinite(result).all():
        raise DicomDecodeError("Pixel array contains NaN or Inf")
    return result


def decode_dicom(path: Path) -> DecodedMammogram:
    """Decode MONOCHROME1/2; exclude declared padding from observed intensity range.

    Stored pixels are used: VOI/window, presentation LUT and modality LUT are not
    applied. Nonidentity modality rescaling/LUT is explicitly unsupported in v1.
    Compressed pixel data requires an installed pydicom decoder plugin.
    """
    try:
        ds = pydicom.dcmread(path)
        photometric = str(getattr(ds, "PhotometricInterpretation", ""))
        if photometric not in ("MONOCHROME1", "MONOCHROME2"):
            raise DicomDecodeError(f"Unsupported PhotometricInterpretation: {photometric!r}")
        if int(getattr(ds, "NumberOfFrames", 1)) != 1 or int(getattr(ds, "SamplesPerPixel", 1)) != 1:
            raise DicomDecodeError("Only single-frame grayscale mammograms are supported")
        if ("ModalityLUTSequence" in ds or float(getattr(ds, "RescaleSlope", 1)) != 1
                or float(getattr(ds, "RescaleIntercept", 0)) != 0):
            raise DicomDecodeError("Nonidentity modality transform is unsupported in canonical_v1")
        pixels = finite_image(ds.pixel_array)
        if pixels.shape != (int(ds.Rows), int(ds.Columns)):
            raise DicomDecodeError("Pixel array dimensions disagree with DICOM Rows/Columns")
        valid = np.ones(pixels.shape, dtype=bool)
        padding = getattr(ds, "PixelPaddingValue", None)
        limit = getattr(ds, "PixelPaddingRangeLimit", padding)
        if padding is None and limit is not None:
            raise DicomDecodeError("PixelPaddingRangeLimit without PixelPaddingValue")
        if padding is not None:
            low, high = sorted((float(padding), float(limit)))
            if not np.isfinite([low, high]).all():
                raise DicomDecodeError("Nonfinite pixel padding specification")
            valid = ~((pixels >= low) & (pixels <= high))
        if not valid.any():
            raise DicomDecodeError("Image has no valid non-padding pixels")
        minimum, maximum = float(pixels[valid].min()), float(pixels[valid].max())
        if minimum == maximum:
            raise DicomDecodeError("Constant valid pixel array has no usable intensity range")
        inverted = photometric == "MONOCHROME1"
        if inverted:
            pixels[valid] = maximum - pixels[valid] + minimum
        pixels[~valid] = minimum
        if not np.isfinite(pixels).all():
            raise DicomDecodeError("Nonfinite values after photometric conversion")
        tags = ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "Modality",
                "BitsAllocated", "BitsStored", "PixelRepresentation", "WindowCenter", "WindowWidth")
        metadata = {tag: str(getattr(ds, tag, "")) for tag in tags}
        metadata.update(photometric_interpretation=photometric, inverted=inverted,
            original_height=pixels.shape[0], original_width=pixels.shape[1],
            observed_valid_min=minimum, observed_valid_max=maximum,
            padding_value=None if padding is None else float(padding),
            padding_range_limit=None if limit is None else float(limit),
            invalid_padding_pixels=int((~valid).sum()),
            transfer_syntax_uid=str(getattr(ds.file_meta, "TransferSyntaxUID", "")),
            pixel_strategy="stored_pixels; identity_modality_only; no_VOI_or_presentation_LUT",
            pydicom_version=pydicom.__version__)
        return DecodedMammogram(pixels, valid, metadata)
    except DicomDecodeError:
        raise
    except (OSError, AttributeError, ValueError, TypeError, RuntimeError, NotImplementedError,
            EOFError, pydicom.errors.InvalidDicomError) as exc:
        raise DicomDecodeError(f"DICOM decode failed ({type(exc).__name__}): {exc}") from exc
