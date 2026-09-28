"""Header-only inventory and exact UID mapping of full mammograms."""

from pathlib import Path
from collections import Counter

import pydicom

from src.data.cbis_ddsm_metadata import issue

TAGS = ["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "Modality",
        "Rows", "Columns", "PhotometricInterpretation", "BitsAllocated", "BitsStored",
        "PixelRepresentation"]
INVENTORY_FIELDS = ["dicom_relative_path", *TAGS, "TransferSyntaxUID", "read_status", "error"]


def inventory_dicoms(images_dir: Path, root: Path) -> list[dict]:
    """Inspect all files (including extensionless DICOMs); never decode pixels."""
    if not images_dir.resolve().is_relative_to(root.resolve()):
        raise ValueError("Inventory must remain inside the repository")
    result = []
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        row = dict.fromkeys(INVENTORY_FIELDS, "")
        row["dicom_relative_path"] = path.relative_to(root).as_posix()
        try:
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("File resolves outside repository")
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            row.update({tag: str(getattr(ds, tag, "")) for tag in TAGS})
            row["TransferSyntaxUID"] = str(getattr(ds.file_meta, "TransferSyntaxUID", ""))
            row["read_status"] = "readable"
        except (OSError, ValueError, EOFError, TypeError, pydicom.errors.InvalidDicomError) as exc:
            row.update(read_status="unreadable", error=f"{type(exc).__name__}: {exc}")
        result.append(row)
    return result


def map_images(images: list[dict], inventory: list[dict]) -> tuple[list[dict], list[dict]]:
    """Require exact study/series UIDs from the full-image reference.

    Filenames and arbitrary DICOM PatientID values are never identity evidence.
    More than one file in the referenced series remains ambiguous.
    """
    mapped, issues, used = [], [], set()
    sop_counts = Counter(d["SOPInstanceUID"] for d in inventory
                         if d["read_status"] == "readable" and d["SOPInstanceUID"])
    for sop, count in sop_counts.items():
        if count > 1:
            problem = issue({}, "duplicate_SOPInstanceUID", stage="mapping")
            problem["reference"] = sop
            issues.append(problem)
    for image in images:
        row = dict(image)
        parts = row["full_mammogram_reference"].split("/")
        candidates = [d for d in inventory if len(parts) == 4 and d["read_status"] == "readable"
                      and d["StudyInstanceUID"] == parts[1] and d["SeriesInstanceUID"] == parts[2]]
        row.update(dicom_relative_path="", mapping_method="exact_full_reference_study_series_uid",
                   mapping_status="unmapped" if not candidates else "ambiguous")
        if len(candidates) == 1:
            candidate = candidates[0]
            if (candidate["Modality"] == "MG" and candidate["SOPInstanceUID"] and
                    candidate["Rows"].isdigit() and int(candidate["Rows"]) > 0 and
                    candidate["Columns"].isdigit() and int(candidate["Columns"]) > 0):
                row.update(dicom_relative_path=candidate["dicom_relative_path"], mapping_status="uniquely_mapped")
                used.add(candidate["dicom_relative_path"])
            else:
                row["mapping_status"] = "unmapped"
        if row["mapping_status"] != "uniquely_mapped":
            problem = issue(row, row["mapping_status"], stage="mapping")
            problem["reference"] += " | candidates=" + repr([c["dicom_relative_path"] for c in candidates])
            issues.append(problem)
        mapped.append(row)
    for dicom in inventory:
        if dicom["dicom_relative_path"] not in used:
            problem = issue({}, "orphan_DICOM", "WARNING", "mapping")
            problem["reference"] = dicom["dicom_relative_path"]
            issues.append(problem)
    return mapped, issues
