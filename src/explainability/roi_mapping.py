"""Conservative CBIS annotation discovery: never identify a mask by filename alone."""

import json
import os
from pathlib import Path

import pydicom

from src.data.cbis_ddsm_metadata import read_metadata


def annotation_role(ds: pydicom.dataset.Dataset) -> str:
    """Use explicit image-level descriptors; unknown or conflicting roles stay unknown."""
    description = " ".join(str(getattr(ds, key, "")) for key in ("SeriesDescription", "ImageComments", "ImageType")).lower()
    mask = "roi mask" in description or "roi_mask" in description
    crop = "cropped" in description
    if mask and not crop:
        return "mask"
    if crop and not mask:
        return "cropped"
    return "unknown"


def annotation_references(root: Path, images: list[dict], partition: str) -> list[dict]:
    """Keep every source lesion; partition-specific metadata only, no test read by default."""
    official = "test" if partition == "test" else "train"
    source = root / f"data/raw/CBIS-DDSM/metadata/mass_case_description_{official}_set.csv"
    lesions, issues, _ = read_metadata(source, official, root)
    ids = {r["canonical_image_id"] for r in images}
    selected = [r for r in lesions if r["canonical_image_id"] in ids]
    selected_ids = {r["record_id"] for r in selected}
    if any(i["record_identifier"] in selected_ids and i["severity"] in ("ERROR", "CRITICAL") for i in issues):
        raise ValueError("Selected lesion metadata has unresolved errors")
    if {r["canonical_image_id"] for r in selected} != ids:
        raise ValueError("Missing official lesion metadata for selected images")
    for image in images:
        found = [r["record_id"] for r in selected if r["canonical_image_id"] == image["canonical_image_id"]]
        if sorted(found) != sorted(json.loads(image["lesion_record_ids_json"])):
            raise ValueError("Lesion cohort differs from the canonical image traceability")
    return selected


def reference_parts(reference: str) -> tuple[str, str, str]:
    parts = reference.strip().replace("\\", "/").split("/")
    if len(parts) != 4 or not parts[0].startswith(("Mass-Training_P_", "Mass-Test_P_")):
        raise ValueError(f"Invalid annotation reference: {reference!r}")
    return parts[0], parts[1], parts[2]


def discover_annotations(roi_root: Path, cases: set[str]) -> list[dict]:
    """Read headers only below exact requested CBIS case directories.

    Preserve NBIA case directory names. Unrecognised layouts are not guessed.
    Other cases (including test during validation) are pruned before file reads.
    """
    inventory = []
    for directory, subdirs, files in os.walk(roi_root, followlinks=False):
        folder = Path(directory)
        subdirs[:] = sorted(d for d in subdirs if not (folder / d).is_symlink()
                            and (not d.startswith(("Mass-Training_P_", "Mass-Test_P_")) or d in cases))
        if not set(folder.relative_to(roi_root).parts) & cases:
            continue
        for name in sorted(files):
            path = folder / name
            if name == ".gitkeep" or path.is_symlink():
                continue
            row = {"path": path.relative_to(roi_root).as_posix(), "role": "unknown", "read_status": "unreadable",
                   "error": "", "StudyInstanceUID": "", "SeriesInstanceUID": "", "SOPInstanceUID": ""}
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                row.update({key: str(getattr(ds, key, "")) for key in
                            ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "SeriesDescription", "Rows", "Columns")})
                row.update(role=annotation_role(ds), read_status="readable")
            except (OSError, ValueError, EOFError, pydicom.errors.InvalidDicomError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            inventory.append(row)
    return inventory


def map_annotations(lesions: list[dict], inventory: list[dict]) -> list[dict]:
    """Use exact study/series + case components + explicit role; retain all candidates."""
    mappings = []
    for lesion in lesions:
        for role, key in (("mask", "roi_mask_reference"), ("cropped", "cropped_image_reference")):
            row = {"record_id": lesion["record_id"], "canonical_image_id": lesion["canonical_image_id"],
                   "participant_id": lesion["participant_id"], "abnormality_id": lesion["abnormality_id"],
                   "lesion_pathology": lesion["pathology"], "role": role, "reference": lesion[key],
                   "mapping_status": "missing", "path": "", "SOPInstanceUID": "", "reason": "",
                   "candidates_json": "[]", "mapping_method": "exact_case_study_series_and_explicit_role"}
            try:
                case, study, series = reference_parts(lesion[key])
                expected_case = lesion["full_mammogram_reference"].split("/")[0] + "_" + lesion["abnormality_id"]
                if case != expected_case:
                    raise ValueError("Annotation case disagrees with source full image/abnormality ID")
                candidates = [r for r in inventory if r["read_status"] == "readable" and r["StudyInstanceUID"] == study
                              and r["SeriesInstanceUID"] == series and case in Path(r["path"]).parts]
                matching = [r for r in candidates if r["role"] == role]
                unknown = [r for r in candidates if r["role"] == "unknown"]
                row["candidates_json"] = json.dumps([r["path"] for r in matching + unknown])
                if len(matching) == 1 and not unknown:
                    candidate = matching[0]
                    if not candidate["SOPInstanceUID"] or sum(r.get("SOPInstanceUID") == candidate["SOPInstanceUID"] for r in inventory) != 1:
                        row.update(mapping_status="ambiguous", reason="missing_or_duplicate_SOPInstanceUID")
                    else:
                        row.update(mapping_status="unique", path=candidate["path"], SOPInstanceUID=candidate["SOPInstanceUID"])
                elif matching or unknown:
                    row.update(mapping_status="ambiguous", reason="multiple_candidates_or_unknown_role")
            except ValueError as exc:
                row["reason"] = str(exc)
            mappings.append(row)
    return mappings
