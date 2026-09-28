"""Preflight the entire cohort before allowing any cache writes or pixel decoding."""

import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import pydicom

from src.data.provenance import sha256_file
from src.data.validate_splits import validate_splits
from src.preprocessing.settings import preprocessing_settings

PARTITIONS = ("train", "validation", "test")


def local_path(root: Path, reference: str, prefix: tuple[str, ...]) -> Path:
    """Require portable, contained paths; reject traversal and escaping symlinks."""
    if not isinstance(reference, str) or not reference or "\\" in reference or ":" in reference:
        raise ValueError(f"Nonportable relative path: {reference!r}")
    relative = PurePosixPath(reference)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:len(prefix)] != prefix:
        raise ValueError(f"Path is outside required {'/'.join(prefix)}: {reference}")
    path = root / reference
    if not path.resolve().is_relative_to((root.joinpath(*prefix)).resolve()) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository: {reference}")
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a required, nonempty manifest and reject malformed rows."""
    if not path.is_file():
        raise ValueError(f"Required validated manifest missing: {path.name}; run the full data audit after download completion")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows or len(fields) != len(set(fields)) or any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f"Empty or malformed manifest: {path.name}")
    return rows


def verify_full_reference(row: dict[str, str], header: pydicom.dataset.Dataset) -> None:
    """Cross-check full-image case name, canonical ID and actual study/series UIDs."""
    required = ("participant_id", "official_partition", "laterality", "view", "full_mammogram_reference",
                "canonical_image_id", "mapping_method")
    if any(not row.get(k) for k in required):
        raise ValueError("Missing full-mammogram traceability fields")
    participant = row["participant_id"]
    if not re.fullmatch(r"P_\d{5}", participant) or row["laterality"] not in ("LEFT", "RIGHT") or row["view"] not in ("CC", "MLO"):
        raise ValueError("Invalid CBIS participant/laterality/view")
    prefix = "Mass-Training" if row["official_partition"] == "train" else "Mass-Test"
    parts = row["full_mammogram_reference"].split("/")
    expected_case = f"{prefix}_{participant}_{row['laterality']}_{row['view']}"
    if (len(parts) != 4 or parts[0] != expected_case or not parts[3].endswith(".dcm")
            or any(not re.fullmatch(r"\d+(?:\.\d+)+", v) for v in parts[1:3])):
        raise ValueError("Reference is not a verified full mammogram (ROI/cropped references prohibited)")
    expected_id = "mass_" + hashlib.sha256(row["full_mammogram_reference"].encode()).hexdigest()[:24]
    if row["canonical_image_id"] != expected_id or row["mapping_method"] != "exact_full_reference_study_series_uid":
        raise ValueError("Unverified image identity or mapping method")
    if (str(getattr(header, "StudyInstanceUID", "")) != parts[1]
            or str(getattr(header, "SeriesInstanceUID", "")) != parts[2]
            or str(getattr(header, "Modality", "")) != "MG"
            or not getattr(header, "SOPInstanceUID", "")):
        raise ValueError("Actual DICOM header does not match mapped full-mammogram reference")


def validated_manifests(root: Path, config: dict) -> tuple[dict, dict]:
    """Require successful full-audit evidence, then independently revalidate all splits.

    This function is read-only. It never scans an image folder or decodes pixels.
    """
    preprocessing_settings(config)
    splits = {part: read_rows(root / "data/splits" / f"{part}.csv") for part in PARTITIONS}
    evidence = root / "reports/evidence"
    required = [evidence / name for name in ("data_audit_details.json", "leakage_validation.txt", "data_provenance.json")]
    if any(not p.is_file() for p in required):
        raise ValueError("Missing full-audit / leakage validation prerequisites")
    details, leakage, provenance = [json.loads(p.read_text(encoding="utf-8")) for p in required]
    if details.get("mode") != "full" or not details.get("split_status", "").startswith("FINAL:") or leakage.get("conclusion") != "PASS":
        raise ValueError("Full audit and leakage validation must both certify final splits")
    sources = provenance.get("sources", [])
    metadata_names = {f"mass_case_description_{p}_set.csv" for p in ("train", "test")}
    if {s.get("source_filename") for s in sources if s.get("source_category") == "metadata"} != metadata_names:
        raise ValueError("Missing official metadata provenance")
    for source in sources:
        path = local_path(root, source["repository_relative_path"], ("data", "raw", "CBIS-DDSM"))
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise ValueError("Source provenance changed; repeat the full data audit")
    result = validate_splits(splits, root)
    if result != leakage:
        raise ValueError("Saved leakage evidence does not match current split validation")
    canonical = read_rows(root / "data/interim/cbis_ddsm_mass_image_level.csv")
    by_id = {r.get("canonical_image_id"): r for r in canonical}
    all_rows = [r for rows in splits.values() for r in rows]
    if (len(by_id) != len(canonical) or set(by_id) != {r["canonical_image_id"] for r in all_rows}
            or details.get("images") != len(all_rows)):
        raise ValueError("Split cohort differs from the fully audited image cohort")
    seen_paths, seen_sops, source_stats = set(), set(), {}
    for row in all_rows:
        audited = by_id[row["canonical_image_id"]]
        if any(row.get(k) != v for k, v in audited.items()):
            raise ValueError("Split row differs from audited canonical image metadata")
        path = local_path(root, row["dicom_relative_path"], ("data", "raw", "CBIS-DDSM", "images"))
        if path.resolve() in seen_paths:
            raise ValueError("Duplicate resolved DICOM path")
        seen_paths.add(path.resolve())
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True)
        except (OSError, ValueError, EOFError, pydicom.errors.InvalidDicomError) as exc:
            raise ValueError(f"Unreadable mapped DICOM header: {row['dicom_relative_path']}: {exc}") from exc
        verify_full_reference(row, header)
        sop = str(header.SOPInstanceUID)
        if sop in seen_sops:
            raise ValueError("Duplicate DICOM SOPInstanceUID")
        seen_sops.add(sop)
        stat = path.stat()
        source_stats[row["canonical_image_id"]] = {"byte_size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sop_instance_uid": sop}
    return splits, {"split_sha256": {p: sha256_file(root / "data/splits" / f"{p}.csv") for p in PARTITIONS},
                    "validation": result, "source_files": source_stats}
