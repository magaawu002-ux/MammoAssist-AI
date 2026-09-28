"""Lossless source-row ingestion and conservative mammogram aggregation."""

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

LABELS = {"MALIGNANT": 1, "BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0}
ALIASES = {
    "patient_id": "original_participant_id", "breast_density": "breast_density",
    "left or right breast": "laterality", "image view": "view",
    "abnormality id": "abnormality_id", "abnormality type": "abnormality_type",
    "mass shape": "mass_shape", "mass margins": "mass_margins",
    "assessment": "assessment", "pathology": "pathology", "subtlety": "subtlety",
    "image file path": "full_mammogram_reference",
    "cropped image file path": "cropped_image_reference",
    "ROI mask file path": "roi_mask_reference",
}
ISSUE_FIELDS = ["record_identifier", "participant", "official_partition", "reason",
                "severity", "processing_stage", "reference"]


def normalize_participant(value: str) -> tuple[str, str]:
    """Accept exact CBIS IDs or documented mass case-name wrappers only."""
    value = value.strip()
    if re.fullmatch(r"P_\d{5}", value):
        return value, "exact_cbis_case_id"
    match = re.fullmatch(r"Mass-(?:Training|Test)_(P_\d{5})_(?:LEFT|RIGHT)_(?:CC|MLO)(?:_\d+)?", value)
    return (match[1], "cbis_mass_case_name") if match else ("", "unparseable")


def pathology_label(value: str) -> int | None:
    """Unknown and blank labels remain unresolved."""
    return LABELS.get(value.strip())


def issue(record: dict, reason: str, severity: str = "ERROR",
          stage: str = "metadata") -> dict:
    return dict(zip(ISSUE_FIELDS, [record.get("record_id", record.get("canonical_image_id", "")),
        record.get("participant_id", ""), record.get("official_partition", ""), reason,
        severity, stage, record.get("full_mammogram_reference", "")]))


def read_metadata(path: Path, partition: str, root: Path) -> tuple[list[dict], list[dict], dict]:
    """Preserve every parsed row and every original field, including unknowns."""
    records, issues = [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = set(ALIASES) - set(columns)
        if missing:
            raise ValueError(f"{path.name}: missing source columns {sorted(missing)}")
        for ordinal, source in enumerate(reader, 1):
            record = {target: (source.get(column) or "").strip()
                      for column, target in ALIASES.items()}
            record.update(record_id=f"{path.name}:{ordinal}", source_file=path.relative_to(root).as_posix(),
                          source_row=ordinal, source_line_end=reader.line_num,
                          source_fields_json=json.dumps(source, sort_keys=True), official_partition=partition)
            record["participant_id"], record["normalization_method"] = normalize_participant(record["original_participant_id"])
            record["binary_label"] = pathology_label(record["pathology"])
            for condition, reason in [
                (not record["participant_id"], "unparseable_participant"),
                (record["binary_label"] is None, "unsupported_pathology"),
                (record["abnormality_type"] != "mass", "non_mass_abnormality"),
                (record["laterality"] not in ("LEFT", "RIGHT"), "invalid_laterality"),
                (record["view"] not in ("CC", "MLO"), "invalid_view"),
                (not record["abnormality_id"], "missing_abnormality_id"),
                (None in source or any(v is None for v in source.values()), "malformed_csv_row"),
            ]:
                if condition:
                    issues.append(issue(record, reason))
            reference = record["full_mammogram_reference"].replace("\\", "/")
            record["full_mammogram_reference"] = reference
            parts = reference.split("/")
            expected = "Mass-Training" if partition == "train" else "Mass-Test"
            case = f"{expected}_{record['participant_id']}_{record['laterality']}_{record['view']}"
            valid = (len(parts) == 4 and parts[0] == case and
                     all(re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", p) for p in parts[1:3]) and
                     parts[3].lower().endswith(".dcm"))
            if not valid:
                issues.append(issue(record, "invalid_or_inconsistent_full_mammogram_reference"))
            key = reference or record["record_id"]
            record["canonical_image_id"] = "mass_" + hashlib.sha256(key.encode()).hexdigest()[:24]
            records.append(record)
    return records, issues, {"source_file": path.relative_to(root).as_posix(),
                            "rows": len(records), "columns": columns,
                            "pathology_values": sorted({r["pathology"] for r in records})}


def aggregate_images(lesions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Collapse full-image references without removing any source lesion rows."""
    groups = defaultdict(list)
    for row in lesions:
        groups[row["canonical_image_id"]].append(row)
    images, issues = [], []
    for image_id, rows in sorted(groups.items()):
        first = rows[0]
        image = {k: first[k] for k in ("participant_id", "original_participant_id", "normalization_method",
            "official_partition", "laterality", "view", "full_mammogram_reference")}
        labels = [r["binary_label"] for r in rows]
        # Unknown lesions prevent a final benign conclusion; malignant evidence is retained.
        label = 1 if 1 in labels else (None if None in labels else 0)
        image.update(canonical_image_id=image_id, binary_label=label,
            pathology="MALIGNANT" if label == 1 else ("BENIGN" if label == 0 else "UNRESOLVED"),
            abnormality_count=len({r["abnormality_id"] for r in rows}), lesion_row_count=len(rows),
            mixed_pathology=len({r["pathology"] for r in rows}) > 1,
            pathology_values_json=json.dumps(sorted({r["pathology"] for r in rows})),
            lesion_record_ids_json=json.dumps([r["record_id"] for r in rows]),
            dicom_relative_path="", mapping_status="pending", mapping_method="not_run")
        for key in ("participant_id", "official_partition", "laterality", "view"):
            if len({r[key] for r in rows}) != 1:
                issues.append(issue(image, f"conflicting_{key}", "CRITICAL"))
        if len({r["abnormality_id"] for r in rows}) < len(rows):
            issues.append(issue(image, "repeated_abnormality_rows", "WARNING"))
        images.append(image)
    return images, issues


def participant_labels(images: list[dict]) -> dict[str, int | None]:
    groups = defaultdict(list)
    for row in images:
        if row["participant_id"]:
            groups[row["participant_id"]].append(row["binary_label"])
    return {p: 1 if 1 in labels else (None if None in labels else 0) for p, labels in groups.items()}


def participant_overlap(images: list[dict]) -> list[str]:
    sets = [{r["participant_id"] for r in images if r["official_partition"] == part and r["participant_id"]}
            for part in ("train", "test")]
    return sorted(sets[0] & sets[1])


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    """Write deterministic UTF-8 CSVs, including headers for empty issue logs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
