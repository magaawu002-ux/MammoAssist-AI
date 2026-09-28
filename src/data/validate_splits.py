"""Fail loudly on participant leakage and invalid classifier manifests."""

import json
from collections import Counter
from pathlib import Path


def validate_splits(splits: dict[str, list[dict]], root: Path,
                    evidence: Path | None = None) -> dict:
    """Validate complete split records and optionally persist PASS/FAIL evidence."""
    errors, seen_images, seen_paths, participants = [], set(), set(), {}
    counts = {}
    if set(splits) != {"train", "validation", "test"}:
        errors.append("Expected train, validation and test partitions")
    for part, rows in splits.items():
        participants[part] = {r.get("participant_id", "") for r in rows}
        counts[part] = {"participants": len(participants[part]), "images": len(rows),
                        "classes": dict(Counter(str(r.get("binary_label")) for r in rows))}
        if not rows:
            errors.append(f"Empty {part} partition")
        for row in rows:
            image, path = row.get("canonical_image_id", ""), row.get("dicom_relative_path", "")
            if not image or image in seen_images:
                errors.append(f"Missing/duplicate image: {image}")
            if not path or path in seen_paths:
                errors.append(f"Missing/duplicate path: {path}")
            seen_images.add(image)
            seen_paths.add(path)
            if not row.get("participant_id"):
                errors.append("Missing participant")
            if str(row.get("binary_label")) not in ("0", "1"):
                errors.append(f"Invalid label: {image}")
            expected = "test" if part == "test" else "train"
            if row.get("official_partition") != expected or row.get("final_partition") != part:
                errors.append(f"Official/final partition violation: {image}")
            local = root / path
            if Path(path).is_absolute() or not local.resolve().is_relative_to(root.resolve()) or not local.is_file():
                errors.append(f"Missing or unsafe mapped path: {path}")
            if row.get("mapping_status") != "uniquely_mapped":
                errors.append(f"Mapping incomplete: {image}")
    overlaps = {}
    names = sorted(participants)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            common = participants[first] & participants[second]
            overlaps[f"{first}/{second}"] = len(common)
            if common:
                errors.append(f"Participant leakage {first}/{second}: {sorted(common)}")
    result = {"conclusion": "FAIL" if errors else "PASS", "counts": counts,
              "pairwise_participant_overlaps": overlaps, "failure_reasons": errors}
    if evidence:
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise ValueError("Split validation FAIL: " + "; ".join(errors))
    return result
