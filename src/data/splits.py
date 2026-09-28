"""Deterministic participant splitting, conditional on complete mapping."""

import math

from sklearn.model_selection import train_test_split

from src.data.cbis_ddsm_metadata import participant_labels, participant_overlap


def create_splits(images: list[dict], seed: int, validation_fraction: float = 0.2) -> tuple[dict, str]:
    """Keep the official test untouched; stratify training participants if feasible."""
    if participant_overlap(images):
        raise ValueError("CRITICAL official train/test participant leakage")
    if not images or any(r.get("mapping_status") != "uniquely_mapped" or
                         not r.get("dicom_relative_path") or not r.get("participant_id") or
                         r.get("binary_label") not in (0, 1) or
                         r.get("official_partition") not in ("train", "test") for r in images):
        raise ValueError("Incomplete metadata or DICOM mapping; splits NOT FINAL")
    labels = participant_labels([r for r in images if r["official_partition"] == "train"])
    ids = sorted(labels)
    if len(ids) < 2:
        raise ValueError("At least two official training participants required")
    targets = [labels[p] for p in ids]
    classes = set(targets)
    n_val = math.ceil(len(ids) * validation_fraction)
    feasible = (len(classes) > 1 and min(targets.count(c) for c in classes) >= 2
                and n_val >= len(classes) and len(ids) - n_val >= len(classes))
    train, validation = train_test_split(ids, test_size=validation_fraction,
        random_state=seed, stratify=targets if feasible else None)
    assignments = {p: "train" for p in train} | {p: "validation" for p in validation}
    result = {p: [] for p in ("train", "validation", "test")}
    for image in sorted(images, key=lambda r: r["canonical_image_id"]):
        part = "test" if image["official_partition"] == "test" else assignments[image["participant_id"]]
        result[part].append(dict(image, final_partition=part))
    return result, "stratified" if feasible else "unstratified: class sizes do not support stratification"
