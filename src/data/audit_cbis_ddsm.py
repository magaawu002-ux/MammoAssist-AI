"""Run reproducible metadata audits; gate full execution on verified inputs."""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data.cbis_ddsm_metadata import (ISSUE_FIELDS, aggregate_images, issue,
    participant_labels, participant_overlap, read_metadata, write_csv)
from src.data.dicom_inventory import INVENTORY_FIELDS, inventory_dicoms, map_images
from src.data.provenance import write_provenance
from src.data.splits import create_splits
from src.data.validate_splits import validate_splits
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT


def class_counts(values: list) -> dict:
    """Include unresolved targets explicitly."""
    return dict(Counter("unresolved" if v is None else str(v) for v in values))


def distribution_figure(groups: dict[str, dict], destination: Path, title: str) -> None:
    """Save a plain publication-readable grouped bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = list(groups)
    for offset, label, name, colour in [(-0.2, "0", "Benign", "#4477AA"),
                                       (0.2, "1", "Malignant", "#AA3377")]:
        bars = ax.bar([i + offset for i in range(len(categories))],
                      [groups[c].get(label, 0) for c in categories], width=0.4,
                      label=name, color=colour)
        ax.bar_label(bars, padding=3)
    ax.set(title=title, xlabel="Partition", ylabel="Full mammograms",
           xticks=range(len(categories)), xticklabels=categories)
    ax.legend()
    ax.margins(y=0.15)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def audit(root: Path = PROJECT_ROOT, full: bool = False) -> dict:
    """Audit actual source CSVs; metadata-only never touches the image directory."""
    root = root.resolve()
    config = load_config(root / "config/project.yaml")
    expected = {"dataset": "CBIS-DDSM", "subset": "mass", "task": "binary_classification",
                "image_type": "full_mammogram", "positive_class": "MALIGNANT"}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Configuration is incompatible with this mass/full-mammogram audit")
    raw = root / "data/raw" / config["dataset"]
    interim, evidence = root / "data/interim", root / "reports/evidence"
    sources, lesions, issues, source_audits = [], [], [], []
    for partition in ("train", "test"):
        source = raw / "metadata" / f"mass_case_description_{partition}_set.csv"
        rows, problems, details = read_metadata(source, partition, root)
        if not rows:
            raise ValueError(f"Empty required metadata: {source.name}")
        sources.append((source, "metadata"))
        lesions.extend(rows)
        issues.extend(problems)
        source_audits.append(details)
    sources.extend((p, "manifest") for p in sorted((raw / "manifests").rglob("*"))
                   if p.is_file() and p.name != ".gitkeep")
    provenance = write_provenance(root, sources, config, evidence / "data_provenance.json")
    images, image_issues = aggregate_images(lesions)
    issues.extend(image_issues)
    overlap = participant_overlap(lesions)
    for participant in overlap:
        issues.append(issue({"participant_id": participant}, "official_train_test_participant_overlap", "CRITICAL"))
    summary = {"mode": "full" if full else "metadata-only", "sources": source_audits,
        "lesion_rows": len(lesions), "images": len(images),
        "participants_total": len(participant_labels(images)),
        "participant_overlap": overlap,
        "mixed_pathology_images": [r["canonical_image_id"] for r in images if r["mixed_pathology"]],
        "partitions": {}, "dicom_status": "PENDING / INCOMPLETE: inventory and mapping not run",
        "split_status": "NOT FINAL: DICOM mapping pending", "issues": {}}
    for part in ("train", "test"):
        subset = [r for r in images if r["official_partition"] == part]
        lesion_subset = [r for r in lesions if r["official_partition"] == part]
        summary["partitions"][part] = {
            "lesion_rows": len(lesion_subset), "images": len(subset),
            "participants": len(participant_labels(subset)),
            "image_classes": class_counts([r["binary_label"] for r in subset]),
            "participant_classes": class_counts(list(participant_labels(subset).values())),
            "images_per_participant": dict(sorted(Counter(r["participant_id"] for r in subset).items())),
            "abnormality_rows_per_participant": dict(sorted(Counter(r["participant_id"] for r in lesion_subset).items())),
            "unique_abnormalities_per_participant": {
                p: len({(r["laterality"], r["abnormality_id"]) for r in lesion_subset if r["participant_id"] == p})
                for p in sorted(participant_labels(subset))},
        }
    failure = None
    if full:
        inventory = inventory_dicoms(raw / "images", root)
        write_csv(interim / "cbis_ddsm_dicom_inventory.csv", inventory, INVENTORY_FIELDS)
        images, mapping_issues = map_images(images, inventory)
        issues.extend(mapping_issues)
        write_csv(interim / "cbis_ddsm_mapping_issues.csv", mapping_issues, ISSUE_FIELDS)
        for entry in inventory:
            if entry["read_status"] != "readable":
                problem = issue({}, "unreadable_DICOM", stage="inventory")
                problem["reference"] = entry["dicom_relative_path"] + ": " + entry["error"]
                issues.append(problem)
        summary["dicom_status"] = {"inventory_files": len(inventory),
            "read_status": dict(Counter(d["read_status"] for d in inventory)),
            "mapping_status": dict(Counter(r["mapping_status"] for r in images))}
        try:
            if any(p["severity"] in ("ERROR", "CRITICAL") for p in issues):
                raise ValueError("Metadata, inventory or mapping errors require resolution")
            splits, strategy = create_splits(images, config["random_seed"])
            validate_splits(splits, root, evidence / "leakage_validation.txt")
            # Never silently replace an already published holdout or split cohort.
            import csv
            for part, records in splits.items():
                existing = root / "data/splits" / f"{part}.csv"
                if existing.exists():
                    with existing.open(encoding="utf-8", newline="") as handle:
                        previous = list(csv.DictReader(handle))
                    serialized = [{k: "" if v is None else str(v) for k, v in r.items()} for r in records]
                    if previous != serialized:
                        raise ValueError(f"Existing {part} manifest differs; refusing to overwrite frozen splits")
            for part, records in splits.items():
                write_csv(root / "data/splits" / f"{part}.csv", records)
            distribution_figure({part: class_counts([r["binary_label"] for r in records])
                for part, records in splits.items()}, root / "results/figures/dataset_split_distribution.png",
                "Validated participant-level split distribution")
            summary["split_status"] = f"FINAL: validated; {strategy}; seed={config['random_seed']}"
        except ValueError as exc:
            failure = str(exc)
            summary["split_status"] = f"NOT FINAL: {failure}"
            (evidence / "leakage_validation.txt").write_text(
                json.dumps({"conclusion": "FAIL", "failure_reasons": [failure],
                            "note": "Final split validation not completed for this run; no new splits certified"},
                           indent=2) + "\n", encoding="utf-8")
    write_csv(interim / "cbis_ddsm_mass_lesion_level.csv", lesions)
    write_csv(interim / "cbis_ddsm_mass_image_level.csv", images)
    write_csv(interim / "cbis_ddsm_mass_issues.csv", issues, ISSUE_FIELDS)
    # No cohort filtering occurs. Actual blocking problems are logged as withheld
    # from finalisation, never as permission to silently drop official test cases.
    exclusions = [dict(record_identifier=p["record_identifier"], participant=p["participant"],
        official_partition=p["official_partition"], exclusion_reason=p["reason"],
        processing_stage=p["processing_stage"], reference=p["reference"],
        action="withheld_from_finalisation; source record preserved")
        for p in issues if p["severity"] in ("ERROR", "CRITICAL")]
    write_csv(interim / "cbis_ddsm_exclusion_log.csv", exclusions,
        ["record_identifier", "participant", "official_partition", "exclusion_reason",
         "processing_stage", "reference", "action"])
    summary["issues"] = dict(Counter(p["reason"] for p in issues))
    distribution_figure({p: d["image_classes"] for p, d in summary["partitions"].items()},
        root / "results/figures/dataset_class_distribution.png", "CBIS-DDSM mass: metadata image-level classes")
    # Full per-participant tables live in machine-readable evidence to keep the report readable.
    (evidence / "data_audit_details.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    concise = dict(summary)
    concise["partitions"] = {p: {k: v for k, v in d.items() if not k.endswith("per_participant")}
                              for p, d in summary["partitions"].items()}
    report = ["# CBIS-DDSM mass data audit", "", f"Generated: {provenance['generated_at_utc']}",
        "", "Counts describe metadata full-mammogram references, not verified local images.",
        "No model, preprocessing or clinical performance evaluation has run.",
        "Source row numbers are 1-based data-record ordinals; quoted multiline CSV fields are preserved.",
        "Mixed pathology means distinct raw pathology categories (including the two benign categories).",
        "Abnormality counts use distinct source abnormality IDs per image. Participant abnormality counts",
        "use (laterality, abnormality ID); row counts are also retained. These are metadata identifiers,",
        "not independently verified biological lesions. No source rows are removed.",
        "See data_audit_details.json for images and abnormalities per participant.", "",
        "Existing split artifacts, if any, are not certified by a metadata-only run.", "",
        "## Observed results", "", "```json", json.dumps(concise, indent=2), "```", "",
        "## Source provenance", "", "```json", json.dumps(provenance, indent=2), "```", ""]
    (evidence / "data_audit_summary.md").write_text("\n".join(report), encoding="utf-8")
    if failure:
        raise ValueError(f"Full audit failed; evidence written. {failure}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--metadata-only", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    try:
        result = audit(full=args.full)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Audit failed: {exc}\n")
    print(f"{result['mode']}: {result['lesion_rows']} lesion rows; {result['images']} image references")
    print(f"Official participant overlap: {len(result['participant_overlap'])}")
    print(result["split_status"])


if __name__ == "__main__":
    main()
