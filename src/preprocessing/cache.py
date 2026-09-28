"""Build canonical caches only from the complete, validated split cohort."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from src.data.cbis_ddsm_metadata import write_csv
from src.data.provenance import sha256_file
from src.preprocessing.canonical import preprocess_dicom
from src.preprocessing.evidence import implementation_identity, write_summary
from src.preprocessing.manifests import PARTITIONS, local_path, validated_manifests
from src.preprocessing.settings import preprocessing_settings
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT


def cache_relative_path(image_id: str, settings: dict, fingerprint: str) -> str:
    """Stable, portable cache filename independent of participant/label text."""
    digest = hashlib.sha256(image_id.encode()).hexdigest()
    if not fingerprint or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("Cache fingerprint must be hexadecimal")
    return (Path(settings["cache_directory"]) / settings["version"] / fingerprint / f"{digest}.npy").as_posix()


def build_cache(split: str, root: Path = PROJECT_ROOT, *, dry_run: bool = False) -> dict:
    """Preflight all partitions, then process one; never omit failures from logs.

    No outputs are written in dry-run. On processing failure an attempt log and
    FAIL summary are written, but no final processed manifest is published.
    """
    if split not in PARTITIONS:
        raise ValueError(f"Unknown split: {split}")
    root = root.resolve()
    config = load_config(root / "config/project.yaml")
    settings = preprocessing_settings(config)
    splits, validation = validated_manifests(root, config)
    local_path(root, settings["cache_directory"], ("data", "processed"))
    if dry_run:
        return {"status": "VALIDATED DRY RUN; pixels not decoded; no cache written", "split": split,
                "image_count": len(splits[split])}
    destination = local_path(root, f"data/processed/manifests/{split}_processed.csv", ("data", "processed"))
    if destination.exists():
        raise ValueError("Processed manifest already exists; review/archive the previous experiment before rebuilding")
    identity = {"settings": settings, "implementation": implementation_identity(),
                "split_sha256": validation["split_sha256"]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    records = []
    for row in sorted(splits[split], key=lambda r: r["canonical_image_id"]):
        record = dict(row, processing_status="failed", processing_error="", cache_relative_path="",
                      cache_sha256="", preprocessing_version=settings["version"],
                      cache_dtype=settings["cache_dtype"], preprocessing_metadata_json="{}")
        try:
            source = local_path(root, row["dicom_relative_path"], ("data", "raw", "CBIS-DDSM", "images"))
            expected = validation["source_files"][row["canonical_image_id"]]
            result = preprocess_dicom(source, config)
            stat = source.stat()
            if (stat.st_size != expected["byte_size"] or stat.st_mtime_ns != expected["mtime_ns"]
                    or result.metadata["SOPInstanceUID"] != expected["sop_instance_uid"]):
                raise ValueError("Source DICOM changed after preflight")
            relative = cache_relative_path(row["canonical_image_id"], settings, fingerprint)
            output = local_path(root, relative, ("data", "processed"))
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, result.image, allow_pickle=False)
            temporary.replace(output)
            record.update(processing_status="success", cache_relative_path=relative,
                cache_sha256=sha256_file(output), preprocessing_metadata_json=json.dumps(result.metadata, sort_keys=True))
        except (ValueError, OSError, cv2.error) as exc:
            record["processing_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    # Check split inputs again before publishing: changed manifests invalidate the run.
    if any(sha256_file(root / "data/splits" / f"{p}.csv") != digest for p, digest in validation["split_sha256"].items()):
        for record in records:
            record.update(processing_status="failed", processing_error="Split manifests changed during build")
    failures = sum(r["processing_status"] != "success" for r in records)
    if not failures:
        try:
            temporary = destination.with_suffix(".csv.tmp")
            write_csv(temporary, records)
            temporary.replace(destination)
        except OSError as exc:
            for record in records:
                record.update(processing_status="failed", processing_error=f"Manifest publication failed: {exc}")
            failures = len(records)
    evidence = root / "reports/evidence"
    write_csv(evidence / f"preprocessing_{split}_attempt.csv", records)
    write_summary(root, split, records, dict(identity, fingerprint=fingerprint, source_files=validation["source_files"]))
    if failures:
        raise ValueError(f"Cache build failed for {failures} image(s); see preprocessing_{split}_attempt.csv; no final processed manifest written")
    return {"status": "COMPLETE", "split": split, "succeeded": len(records), "failed": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=PARTITIONS, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate prerequisites and headers without decoding pixels or writing outputs")
    args = parser.parse_args()
    try:
        result = build_cache(args.split, dry_run=args.dry_run)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Preprocessing refused: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
