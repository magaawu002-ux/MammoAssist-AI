"""Reproducibility metadata and scoped summaries for eventual cache runs."""

import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.data.provenance import sha256_file


def implementation_identity() -> dict:
    """Record runtime versions and implementation hashes, including dirty edits."""
    package = Path(__file__).parent
    optional_decoders = {}
    for name in ("pillow", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "pyjpegls", "python-gdcm"):
        try:
            optional_decoders[name] = version(name)
        except PackageNotFoundError:
            optional_decoders[name] = None
    return {"python": platform.python_version(), "optional_decoder_packages": optional_decoders,
        "packages": {p: version(p) for p in ("numpy", "pydicom", "opencv-python")},
        "source_sha256": {p.name: sha256_file(p) for p in sorted(package.glob("*.py"))}}


def write_summary(root: Path, split: str, records: list[dict], provenance: dict) -> None:
    """Report only the attempted split; failed runs never claim usable manifests."""
    succeeded = [r for r in records if r["processing_status"] == "success"]
    metadata = [json.loads(r["preprocessing_metadata_json"]) for r in succeeded]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    summary = dict(provenance, generated_at_utc=datetime.now(timezone.utc).isoformat(), git_commit=commit,
        scope=f"Only attempted split: {split}", succeeded=len(succeeded), failed=len(records) - len(succeeded),
        status="COMPLETE" if len(succeeded) == len(records) else "FAILED; no processed manifest published",
        photometric_counts=dict(Counter(m["photometric_interpretation"] for m in metadata)),
        crop_usage=dict(Counter(str(m["crop_applied"]) for m in metadata)),
        dimensions_before=dict(Counter(f"{m['original_height']}x{m['original_width']}" for m in metadata)),
        dimensions_after_crop=dict(Counter(f"{m['crop_height']}x{m['crop_width']}" for m in metadata)),
        padding=dict(Counter(str([m[f"padding_{s}"] for s in ("top", "bottom", "left", "right")]) for m in metadata)),
        failures=[{"canonical_image_id": r["canonical_image_id"], "reason": r["processing_error"]}
                  for r in records if r["processing_status"] != "success"])
    directory = root / "reports/evidence"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"preprocessing_{split}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (directory / "preprocessing_summary.md").write_text(
        "# Preprocessing execution summary\n\nLatest attempted split only; other partitions are not certified here.\n"
        "Per-split JSON files retain prior partition summaries. No model evaluation is performed.\n\n```json\n"
        + json.dumps(summary, indent=2) + "\n```\n", encoding="utf-8")
