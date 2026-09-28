"""Staged annotation audit, validation localisation, and locked final localisation."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from src.data.cbis_ddsm_metadata import write_csv
from src.data.provenance import sha256_file
from src.explainability.alignment import transform_mask, union_masks
from src.explainability.coordinate_protocol import PROTOCOL, bound_hash, load_coordinate_protocol
from src.explainability.localisation_metrics import localisation_metrics
from src.explainability.roi_audit import audit_roi, digest, localisation_config, write_audit
from src.explainability.roi_masks import decode_mask
from src.preprocessing.manifests import local_path
from src.training.data import load_canonical
from src.utils.paths import PROJECT_ROOT


def write_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def protocol_identity(root: Path) -> dict:
    """Bind local choices and implementation before any test localisation."""
    config = localisation_config(root)
    coordinate_protocol = load_coordinate_protocol(root)
    selection = json.loads((root / "results/experiments/final_selection.json").read_text(encoding="utf-8"))
    checkpoint = local_path(root, selection["run"] + "/best_model.keras", ("results", "experiments"))
    if selection.get("status") != "LOCKED_FOR_FINAL_TEST" or sha256_file(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("Selected classifier checkpoint/lock mismatch")
    package = Path(__file__).parent
    source_hashes = {p.name: sha256_file(p) for p in sorted(package.glob("*.py"))}
    return {"protocol": config, "implementation_sha256": source_hashes,
            "coordinate_protocol_sha256": bound_hash(root/PROTOCOL) if coordinate_protocol else None,
            "classifier_selection_sha256": sha256_file(root / "results/experiments/final_selection.json"),
            "checkpoint": checkpoint.relative_to(root).as_posix(), "checkpoint_sha256": selection["checkpoint_sha256"],
            "split_hashes": selection["split_hashes"], "cache_fingerprint": selection["cache_fingerprint"]}


def lock_localisation(root: Path, validation_run: str) -> dict:
    """Freeze a completed validation protocol; do not read test masks/manifests."""
    directory = local_path(root, validation_run, ("results", "explainability", "validation"))
    result = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    identity = protocol_identity(root)
    if result.get("status") != "COMPLETE" or result.get("partition") != "validation" or result.get("identity") != identity:
        raise ValueError("Need a completed validation run with the current frozen protocol")
    for name, expected in result["artifact_sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError("Validation localisation artifact changed")
    lock = {"status": "LOCKED", "identity": identity, "validation_run": validation_run,
            "validation_summary_sha256": sha256_file(directory / "summary.json")}
    write_json(root / "results/explainability/localisation_lock.json", lock, exclusive=True)
    return lock


def require_final_lock(root: Path) -> dict:
    lock_path = root / "results/explainability/localisation_lock.json"
    if not lock_path.exists():
        raise ValueError("Explicit final localisation requires --lock-final after validation")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED" or lock.get("identity") != protocol_identity(root):
        raise ValueError("Localisation implementation/protocol changed after lock")
    directory = local_path(root, lock["validation_run"], ("results", "explainability", "validation"))
    if sha256_file(directory / "summary.json") != lock["validation_summary_sha256"]:
        raise ValueError("Validation summary changed after lock")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    for name, expected in summary["artifact_sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError("Validation localisation artifact changed after lock")
    if (root / "results/explainability/final_localisation_attempt.json").exists():
        raise ValueError("Final localisation already attempted; no repeated test-driven tuning")
    return lock


def save_overlay(path: Path, image: np.ndarray, heatmap: np.ndarray, mask: np.ndarray) -> None:
    """Show same-image Grad-CAM and aligned union annotation; not a clinical display."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.explainability.overlay import overlay_heatmap
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    try:
        axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[1].imshow(overlay_heatmap(image, heatmap))
        for ax in axes:
            ax.contour(mask.astype(float), levels=[0.5], colors=["lime"], linewidths=0.8)
            ax.set_axis_off()
        axes[0].set_title("Full mammogram + lesion contour")
        axes[1].set_title("Malignant Grad-CAM + lesion contour")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)


def evaluate_localisation(root: Path, *, final: bool = False) -> dict:
    """Compute localisation only; never import/call classifier evaluation routines."""
    if final:
        lock = require_final_lock(root)  # Before test manifest or annotation access.
        identity = lock["identity"]
    else:
        if (root / "results/explainability/localisation_lock.json").exists():
            raise ValueError("Localisation choices are frozen; further development evaluation blocked")
        identity = protocol_identity(root)
    partition = "test" if final else "validation"
    audit, table, images = audit_roi(root, partition=partition, final_authorized=final)
    if audit["status"] not in ("READY", "READY_FOR_VALIDATION_LOCALISATION"):
        return audit  # No localisation output directories or fabricated metrics.
    if audit["split_hashes"] != identity["split_hashes"] or audit["cache_fingerprint"] != identity["cache_fingerprint"]:
        raise ValueError("Annotation image cohort differs from selected classifier cohort")
    config = identity["protocol"]
    if final:
        receipt = root / "results/explainability/final_localisation_attempt.json"
        write_json(receipt, {"status": "STARTED", "identity": identity}, exclusive=True)
    directory = root / "results/explainability" / partition / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    summary = {"status": "STARTED", "partition": partition, "identity": identity, "audit": audit}
    write_json(directory / "summary.json", summary)
    try:
        import tensorflow as tf
        from src.utils.config import load_config
        from src.explainability.gradcam import ResNet50GradCAM
        seed = load_config(root / "config/project.yaml")["random_seed"]
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        summary.update(seed=seed, tensorflow_version=tf.__version__, numpy_version=np.__version__)
        explainer = ResNet50GradCAM.from_checkpoint(root / identity["checkpoint"])
        mask_rows = defaultdict(list)
        for row in table:
            if row["role"] == "mask" and row['alignment_status'] == 'verified':
                mask_rows[row["canonical_image_id"]].append(row)
        metrics_rows = []
        for index, image_row in enumerate(sorted(images, key=lambda r: r["canonical_image_id"])):
            image_id = image_row["canonical_image_id"]
            cache = local_path(root, image_row["cache_relative_path"], ("data", "processed", "cache"))
            if sha256_file(cache) != image_row["cache_sha256"]:
                raise ValueError("Classifier cache changed")
            geometry = json.loads(image_row["preprocessing_metadata_json"])
            masks, lesions = [], mask_rows[image_id]
            for lesion in lesions:
                path = local_path(root, config["roi_root"] + "/" + lesion["path"], ("data", "raw", "CBIS-DDSM", "roi_cropped"))
                if sha256_file(path) != lesion["mask_sha256"]:
                    raise ValueError("Mask changed after alignment audit")
                masks.append(transform_mask(decode_mask(path, lesion["SOPInstanceUID"]).mask, geometry))
            image = load_canonical(cache, geometry["input_size"])
            # Only the canonical full mammogram reaches the classifier.
            result = explainer.explain(image)
            union = union_masks(masks)
            for lesion, mask in list(zip(lesions, masks)) + [(None, union)]:
                metrics_rows.append({"canonical_image_id": image_id, "participant_id": image_row["participant_id"],
                    "partition": partition, "scope": "lesion" if lesion else "image_union",
                    "lesion_id": lesion["record_id"] if lesion else "union",
                    "pathology": lesion["lesion_pathology"] if lesion else image_row["pathology"],
                    "mask_sha256": lesion["mask_sha256"] if lesion else "",
                    **localisation_metrics(result.heatmap, mask, config["heatmap_threshold"])})
            if index < config["overlay_limit"]:
                save_overlay(directory / f"{image_id}.png", image, result.heatmap, union)
        write_csv(directory / "localisation_metrics.csv", metrics_rows)
        write_csv(directory / "mapping_audit.csv", table, sorted({k for row in table for k in row}))
        aggregate = {}
        for scope in ("lesion", "image_union"):
            for pathology in ("MALIGNANT", "BENIGN", "BENIGN_WITHOUT_CALLBACK"):
                subset = [r for r in metrics_rows if r["scope"] == scope and r["pathology"] == pathology]
                if subset:
                    aggregate[f"{scope}/{pathology}"] = {"n": len(subset),
                        "pointing_game_rate": float(np.mean([r["pointing_game"] for r in subset])),
                        "lesion_hit_rate": float(np.mean([r["lesion_hit"] for r in subset])),
                        "mean_iou": float(np.mean([r["iou"] for r in subset])),
                        "mean_dice": float(np.mean([r["dice"] for r in subset])),
                        "zero_heatmaps": sum(r["zero_heatmap"] for r in subset)}
        summary.update(status="COMPLETE", aggregate=aggregate,
                       artifact_sha256={name: sha256_file(directory / name) for name in ("localisation_metrics.csv", "mapping_audit.csv")})
        if sha256_file(root / identity["checkpoint"]) != identity["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed during localisation")
        write_json(directory / "summary.json", summary)
        if final:
            write_json(receipt, {"status": "COMPLETE", "run": directory.relative_to(root).as_posix(),
                                 "summary_sha256": sha256_file(directory / "summary.json")})
        return {"status": "COMPLETE", "run": directory.relative_to(root).as_posix()}
    except (ValueError, OSError, RuntimeError) as exc:
        summary.update(status="FAILED", reason=str(exc))
        write_json(directory / "summary.json", summary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--audit", "--dry-run", dest="audit", action="store_true")
    action.add_argument("--evaluate-validation", action="store_true")
    action.add_argument("--lock-final", metavar="VALIDATION_RUN")
    action.add_argument("--evaluate-final", action="store_true")
    parser.add_argument("--partition", choices=("train", "validation"), default="validation", help="Audit scope only; never implicitly test")
    args = parser.parse_args()
    try:
        if args.audit:
            result, table, _ = audit_roi(PROJECT_ROOT, partition=args.partition)
            write_audit(PROJECT_ROOT, result, table)
        elif args.lock_final:
            result = lock_localisation(PROJECT_ROOT, args.lock_final)
        else:
            result = evaluate_localisation(PROJECT_ROOT, final=args.evaluate_final)
        print(json.dumps(result, indent=2))
        if result["status"] == "NOT_READY":
            parser.exit(2)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Localisation refused: {exc}\n")


if __name__ == "__main__":
    main()
