"""One validation-image Grad-CAM check; never train or access the test cohort."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pydicom
import tensorflow as tf

from src.data.provenance import sha256_file
from src.explainability.gradcam import ResNet50GradCAM, TARGET_LAYER
from src.preprocessing.manifests import local_path, read_rows, verify_full_reference
from src.preprocessing.model_inputs import transfer_input
from src.training.data import load_canonical
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT


def validation_smoke(run: str, root: Path = PROJECT_ROOT) -> dict:
    """Choose the smallest validation image ID, independently of labels/scores.

    Write a new evidence file exclusively, protecting previous evidence. No
    experiment file is opened for writing. Only one cache array is loaded.
    """
    destination = root / "reports/evidence/gradcam_smoke.json"
    if destination.exists():
        raise ValueError("Grad-CAM smoke evidence already exists; refusing to overwrite")
    evidence = {"status": "FAIL", "run": run, "partition": "validation", "target_layer": TARGET_LAYER,
                "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        directory = local_path(root, run, ("results", "experiments"))
        checkpoint = directory / "best_model.keras"
        threshold = json.loads((directory / "threshold.json").read_text(encoding="utf-8"))
        lock = json.loads((root / "results/experiments/final_selection.json").read_text(encoding="utf-8"))
        config = load_config(directory / "config_used.yaml")
        manifest = root / "data/processed/manifests/validation_processed.csv"
        checkpoint_hash, manifest_hash = sha256_file(checkpoint), sha256_file(manifest)
        if (config["experiment"]["model"] != "resnet50" or config["project"]["image_type"] != "full_mammogram"
                or config["project"]["positive_class"] != "MALIGNANT" or config["project"]["subset"] != "mass"
                or config["project"]["dataset"] != "CBIS-DDSM" or config["project"]["input_size"] != 224):
            raise ValueError("Selected experiment is outside the validated mammogram scope")
        if (threshold.get("partition") != "validation" or checkpoint_hash != threshold.get("checkpoint_sha256")
                or manifest_hash != threshold.get("validation_manifest_sha256") or lock.get("run") != run
                or lock.get("checkpoint_sha256") != checkpoint_hash or lock.get("threshold") != threshold["threshold"]):
            raise ValueError("Checkpoint, validation manifest or locked selection hash mismatch")
        rows = read_rows(manifest)
        if any(r.get("final_partition") != "validation" or r.get("official_partition") != "train" for r in rows):
            raise ValueError("Smoke input must consist of official-training validation records only")
        row = min(rows, key=lambda r: r["canonical_image_id"])
        if row.get("processing_status") != "success" or row.get("mapping_status") != "uniquely_mapped":
            raise ValueError("Validation record is not successfully mapped and processed")
        metadata = json.loads(row["preprocessing_metadata_json"])
        header = pydicom.dataset.Dataset()
        for tag in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "Modality"):
            setattr(header, tag, metadata.get(tag, ""))
        verify_full_reference(row, header)
        if metadata.get("clahe") is not False or metadata.get("input_size") != 224:
            raise ValueError("Expected the canonical non-CLAHE cache")
        cache = local_path(root, row["cache_relative_path"], ("data", "processed", "cache"))
        cache_hash = sha256_file(cache)
        if cache_hash != row["cache_sha256"]:
            raise ValueError("Canonical cache content hash mismatch")
        image = load_canonical(cache, config["project"]["input_size"])
        tf.keras.utils.set_random_seed(config["project"]["random_seed"])
        tf.config.experimental.enable_op_determinism()
        explainer = ResNet50GradCAM.from_checkpoint(checkpoint, input_size=config["project"]["input_size"])
        ordinary = float(explainer.model(transfer_input(image, "resnet50")[None, ...], training=False).numpy()[0, 0])
        result = explainer.explain(image)
        difference = abs(ordinary - result.malignant_probability)
        finite = bool(np.isfinite(result.heatmap).all() and np.isfinite(ordinary))
        unchanged = checkpoint_hash == sha256_file(checkpoint)
        passed = (difference <= 1e-6 and finite and result.heatmap.shape == (224, 224)
                  and result.feature_map_shape == (7, 7, 2048) and unchanged
                  and float(result.heatmap.min()) >= 0 and float(result.heatmap.max()) <= 1)
        evidence.update(checkpoint=checkpoint.relative_to(root).as_posix(), checkpoint_sha256=checkpoint_hash,
            checkpoint_unchanged=unchanged, validation_manifest_sha256=manifest_hash,
            canonical_image_id=row["canonical_image_id"], cache_relative_path=row["cache_relative_path"], cache_sha256=cache_hash,
            selection_rule="lexicographically_smallest_validation_canonical_image_id",
            locked_threshold=threshold["threshold"], threshold_used_for_gradcam=False,
            ordinary_probability=ordinary, gradcam_probability=result.malignant_probability, absolute_difference=difference,
            probability_absolute_tolerance=1e-6, heatmap_shape=list(result.heatmap.shape),
            feature_map_shape=list(result.feature_map_shape), heatmap_min=float(result.heatmap.min()),
            heatmap_max=float(result.heatmap.max()), finite=finite, gradient_l1=result.gradient_l1,
            raw_positive_max=result.raw_positive_max, zero_heatmap=result.zero_heatmap,
            gradient_target="sigmoid_probability_of_MALIGNANT", status="PASS" if passed else "FAIL",
            tensorflow_version=tf.__version__, keras_version=tf.keras.__version__, numpy_version=np.__version__,
            implementation_sha256=sha256_file(Path(__file__).with_name("gradcam.py")),
            limitations="Single-image execution check only; no ROI/localisation or clinical performance evaluation")
    except (ValueError, OSError, KeyError, RuntimeError, tf.errors.OpError) as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if evidence["status"] != "PASS":
        raise ValueError(f"Grad-CAM smoke failed; see {destination}")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Repository-relative selected run directory")
    args = parser.parse_args()
    try:
        print(json.dumps(validation_smoke(args.run), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Grad-CAM smoke refused/failed: {exc}\n")


if __name__ == "__main__":
    main()
