"""Synthetic saved artifacts; no model, image arrays, network or inference needed."""

import csv
import json
from pathlib import Path

import pytest

from src.data.provenance import sha256_file
from src.reporting.artifacts import MODELS, Sources, load_evidence
from src.reporting.model_evaluation import generate, table
from src.reporting.summaries import false_negative_tables, history_values


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def artifacts(tmp_path):
    split_hashes, manifests, splits = {}, {}, {}
    for part in ("train", "validation", "test"):
        rows = [dict(canonical_image_id=f"{part}_{i}", participant_id=f"{part}_{i//2}", binary_label=str(y),
                     laterality="LEFT" if i % 2 else "RIGHT", view="CC" if i % 2 else "MLO",
                     final_partition=part, official_partition="test" if part == "test" else "train")
                for i, y in enumerate((0, 0, 1, 1))]
        splits[part] = rows
        for path in (f"data/splits/{part}.csv", f"data/processed/manifests/{part}_processed.csv"):
            write_csv(tmp_path/path, rows)
        split_hashes[part] = sha256_file(tmp_path/f"data/splits/{part}.csv")
        manifests[part] = sha256_file(tmp_path/f"data/processed/manifests/{part}_processed.csv")
    metrics = dict(threshold=.5, n=4, tn=1, fp=1, fn=1, tp=1, accuracy=.5, sensitivity=.5, specificity=.5,
                   precision=.5, f1=.5, balanced_accuracy=.5, roc_auc=.5, pr_auc=.5, pr_auc_definition="average_precision", brier_score=.3125)
    calibration = dict(brier_score=.3125, bins=[dict(lower=0, upper=.5, count=2, mean_probability=.25, positive_fraction=.5),
                                              dict(lower=.5, upper=1, count=2, mean_probability=.75, positive_fraction=.5)])
    def evaluation(directory, part):
        write_json(directory/part/"metrics.json", dict(metrics, partition=part))
        write_json(directory/part/"calibration.json", calibration)
        predictions = [dict(r, true_label=r["binary_label"], predicted_probability=p, threshold=.5, predicted_class=c,
                            correct=str(int(r["binary_label"]) == c), error_type=e)
                       for r, p, c, e in zip(splits[part], (.25, .75, .25, .75), (0, 1, 0, 1),
                                            ("true_negative", "false_positive", "false_negative", "true_positive"))]
        write_csv(directory/part/"predictions.csv", predictions)
    participants = sorted({r["participant_id"] for part in ("train", "validation") for r in splits[part]})
    for model in MODELS:
        directory = tmp_path/f"results/experiments/{model}/run"
        evaluation(directory, "validation")
        threshold = dict(threshold=.5, partition="validation", checkpoint_sha256="recorded-hash",
                         validation_manifest_sha256=manifests["validation"])
        write_json(directory/"threshold.json", threshold)
        (directory/"config_used.yaml").write_text("synthetic: true\n")
        write_csv(directory/"training_history.csv", [dict(epoch=1, stage="head", loss=.8, val_loss=.9),
                                                     dict(epoch=2, stage="fine_tuning", loss=.6, val_loss=.7)])
        meta = dict(model=model, status="COMPLETE", selection_only=True, split_hashes=split_hashes,
                    cache_fingerprint="synthetic", processed_manifest_sha256={k: manifests[k] for k in ("train", "validation")},
                    selection_participants=participants, checkpoint_sha256="recorded-hash")
        for path, key in (("threshold.json", "threshold_sha256"), ("config_used.yaml", "config_sha256"),
                          ("validation/metrics.json", "validation_metrics_sha256"), ("validation/predictions.csv", "validation_predictions_sha256")):
            meta[key] = sha256_file(directory/path)
        write_json(directory/"run_metadata.json", meta)
    selected = "results/experiments/resnet50/run"
    directory = tmp_path/selected
    meta = json.loads((directory/"run_metadata.json").read_text())
    lock = {k: meta[k] for k in ("checkpoint_sha256", "threshold_sha256", "config_sha256", "split_hashes", "cache_fingerprint", "selection_participants")}
    lock.update(status="LOCKED_FOR_FINAL_TEST", run=selected, threshold=.5, run_metadata_sha256=sha256_file(directory/"run_metadata.json"))
    write_json(tmp_path/"results/experiments/final_selection.json", lock)
    evaluation(directory, "test")
    write_json(tmp_path/"results/experiments/final_test_attempt.json", dict(status="COMPLETE", run=selected,
        metrics_sha256=sha256_file(directory/"test/metrics.json"), test_manifest_sha256=manifests["test"]))
    write_json(directory/"test/bootstrap_intervals.json", dict(method="participant_cluster_percentile", seed=42,
        iterations=2000, confidence=.95, threshold_fixed=.5, intervals={k: dict(lower=.1, upper=.9, valid_replicates=2000, invalid_replicates=0)
        for k in ("sensitivity", "specificity", "balanced_accuracy", "roc_auc")}))
    return tmp_path


def test_generate_package_preserves_sources_and_has_no_model_dependency(artifacts):
    before = {p: sha256_file(p) for p in artifacts.rglob("*") if p.is_file()}
    destination = generate(artifacts)
    assert all(sha256_file(p) == digest for p, digest in before.items())
    assert len(list(destination.glob("*.png"))) == 4
    assert json.loads((destination/"provenance.json").read_text())["status"] == "PASS"
    assert "participant-cluster percentile" in (destination/"test_confidence_intervals.md").read_text()
    assert "OFFICIAL HOLDOUT TEST" in (destination/"resnet50_validation_vs_test.md").read_text()
    assert not list(artifacts.rglob("*.keras"))


@pytest.mark.parametrize("path", ["vgg16/run/validation/metrics.json", "resnet50/run/test/bootstrap_intervals.json"])
def test_missing_artifact_refusal(artifacts, path):
    (artifacts/"results/experiments"/path).unlink()
    with pytest.raises(ValueError, match="missing"):
        generate(artifacts)
    assert not (artifacts/"reports/evidence/model_evaluation").exists()


def test_extra_test_results_rejected(artifacts):
    write_json(artifacts/"results/experiments/vgg16/run/test/metrics.json", {})
    with pytest.raises(ValueError, match="outside the locked"):
        load_evidence(artifacts)


def test_incomplete_model_list(artifacts):
    (artifacts/"results/experiments/vgg16/run/run_metadata.json").unlink()
    with pytest.raises(ValueError, match="Incomplete model list"):
        load_evidence(artifacts)


def test_selected_metrics_hash_chain(artifacts):
    path = artifacts/"results/experiments/resnet50/run/validation/metrics.json"
    value = json.loads(path.read_text())
    value["roc_auc"] = .9
    write_json(path, value)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_evidence(artifacts)


@pytest.mark.parametrize("key,value,match", [("threshold", .6, "threshold differs"), ("fn", 3, "confusion matrix")])
def test_test_metric_consistency_even_with_updated_receipt(artifacts, key, value, match):
    path = artifacts/"results/experiments/resnet50/run/test/metrics.json"
    metrics = json.loads(path.read_text()); metrics[key] = value
    write_json(path, metrics)
    receipt = artifacts/"results/experiments/final_test_attempt.json"
    data = json.loads(receipt.read_text()); data["metrics_sha256"] = sha256_file(path)
    write_json(receipt, data)
    with pytest.raises(ValueError, match=match):
        load_evidence(artifacts)


def test_proximity_and_subgroup_denominators():
    rows = [dict(true_label=1, error_type="false_negative", participant_id="a", view="CC", laterality="LEFT", predicted_probability=.49),
            dict(true_label=1, error_type="false_negative", participant_id="a", view="MLO", laterality="RIGHT", predicted_probability=.48),
            dict(true_label=1, error_type="true_positive", participant_id="b", view="CC", laterality="LEFT", predicted_probability=.8),
            dict(true_label=0, error_type="false_positive", participant_id="c", view="MLO", laterality="LEFT", predicted_probability=.51)]
    summary, groups, proximity = false_negative_tables(rows, .5)
    assert summary == dict(malignant_images=3, false_negative_images=2, false_negative_participants=1, false_negative_rate=2/3)
    assert groups[0]["false_negative_rate"] == .5
    assert groups[1]["false_negative_rate"] == 1
    assert next(r for r in proximity if r["scope"] == "false_negative_images" and r["absolute_distance_at_most"] == .025)["count"] == 2


def test_zero_malignant_denominator():
    summary, groups, _ = false_negative_tables([dict(true_label=0, error_type="true_negative", participant_id="a", view="CC", laterality="LEFT", predicted_probability=.1)], .5)
    assert summary["false_negative_rate"] is None
    assert all(r["false_negative_rate"] is None for r in groups)


def test_null_precision_not_fabricated(tmp_path):
    table(tmp_path, "example", [dict(model="majority", precision=None)])
    assert "NA" in (tmp_path/"example.md").read_text()
    with (tmp_path/"example.csv").open() as handle:
        assert list(csv.DictReader(handle))[0]["precision"] == ""


def test_history_phases_and_unknown_phase():
    rows = [dict(epoch=1, loss=.8, val_loss=.9, stage="head"), dict(epoch=2, loss=.6, val_loss=.7, stage="fine_tuning")]
    assert history_values(rows)[3] == 1
    rows[1]["stage"] = "invented"
    with pytest.raises(ValueError, match="Unknown"):
        history_values(rows)


def test_source_mutation_detected(tmp_path):
    write_json(tmp_path/"example.json", dict(x=1))
    sources = Sources(tmp_path)
    sources.json("example.json")
    write_json(tmp_path/"example.json", dict(x=2))
    with pytest.raises(ValueError, match="changed during reporting"):
        sources.unchanged()


def test_bootstrap_protocol_not_silently_relabelled(artifacts):
    path = artifacts/"results/experiments/resnet50/run/test/bootstrap_intervals.json"
    value = json.loads(path.read_text()); value["method"] = "image_percentile"
    write_json(path, value)
    with pytest.raises(ValueError, match="bootstrap protocol"):
        load_evidence(artifacts)
