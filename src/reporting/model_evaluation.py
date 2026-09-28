"""Generate a report package from saved artifacts only: python -m src.reporting.model_evaluation."""

import argparse
import csv
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.provenance import sha256_file
from src.reporting.artifacts import MODELS, load_evidence
from src.reporting.summaries import calibration_row, confidence_rows, false_negative_tables, history_values, metric_row
from src.utils.paths import PROJECT_ROOT

LABELS = dict(zip(MODELS, ("Majority reference", "MammoCNN-Baseline", "MammoCNN-Regularised", "VGG16", "ResNet50", "EfficientNetB0")))


def table(destination: Path, name: str, rows: list[dict], note: str = "") -> None:
    """CSV retains numerical precision; Markdown rounds display only, null stays NA."""
    fields = list(rows[0])
    with (destination/f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    def display(key: str, value: object) -> str:
        if value is None:
            return "NA"
        if key == "model":
            return LABELS.get(value, str(value))
        if key == "threshold":
            return str(value)
        return f"{value:.6f}" if isinstance(value, float) else str(value).replace("|", "\\|")
    shown = [k for k in fields if k != "run"]
    labels = {"accuracy": "Acc.", "sensitivity": "Sens.", "specificity": "Spec.", "precision": "Prec.",
              "balanced_accuracy": "Bal. acc.", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier_score": "Brier",
              "false_negatives": "FN", "false_negative_rate": "FNR"}
    text = "| " + " | ".join(labels.get(k, k.replace("_", " ")) for k in shown) + " |\n| " + " | ".join(["---"]*len(shown)) + " |\n"
    text += "".join("| " + " | ".join(display(k, r[k]) for k in shown) + " |\n" for r in rows)
    (destination/f"{name}.md").write_text(text + ("\n"+note+"\n" if note else ""), encoding="utf-8")


def figures(destination: Path, comparison: list[dict], histories: dict) -> None:
    keys = ("sensitivity", "specificity", "balanced_accuracy", "roc_auc", "pr_auc")
    fig, axes = plt.subplots(1, len(keys), figsize=(15, 5), sharey=True)
    try:
        for ax, key in zip(axes, keys):
            values = [r[key] for r in comparison]
            ax.barh(range(len(comparison)), values, color="#346b8c")
            ax.set_xlim(0, 1)
            ax.set_xticks([0, .5, 1])
            ax.set_title(key.replace("_", " ").upper() if "auc" in key else key.replace("_", " ").capitalize(), fontsize=10)
            ax.set_xlabel("Score")
            for i, value in enumerate(values):
                ax.text(max(.02, value-.025), i, f"{value:.3f}", ha="right" if value > .15 else "left",
                        va="center", color="white" if value > .15 else "black", fontsize=9)
        axes[0].set_yticks(range(len(comparison)), [LABELS[r["model"]] for r in comparison])
        axes[0].invert_yaxis()
        fig.suptitle("Validation performance at saved thresholds (PR-AUC = average precision)")
        fig.tight_layout()
        fig.savefig(destination/"validation_comparison.png", dpi=250)
    finally:
        plt.close(fig)
    for model, rows in histories.items():
        epochs, losses, validation, boundary = history_values(rows)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        try:
            ax.plot(epochs, losses, label="Training loss", color="#346b8c")
            ax.plot(epochs, validation, label="Validation loss", color="#b34b2f")
            if boundary is not None:
                ax.axvline(boundary+.5, color="gray", linestyle="--", linewidth=1)
                ax.text((1+boundary)/2, .97, "Frozen base", transform=ax.get_xaxis_transform(), ha="center", va="top")
                ax.text((boundary+1+len(rows))/2, .97, "Fine-tuning", transform=ax.get_xaxis_transform(), ha="center", va="top")
            ax.set(xlabel="Epoch (cumulative)", ylabel="Saved training objective (loss)", title=f"{LABELS[model]} training history", ylim=(0, max(losses+validation)*1.16))
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(destination/f"{model}_training_history.png", dpi=250)
        finally:
            plt.close(fig)


def generate(root: Path = PROJECT_ROOT) -> Path:
    evidence = load_evidence(root)
    runs, test, lock = evidence["runs"], evidence["test"], evidence["lock"]
    comparison = [metric_row(m, runs[m]) for m in MODELS]
    selected_keys = ("sensitivity", "specificity", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score",
                     "false_negatives", "false_negative_rate", "threshold")
    selected = [dict(partition=partition, **{k: row[k] for k in selected_keys}) for partition, row in
                (("VALIDATION", metric_row("resnet50", runs["resnet50"])),
                 ("OFFICIAL HOLDOUT TEST", metric_row("resnet50", dict(metrics=test, run=lock["run"]))))]
    ci = confidence_rows(test, evidence["bootstrap"])
    fn_summary, subgroups, proximity = false_negative_tables(evidence["predictions"], lock["threshold"])
    calibration = [calibration_row("VALIDATION", runs["resnet50"]["calibration"], runs["resnet50"]["metrics"]),
                   calibration_row("OFFICIAL HOLDOUT TEST", evidence["calibration"], test)]
    for rows in evidence["histories"].values():
        history_values(rows)
    sources = evidence["sources"]
    sources.unchanged()
    destination = root/"reports/evidence/model_evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    table(destination, "validation_comparison", comparison,
          "Validation only; no new selection. PR-AUC = average precision. NA/blank CSV precision is undefined (no positive predictions). CSV retains full precision.")
    table(destination, "resnet50_validation_vs_test", selected, "Threshold fixed from validation; test outcomes do not change model or threshold selection.")
    table(destination, "test_confidence_intervals", ci,
          f"95% participant-cluster percentile bootstrap; {evidence['bootstrap']['iterations']} iterations; seed {evidence['bootstrap']['seed']}; threshold fixed from validation at {lock['threshold']}. Existing intervals read, not resampled.")
    table(destination, "test_confusion_matrix", [{"partition": "OFFICIAL HOLDOUT TEST", **{k: test[k] for k in ("n", "tn", "fp", "fn", "tp")}}],
          "MALIGNANT=1. TN: benign correctly classified; FP: benign classified malignant; FN: malignant classified benign; TP: malignant correctly classified.")
    table(destination, "test_false_negative_summary", [fn_summary])
    table(destination, "test_false_negative_subgroups", subgroups, "FNR = false-negative images / malignant images in subgroup. Descriptive only; no subgroup significance tests.")
    table(destination, "test_threshold_proximity", proximity,
          "Cumulative inclusive bands: abs(saved probability - locked threshold) <= distance. Bands overlap and must not be summed. FN scope includes only malignant misses below threshold.")
    table(destination, "calibration_summary", calibration)
    delta = calibration[1]["brier_score"]-calibration[0]["brier_score"]
    text = (f"Validation Brier score: {calibration[0]['brier_score']}; official holdout test Brier score: {calibration[1]['brier_score']}. "
            f"Test minus validation: {delta:+.12f} (lower Brier is better).\n\n"
            f"Mean predicted probability minus observed malignant fraction, reconstructed from saved calibration bins: "
            f"validation {calibration[0]['mean_probability_minus_observed']:+.6f}; test {calibration[1]['mean_probability_minus_observed']:+.6f}. "
            "Positive differences indicate average overprediction; negative differences indicate average underprediction. "
            "These descriptive shifts do not establish clinical calibration or their cause. Brier score reflects both calibration "
            "and discrimination and depends on case mix/prevalence. No recalibration or threshold adjustment was performed.\n")
    (destination/"calibration_interpretation.md").write_text(text, encoding="utf-8")
    counts = [dict(partition=p, images=len(rows), participants=len({r["participant_id"] for r in rows})) for p, rows in evidence["splits"].items()]
    table(destination, "cohort_counts", counts)
    figures(destination, comparison, evidence["histories"])
    methodology = ("# Model evaluation evidence\n\n"
        "This package aggregates saved real experiment artifacts. It performs no inference, training, evaluation rerun, bootstrap resampling or threshold selection.\n\n"
        + "; ".join(f"{r['partition']}: {r['images']} images / {r['participants']} participants" for r in counts) + ". "
        "Split hashes match the selection lock; pairwise participant and image intersections are empty. "
        "All six runs share the same split definitions, processed selection manifest hashes and canonical cache fingerprint. "
        "Classification uses full CBIS-DDSM mass mammograms; MALIGNANT=1. No ROI data/results were read.\n\n"
        f"Selected run: `{lock['run']}`. Fixed validation-selected threshold: **{lock['threshold']}**. "
        "The saved model-selection lock and completed one-time test receipt were verified. Validation metrics are bound through "
        "the locked run-metadata hash; test metrics match the receipt hash. These records document validation-only selection followed "
        "by the one-time official holdout evaluation, but cannot prove the absence of unrecorded activity. "
        "No test-driven retuning or new model choice is performed here.\n\n"
        "Existing 95% participant-cluster percentile bootstrap intervals use 2000 iterations, seed 42 and the fixed validation threshold. "
        "No new confidence intervals or error bars are inferred for validation comparisons. "
        "PR-AUC denotes average precision. Majority-reference precision is undefined, not zero.\n\n"
        "Training curves use saved losses and recorded head/fine_tuning stages. Loss includes whatever objective was recorded during training; "
        "curves are not reconstructed from checkpoints. Comparison ordering follows the planned experimental progression, not test ranking.\n\n"
        "Test false-negative subgroups are descriptive image-level rates with malignant-image denominators; "
        "correlated views/participants and small subgroups limit interpretation. Threshold-proximity bands are cumulative. "
        "Validation estimates were used for development and can be optimistic. This digitised-film research prototype is not clinically validated.\n\n"
        "Reproduce: `python -m src.reporting.model_evaluation`. Outputs are restricted to this reporting directory; "
        "source classifier artifacts remain unchanged. `provenance.json` records source and implementation SHA256 hashes. "
        "Checkpoint hashes are checked for consistency across metadata/locks; model weights and image arrays are never opened by this report.\n")
    (destination/"methodology.md").write_text(methodology, encoding="utf-8")
    sources.unchanged()
    provenance = {"status": "PASS", "source_sha256": sources.hashes,
                  "implementation_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                  "environment": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
                  "outputs_sha256": {p.name: sha256_file(p) for p in sorted(destination.iterdir()) if p.is_file() and p.name != "provenance.json"}}
    (destination/"provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        print(f"PASS: saved-artifact reporting package: {generate()}")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Reporting refused: {exc}\n")


if __name__ == "__main__":
    main()
