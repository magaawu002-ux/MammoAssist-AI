"""Write observed predictions and plain report figures, never placeholder metrics."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data.cbis_ddsm_metadata import write_csv
from src.evaluation.calibration import calibration_curve
from src.evaluation.errors import prediction_records, error_summary
from src.evaluation.metrics import evaluate_metrics


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_evaluation(directory: Path, rows: list[dict], probabilities: np.ndarray,
                    threshold: float, *, partition: str) -> dict:
    """Save metrics and curves, clearly mark undefined one-class curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = np.array([int(r["binary_label"]) for r in rows])
    metrics = dict(evaluate_metrics(y, probabilities, threshold), partition=partition)
    predictions = prediction_records(rows, probabilities, threshold)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "metrics.json", metrics)
    write_csv(directory / "predictions.csv", predictions)
    write_json(directory / "false_negative_summary.json", error_summary(predictions))
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, str(count), ha="center", va="center", color="black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"],
           xlabel="Predicted class", ylabel="True class", title=f"Confusion matrix ({partition})")
    fig.tight_layout(); fig.savefig(directory / "confusion_matrix.png", dpi=200); plt.close(fig)
    for name in ("roc_curve", "precision_recall_curve"):
        fig, ax = plt.subplots(figsize=(5, 4))
        if len(np.unique(y)) == 2:
            if name == "roc_curve":
                x, value, _ = roc_curve(y, probabilities)
                ax.plot(x, value)
                ax.plot([0, 1], [0, 1], "--", color="gray")
            else:
                value, x, _ = precision_recall_curve(y, probabilities)
                ax.step(x, value, where="post")
        else:
            ax.text(0.5, 0.5, "Undefined: one-class labels", ha="center")
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False-positive rate" if name == "roc_curve" else "Recall (malignant)",
               ylabel="Sensitivity" if name == "roc_curve" else "Precision", title=f"{name.replace('_', ' ').title()} ({partition})")
        fig.tight_layout(); fig.savefig(directory / f"{name}.png", dpi=200); plt.close(fig)
    calibration = calibration_curve(y, probabilities)
    write_json(directory / "calibration.json", calibration)
    points = [b for b in calibration["bins"] if b["count"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([b["mean_probability"] for b in points], [b["positive_fraction"] for b in points], "o-")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted malignancy probability", ylabel="Observed malignant fraction",
           title=f"Reliability diagram ({partition})")
    fig.tight_layout(); fig.savefig(directory / "calibration_curve.png", dpi=200); plt.close(fig)
    return metrics


def save_history(directory: Path, history: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not history:
        return  # A non-trained majority reference has no training history.
    write_csv(directory / "training_history.csv", history)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("loss", "val_loss"):
        ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
    ax.set(xlabel="Epoch (stages concatenated)", ylabel="Binary cross entropy plus regularisation loss",
           title="Training and validation loss")
    ax.legend(); fig.tight_layout(); fig.savefig(directory / "training_history.png", dpi=200); plt.close(fig)
