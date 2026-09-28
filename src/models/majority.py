"""A hard majority-class reference fitted using training labels only."""

import numpy as np


def binary_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0 or not np.isin(values, [0, 1]).all():
        raise ValueError("Expected nonempty one-dimensional binary labels (0/1)")
    return values.astype(np.int32)


class MajorityReference:
    """Ties choose benign (0); output 0/1 scores, not calibrated risk estimates."""

    def __init__(self) -> None:
        self.majority_class: int | None = None

    def fit(self, labels: np.ndarray, *, partition: str) -> "MajorityReference":
        if partition != "train":
            raise ValueError("Majority reference can be fitted on training labels only")
        labels = binary_labels(labels)
        self.majority_class = int(np.count_nonzero(labels) > len(labels) / 2)
        return self

    def predict_probability(self, count: int) -> np.ndarray:
        if self.majority_class is None or count < 1:
            raise ValueError("Reference must be fitted and prediction count positive")
        return np.full(count, self.majority_class, dtype=np.float32)
