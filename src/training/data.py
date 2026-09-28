"""Processed-cache loading with partition-specific provenance and no raw image access."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom

from src.data.provenance import sha256_file
from src.preprocessing.manifests import local_path, read_rows, verify_full_reference
from src.preprocessing.model_inputs import canonical_array, custom_cnn_input, transfer_input
from src.preprocessing.settings import preprocessing_settings
from src.training.config import TRANSFER_MODELS


@dataclass
class ProcessedSplit:
    partition: str
    rows: list[dict]
    images: np.ndarray
    labels: np.ndarray
    manifest_sha256: str
    cache_fingerprint: str
    split_hashes: dict


def load_canonical(path: Path, size: int) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.shape != (size, size) or array.dtype != np.float32:
        raise ValueError("Cache must be float32 with configured square shape")
    return canonical_array(array)


def _load_processed(root: Path, partition: str, project: dict) -> ProcessedSplit:
    """Read only this partition; do not call the all-partition DICOM audit here."""
    settings = preprocessing_settings(project)
    path = root / f"data/processed/manifests/{partition}_processed.csv"
    rows = read_rows(path)
    evidence_path = root / f"reports/evidence/preprocessing_{partition}.json"
    if not evidence_path.is_file():
        raise ValueError("Missing completed preprocessing evidence")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (evidence.get("status") != "COMPLETE" or evidence.get("failed") != 0 or
            evidence.get("succeeded") != len(rows) or evidence.get("settings") != settings):
        raise ValueError("Preprocessing evidence is incomplete or settings differ")
    original = read_rows(root / f"data/splits/{partition}.csv")
    if evidence.get("split_sha256", {}).get(partition) != sha256_file(root / f"data/splits/{partition}.csv"):
        raise ValueError("Final split changed after preprocessing")
    indexed = {r["canonical_image_id"]: r for r in original}
    if len(indexed) != len(original) or len(rows) != len(original):
        raise ValueError("Incomplete or duplicate processed cohort")
    attempts = read_rows(root / f"reports/evidence/preprocessing_{partition}_attempt.csv")
    if attempts != rows:
        raise ValueError("Processed manifest differs from completed attempt log")
    images, seen, paths = [], set(), set()
    for row in rows:
        key = row.get("canonical_image_id")
        if key in seen or key not in indexed or any(row.get(k) != v for k, v in indexed[key].items()):
            raise ValueError("Processed row differs from its unique validated split record")
        seen.add(key)
        if (row.get("binary_label") not in ("0", "1") or row.get("final_partition") != partition or
                row.get("official_partition") != ("test" if partition == "test" else "train") or
                row.get("processing_status") != "success" or row.get("mapping_status") != "uniquely_mapped" or
                row.get("cache_dtype") != "float32" or row.get("preprocessing_version") != settings["version"]):
            raise ValueError("Invalid label, partition, mapping or processing status")
        metadata = json.loads(row["preprocessing_metadata_json"])
        header = pydicom.dataset.Dataset()
        for tag in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "Modality"):
            setattr(header, tag, metadata.get(tag, ""))
        verify_full_reference(row, header)
        if metadata.get("clahe") is not False or metadata.get("input_size") != settings["input_size"]:
            raise ValueError("Noncanonical preprocessing metadata")
        local_path(root, row["dicom_relative_path"], ("data", "raw", "CBIS-DDSM", "images"))
        cache = local_path(root, row["cache_relative_path"], tuple(Path(settings["cache_directory"]).parts))
        if evidence.get("fingerprint") not in cache.parts or cache.resolve() in paths:
            raise ValueError("Wrong cache namespace or duplicate cache path")
        paths.add(cache.resolve())
        if sha256_file(cache) != row["cache_sha256"]:
            raise ValueError("Cache content hash mismatch")
        images.append(load_canonical(cache, settings["input_size"]))
    return ProcessedSplit(partition, rows, np.stack(images), np.array([int(r["binary_label"]) for r in rows]),
                          sha256_file(path), evidence["fingerprint"], evidence["split_sha256"])


def load_selection_data(root: Path, project: dict) -> tuple[ProcessedSplit, ProcessedSplit]:
    """Normal training API has no test argument and never opens a test artifact."""
    train = _load_processed(root, "train", project)
    validation = _load_processed(root, "validation", project)
    if train.cache_fingerprint != validation.cache_fingerprint or train.split_hashes != validation.split_hashes:
        raise ValueError("Train/validation must use the same canonical cache and split definition")
    for key in ("participant_id", "canonical_image_id", "dicom_relative_path", "cache_relative_path"):
        if {r[key] for r in train.rows} & {r[key] for r in validation.rows}:
            raise ValueError(f"Train/validation leakage: {key}")
    return train, validation


def load_final_test(root: Path, project: dict, lock: dict) -> ProcessedSplit:
    """Explicit final-only entry point; require locked run provenance."""
    if lock.get("status") != "LOCKED_FOR_FINAL_TEST" or not lock.get("checkpoint_sha256"):
        raise ValueError("Final test loading requires a locked model and threshold")
    data = _load_processed(root, "test", project)
    if data.split_hashes != lock["split_hashes"] or data.cache_fingerprint != lock["cache_fingerprint"]:
        raise ValueError("Test cache/splits differ from the locked experiment")
    if {r["participant_id"] for r in data.rows} & set(lock["selection_participants"]):
        raise ValueError("Test participant leakage")
    return data


def make_dataset(data: ProcessedSplit, model: str, batch_size: int, seed: int, *, shuffle: bool = False):
    """Deterministic tf.data stream; validation/test always preserve manifest order."""
    import tensorflow as tf
    if shuffle and data.partition != "train":
        raise ValueError("Only training data may be shuffled")
    channels = 3 if model in TRANSFER_MODELS else 1
    size = data.images.shape[1]
    def generate():
        for image, label in zip(data.images, data.labels):
            adapted = transfer_input(image, model) if channels == 3 else custom_cnn_input(image)
            yield adapted, np.array([label], dtype=np.float32)
    dataset = tf.data.Dataset.from_generator(generate, output_signature=(
        tf.TensorSpec((size, size, channels), tf.float32), tf.TensorSpec((1,), tf.float32)))
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(data.labels)))
    if shuffle:
        dataset = dataset.shuffle(len(data.labels), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    return dataset.batch(batch_size).with_options(options).prefetch(1)


def training_class_weights(labels: np.ndarray, *, partition: str) -> dict[int, float]:
    from src.models.majority import binary_labels
    if partition != "train":
        raise ValueError("Class weights require training labels only")
    y = binary_labels(labels)
    counts = np.bincount(y, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Class weights require both training classes")
    return {i: float(len(y) / (2 * counts[i])) for i in (0, 1)}
