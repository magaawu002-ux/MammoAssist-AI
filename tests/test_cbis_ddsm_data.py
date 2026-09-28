"""Synthetic data tests; no downloaded images or external directories required."""

import csv
import json
from pathlib import Path

import pydicom
import pytest

from src.data.audit_cbis_ddsm import audit
from src.data.cbis_ddsm_metadata import (ALIASES, aggregate_images, normalize_participant,
    participant_labels, pathology_label, read_metadata)
from src.data.dicom_inventory import inventory_dicoms, map_images
from src.data.provenance import sha256_file
from src.data.splits import create_splits
from src.data.validate_splits import validate_splits
from src.utils.paths import PROJECT_ROOT


def source_row(participant="P_00001", pathology="BENIGN", abnormality="1", partition="train"):
    row = dict.fromkeys(ALIASES, "")
    prefix = "Training" if partition == "train" else "Test"
    row.update({"patient_id": participant, "pathology": pathology, "abnormality id": abnormality,
                "abnormality type": "mass", "left or right breast": "LEFT", "image view": "CC",
                "image file path": f"Mass-{prefix}_{participant}_LEFT_CC/1.2.3/1.2.4/000000.dcm"})
    row["unknown future field"] = "preserve me"
    return row


def metadata_file(root, rows, partition="train"):
    path = root / f"data/raw/CBIS-DDSM/metadata/mass_case_description_{partition}_set.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("value,expected", [("MALIGNANT", 1), ("BENIGN", 0),
    ("BENIGN_WITHOUT_CALLBACK", 0), ("", None), ("OTHER", None)])
def test_pathology(value, expected):
    assert pathology_label(value) == expected


@pytest.mark.parametrize("value,expected", [("P_00001", "P_00001"),
    ("Mass-Training_P_00001_LEFT_MLO_1", "P_00001"),
    ("Mass-Test_P_00001_RIGHT_CC", "P_00001"), ("patient1", ""),
    ("garbage_P_00001", ""), ("P_00001_extra", "")])
def test_participant_normalization(value, expected):
    normalized, method = normalize_participant(value)
    assert normalized == expected
    assert (method == "unparseable") == (not expected)


def test_lesions_and_unknown_columns_preserved(tmp_path):
    rows = [source_row(), source_row(), source_row(pathology="MALIGNANT", abnormality="2")]
    path = metadata_file(tmp_path, rows)
    lesions, issues, details = read_metadata(path, "train", tmp_path)
    images, problems = aggregate_images(lesions)
    assert not issues
    assert len(lesions) == 3 and len(images) == 1
    assert details["rows"] == 3
    assert images[0]["abnormality_count"] == 2
    assert images[0]["lesion_row_count"] == 3
    assert images[0]["binary_label"] == 1 and images[0]["mixed_pathology"]
    assert len(json.loads(images[0]["lesion_record_ids_json"])) == 3
    assert json.loads(lesions[0]["source_fields_json"])["unknown future field"] == "preserve me"
    assert problems[0]["reason"] == "repeated_abnormality_rows"


def test_unknown_labels_and_identity_flagged(tmp_path):
    rows = [source_row(pathology="UNKNOWN"), source_row(participant="bad", pathology="")]
    lesions, issues, _ = read_metadata(metadata_file(tmp_path, rows), "train", tmp_path)
    assert sum(p["reason"] == "unsupported_pathology" for p in issues) == 2
    assert any(p["reason"] == "unparseable_participant" for p in issues)
    assert aggregate_images(lesions)[0][0]["binary_label"] is None


def test_participant_malignancy():
    assert participant_labels([{"participant_id": "P_00001", "binary_label": v}
                               for v in (0, 1, 0)]) == {"P_00001": 1}


def mapped_images(root):
    rows = []
    for i in range(22):
        for view in ("CC", "MLO"):
            name = f"image_{i}_{view}.dcm"
            (root / name).write_bytes(b"synthetic existence fixture")
            rows.append(dict(participant_id=f"P_{i:05}", canonical_image_id=name,
                binary_label=i % 2, official_partition="train" if i < 20 else "test",
                dicom_relative_path=name, mapping_status="uniquely_mapped"))
    return rows


def test_patient_split_deterministic_and_official_holdout(tmp_path):
    images = mapped_images(tmp_path)
    splits, strategy = create_splits(images, 42)
    assert (splits, strategy) == create_splits(list(reversed(images)), 42)
    assert strategy == "stratified"
    assert len(splits["train"]) == 32 and len(splits["validation"]) == 8
    assert all(r["official_partition"] == "test" for r in splits["test"])
    assert all(r["official_partition"] == "train" for p in ("train", "validation") for r in splits[p])
    sets = [{r["participant_id"] for r in splits[p]} for p in splits]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert validate_splits(splits, tmp_path)["conclusion"] == "PASS"


def test_intentional_leakage_fails(tmp_path):
    images = mapped_images(tmp_path)
    images[-1]["participant_id"] = images[0]["participant_id"]
    with pytest.raises(ValueError, match="CRITICAL"):
        create_splits(images, 42)
    splits, _ = create_splits(mapped_images(tmp_path), 42)
    splits["test"][0]["participant_id"] = splits["train"][0]["participant_id"]
    evidence = tmp_path / "leakage.txt"
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(splits, tmp_path, evidence)
    assert json.loads(evidence.read_text())["conclusion"] == "FAIL"


@pytest.mark.parametrize("field,value", [("binary_label", 2), ("dicom_relative_path", "absent.dcm"),
    ("official_partition", "test"), ("mapping_status", "pending"), ("participant_id", "")])
def test_validator_rejects_invalid_records(tmp_path, field, value):
    splits, _ = create_splits(mapped_images(tmp_path), 42)
    splits["train"][0][field] = value
    with pytest.raises(ValueError):
        validate_splits(splits, tmp_path)


def test_duplicate_images_and_paths_detected(tmp_path):
    splits, _ = create_splits(mapped_images(tmp_path), 42)
    splits["validation"][0].update({key: splits["train"][0][key]
                                    for key in ("canonical_image_id", "dicom_relative_path")})
    with pytest.raises(ValueError, match="duplicate"):
        validate_splits(splits, tmp_path)


def test_incomplete_mapping_blocks_split(tmp_path):
    images = mapped_images(tmp_path)
    images[0]["mapping_status"] = "pending"
    with pytest.raises(ValueError, match="NOT FINAL"):
        create_splits(images, 42)


def test_sha256(tmp_path):
    file = tmp_path / "source"
    file.write_bytes(b"abc")
    assert sha256_file(file) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_unreadable_dicom(tmp_path):
    (tmp_path / "bad.dcm").write_bytes(b"invalid")
    rows = inventory_dicoms(tmp_path, tmp_path)
    assert rows[0]["read_status"] == "unreadable"
    assert "InvalidDicomError" in rows[0]["error"]


def test_header_inventory_and_mapping(tmp_path, monkeypatch):
    meta = pydicom.dataset.FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.DigitalMammographyXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID = "1.2.5"
    ds = pydicom.dataset.FileDataset(str(tmp_path / "image"), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID = "1.2.3", "1.2.4", "1.2.5"
    ds.Modality, ds.Rows, ds.Columns, ds.PatientID = "MG", 2, 2, "arbitrary-not-identity"
    ds.save_as(tmp_path / "image", enforce_file_format=True)
    original = pydicom.dcmread
    def checked(*args, **kwargs):
        assert kwargs["stop_before_pixels"] is True
        return original(*args, **kwargs)
    monkeypatch.setattr(pydicom, "dcmread", checked)
    inventory = inventory_dicoms(tmp_path, tmp_path)
    row = {"canonical_image_id": "image", "full_mammogram_reference": "case/1.2.3/1.2.4/000000.dcm"}
    mapped, issues = map_images([row], inventory)
    assert mapped[0]["mapping_status"] == "uniquely_mapped" and not issues
    ambiguous, _ = map_images([row], inventory + inventory)
    assert ambiguous[0]["mapping_status"] == "ambiguous" and not ambiguous[0]["dicom_relative_path"]
    unmapped, issues = map_images([dict(row, full_mammogram_reference="case/9.9/9.8/file.dcm")], inventory)
    assert unmapped[0]["mapping_status"] == "unmapped"
    assert any(p["reason"] == "orphan_DICOM" for p in issues)


def prepare_audit(root):
    (root / "config").mkdir()
    (root / "config/project.yaml").write_bytes((PROJECT_ROOT / "config/project.yaml").read_bytes())
    metadata_file(root, [source_row()])
    metadata_file(root, [source_row("P_00002", "MALIGNANT", partition="test")], "test")


def test_metadata_only_without_dicoms(tmp_path, monkeypatch):
    prepare_audit(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("Metadata-only must not inspect DICOM directories")
    monkeypatch.setattr("src.data.audit_cbis_ddsm.inventory_dicoms", forbidden)
    result = audit(tmp_path)
    assert result["lesion_rows"] == 2 and result["images"] == 2
    assert not result["participant_overlap"]
    assert result["split_status"].startswith("NOT FINAL")
    assert not (tmp_path / "data/splits/train.csv").exists()
    assert not (tmp_path / "data/interim/cbis_ddsm_dicom_inventory.csv").exists()
    assert (tmp_path / "reports/evidence/data_provenance.json").exists()
    assert (tmp_path / "results/figures/dataset_class_distribution.png").exists()


def test_full_missing_images_fails_with_evidence(tmp_path):
    prepare_audit(tmp_path)
    with pytest.raises(ValueError, match="Full audit failed"):
        audit(tmp_path, full=True)
    assert not (tmp_path / "data/splits/train.csv").exists()
    details = json.loads((tmp_path / "reports/evidence/data_audit_details.json").read_text())
    assert details["split_status"].startswith("NOT FINAL")


def test_full_synthetic_pipeline_and_frozen_manifests(tmp_path):
    """Exercise the complete pipeline only on tiny synthetic header fixtures."""
    prepare_audit(tmp_path)
    train = [source_row("P_00001"), source_row("P_00003", "MALIGNANT")]
    test = [source_row("P_00002", partition="test")]
    images_dir = tmp_path / "data/raw/CBIS-DDSM/images"
    images_dir.mkdir()
    for i, row in enumerate(train + test, 1):
        study, series, sop = f"1.2.{i}.1", f"1.2.{i}.2", f"1.2.{i}.3"
        parts = row["image file path"].split("/")
        row["image file path"] = "/".join([parts[0], study, series, parts[3]])
        meta = pydicom.dataset.FileMetaDataset()
        meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = pydicom.uid.DigitalMammographyXRayImageStorageForPresentation
        meta.MediaStorageSOPInstanceUID = sop
        ds = pydicom.dataset.FileDataset("", {}, file_meta=meta, preamble=b"\0" * 128)
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID = study, series, sop
        ds.Modality, ds.Rows, ds.Columns = "MG", 2, 2
        ds.save_as(images_dir / f"{i}.dcm", enforce_file_format=True)
    metadata_file(tmp_path, train)
    metadata_file(tmp_path, test, "test")
    result = audit(tmp_path, full=True)
    assert result["split_status"].startswith("FINAL")
    holdout = tmp_path / "data/splits/test.csv"
    original = holdout.read_bytes()
    assert audit(tmp_path, full=True)["split_status"].startswith("FINAL")
    assert holdout.read_bytes() == original
    test[0]["pathology"] = "MALIGNANT"
    metadata_file(tmp_path, test, "test")
    with pytest.raises(ValueError, match="frozen splits"):
        audit(tmp_path, full=True)
    assert holdout.read_bytes() == original
