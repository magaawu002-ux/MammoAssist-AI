"""Synthetic preprocessing checks; no real dataset or model construction."""

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pydicom
import pytest

from src.data.audit_cbis_ddsm import audit
from src.data.cbis_ddsm_metadata import ALIASES, write_csv
from src.data.provenance import sha256_file
from src.preprocessing.cache import build_cache, cache_relative_path
from src.preprocessing.canonical import crop_background, preprocess_dicom, resize_and_pad
from src.preprocessing.clahe import optional_clahe
from src.preprocessing.dicom import DicomDecodeError, decode_dicom, finite_image
from src.preprocessing.manifests import local_path, read_rows, validated_manifests
from src.preprocessing.model_inputs import custom_cnn_input, replicated_rgb, transfer_input
from src.preprocessing.qa import preprocessing_qa
from src.preprocessing.settings import preprocessing_settings
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT


def make_dicom(path, pixels=None, photometric="MONOCHROME2", uid=1, **tags):
    if pixels is None:
        pixels = np.arange(800, dtype=np.uint16).reshape(40, 20)
    pixels = np.asarray(pixels, dtype=np.uint16)
    meta = pydicom.dataset.FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.DigitalMammographyXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID = f"1.2.{uid}.3"
    ds = pydicom.dataset.FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID = f"1.2.{uid}.1", f"1.2.{uid}.2", f"1.2.{uid}.3"
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.Modality, ds.PhotometricInterpretation = "MG", photometric
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 1, 16, 16, 15, 0
    ds.PixelData = pixels.astype("<u2").tobytes()
    for key, value in tags.items():
        setattr(ds, key, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return path


@pytest.fixture
def cohort(tmp_path):
    """Run the existing full audit solely on three tiny synthetic mammograms."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config/project.yaml").write_bytes((PROJECT_ROOT / "config/project.yaml").read_bytes())
    rows = {"train": [], "test": []}
    for i, part in enumerate(("train", "train", "test"), 1):
        prefix = "Training" if part == "train" else "Test"
        row = dict.fromkeys(ALIASES, "")
        row.update({"patient_id": f"P_{i:05}", "pathology": "MALIGNANT" if i == 2 else "BENIGN",
            "abnormality id": "1", "abnormality type": "mass", "left or right breast": "LEFT", "image view": "CC",
            "image file path": f"Mass-{prefix}_P_{i:05}_LEFT_CC/1.2.{i}.1/1.2.{i}.2/000000.dcm"})
        rows[part].append(row)
        make_dicom(tmp_path / f"data/raw/CBIS-DDSM/images/{i}.dcm", uid=i)
    for part, records in rows.items():
        write_csv(tmp_path / f"data/raw/CBIS-DDSM/metadata/mass_case_description_{part}_set.csv", records)
    audit(tmp_path, full=True)
    return tmp_path


@pytest.mark.parametrize("photometric,inverted", [("MONOCHROME1", True), ("MONOCHROME2", False)])
def test_photometric_observed_range_and_source_unchanged(tmp_path, photometric, inverted):
    pixels = np.array([[100, 200], [350, 900]], dtype=np.uint16)
    path = make_dicom(tmp_path / "source.dcm", pixels, photometric)
    original = path.read_bytes()
    result = decode_dicom(path)
    np.testing.assert_array_equal(result.pixels, 1000 - pixels if inverted else pixels)
    assert result.metadata["inverted"] is inverted
    assert path.read_bytes() == original


def test_padding_excluded_from_inversion_and_normalization(tmp_path):
    pixels = np.array([[65535, 100], [200, 900]], dtype=np.uint16)
    path = make_dicom(tmp_path / "padding.dcm", pixels, "MONOCHROME1", PixelPaddingValue=65535)
    decoded = decode_dicom(path)
    np.testing.assert_array_equal(decoded.pixels, [[100, 900], [800, 100]])
    result = preprocess_dicom(path, load_config(), keep_stages=True)
    np.testing.assert_allclose(result.decoded, [[0, 1], [0.875, 0]])


@pytest.mark.parametrize("photometric", ["RGB", "PALETTE COLOR", ""])
def test_unsupported_photometric(tmp_path, photometric):
    with pytest.raises(DicomDecodeError, match="PhotometricInterpretation"):
        decode_dicom(make_dicom(tmp_path / "bad.dcm", photometric=photometric))


@pytest.mark.parametrize("pixels", [np.empty((0, 2)), np.zeros((2, 2, 2)),
    np.array([[np.nan, 1]]), np.array([[np.inf, 1]]), np.array([[1 + 2j]])])
def test_invalid_arrays(pixels):
    with pytest.raises(DicomDecodeError):
        finite_image(pixels)


@pytest.mark.parametrize("variant", ["corrupt", "missing", "truncated", "constant", "padding_only", "multiframe", "rescale"])
def test_invalid_dicom_pixel_data(tmp_path, variant):
    path = make_dicom(tmp_path / "bad.dcm")
    ds = pydicom.dcmread(path)
    if variant == "corrupt":
        path.write_bytes(b"not a dicom")
    else:
        if variant == "missing":
            del ds.PixelData
        elif variant == "truncated":
            ds.PixelData = b"\0\0"
        elif variant in ("constant", "padding_only"):
            ds.PixelData = np.zeros((40, 20), dtype="<u2").tobytes()
            if variant == "padding_only":
                ds.PixelPaddingValue = 0
                ds["PixelPaddingValue"].VR = "US"
        elif variant == "multiframe":
            ds.NumberOfFrames = 2
        else:
            ds.RescaleSlope = 2
        ds.save_as(path, enforce_file_format=True)
    with pytest.raises(DicomDecodeError):
        decode_dicom(path)


@pytest.mark.parametrize("shape,resized,padding", [((400, 200), (224, 112), (0, 0, 56, 56)),
    ((200, 400), (112, 224), (56, 56, 0, 0)), ((3, 7), (96, 224), (64, 64, 0, 0)),
    ((5, 8), (140, 224), (42, 42, 0, 0))])
def test_aspect_ratio_and_padding(shape, resized, padding):
    output, info = resize_and_pad(np.ones(shape), 224)
    assert output.shape == (224, 224)
    assert (info["resized_height"], info["resized_width"]) == resized
    assert tuple(info[f"padding_{k}"] for k in ("top", "bottom", "left", "right")) == padding
    top, bottom, left, right = padding
    np.testing.assert_allclose(output[top:224-bottom, left:224-right], 1, rtol=0, atol=1e-7)
    assert np.count_nonzero(output) == resized[0] * resized[1]


def test_odd_padding_and_no_flip():
    output, info = resize_and_pad(np.tile(np.arange(8), (5, 1)), 7)
    assert info["padding_top"] == 1 and info["padding_bottom"] == 2
    assert output[2, 0] < output[2, -1]


def test_crop_margin_and_bounds():
    image = np.zeros((100, 80))
    image[20:80, 10:60] = 5
    cropped, info = crop_background(image, True, 0.05)
    assert info["crop_box_y0_y1_x0_x1"] == [15, 85, 6, 64]
    assert cropped.shape == (70, 58) and info["crop_applied"]
    image[0, 0] = 1  # Preserve even an isolated non-background marker; no component removal.
    _, info = crop_background(image, True, 0.05)
    assert info["crop_box_y0_y1_x0_x1"][:1] == [0]
    assert info["crop_box_y0_y1_x0_x1"][2] == 0


@pytest.mark.parametrize("kind", ["constant", "sparse", "disabled", "noisy_borders"])
def test_crop_fallback(kind):
    image = np.zeros((100, 100))
    if kind == "sparse":
        image[50, 50] = 1
    elif kind == "disabled":
        image[20:80, 20:80] = 1
    elif kind == "noisy_borders":
        image = np.arange(10000).reshape(100, 100)
    cropped, info = crop_background(image, kind != "disabled", 0.02)
    np.testing.assert_array_equal(cropped, image)
    assert not info["crop_applied"]


def test_canonical_deterministic_range_no_roi(tmp_path):
    path = make_dicom(tmp_path / "source.dcm")
    before = sha256_file(path)
    first = preprocess_dicom(path, load_config())
    second = preprocess_dicom(path, load_config())
    np.testing.assert_array_equal(first.image, second.image)
    assert first.metadata == second.metadata
    assert first.image.shape == (224, 224) and first.image.dtype == np.float32
    assert np.isfinite(first.image).all() and 0 <= first.image.min() <= first.image.max() <= 1
    assert not first.metadata["clahe"] and first.metadata["reorientation"] == "none"
    assert sha256_file(path) == before


def test_custom_and_replicated_channels():
    image = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
    custom, rgb = custom_cnn_input(image), replicated_rgb(image)
    assert custom.shape == (3, 4, 1) and rgb.shape == (3, 4, 3)
    np.testing.assert_array_equal(custom[..., 0], image)
    for channel in range(3):
        np.testing.assert_array_equal(rgb[..., channel], image * 255)


@pytest.mark.parametrize("family", ["vgg16", "resnet50", "efficientnetb0"])
def test_transfer_adapters_shape_values_and_no_mutation(family):
    image = np.ones((224, 224), dtype=np.float32)
    result = transfer_input(image, family)
    assert result.shape == (224, 224, 3) and result.dtype == np.float32
    expected = [255, 255, 255] if family == "efficientnetb0" else np.array([255, 255, 255]) - [103.939, 116.779, 123.68]
    np.testing.assert_allclose(result[0, 0], expected, atol=1e-4)
    np.testing.assert_array_equal(image, 1)


@pytest.mark.parametrize("array", [np.array([[np.nan]]), np.array([[2]]), np.ones((3, 3, 3))])
def test_adapter_rejects_noncanonical(array):
    with pytest.raises(ValueError):
        custom_cnn_input(array)


def test_clahe_explicit_and_separate():
    image = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    copy_image = image.copy()
    np.testing.assert_array_equal(optional_clahe(image), image)
    result = optional_clahe(image, enabled=True)
    assert result.dtype == np.float32 and np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 1 and not np.array_equal(image, result)
    np.testing.assert_array_equal(image, copy_image)


def test_cache_paths_deterministic():
    settings = preprocessing_settings(load_config())
    first = cache_relative_path("mass_example", settings, "abc123")
    assert first == cache_relative_path("mass_example", settings, "abc123")
    assert first != cache_relative_path("mass_other", settings, "abc123")
    assert "canonical_v1" in first and first.endswith(".npy") and "\\" not in first


def test_missing_splits_refused_without_outputs(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/project.yaml").write_bytes((PROJECT_ROOT / "config/project.yaml").read_bytes())
    with pytest.raises(ValueError, match="manifest missing"):
        build_cache("train", tmp_path, dry_run=True)
    assert not (tmp_path / "data/processed").exists()


def test_validated_dry_run_and_cache(cohort, monkeypatch):
    import src.preprocessing.cache as module
    original = module.preprocess_dicom
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run decoded pixels")
    monkeypatch.setattr(module, "preprocess_dicom", forbidden)
    assert build_cache("train", cohort, dry_run=True)["image_count"] == 1
    assert not (cohort / "data/processed").exists()
    monkeypatch.setattr(module, "preprocess_dicom", original)
    result = build_cache("train", cohort)
    assert result["succeeded"] == 1
    rows = read_rows(cohort / "data/processed/manifests/train_processed.csv")
    image = np.load(cohort / rows[0]["cache_relative_path"], allow_pickle=False)
    assert image.shape == (224, 224) and image.dtype == np.float32
    assert rows[0]["processing_status"] == "success" and rows[0]["final_partition"] == "train"
    assert json.loads(rows[0]["preprocessing_metadata_json"])["original_height"] == 40
    assert (cohort / "reports/evidence/preprocessing_summary.md").is_file()
    with pytest.raises(ValueError, match="already exists"):
        build_cache("train", cohort)


@pytest.mark.parametrize("problem", ["missing_evidence", "failed_evidence", "pending_mapping", "missing_source",
    "roi_reference", "wrong_uid", "changed_metadata", "leakage"])
def test_cache_preflight_rejects_invalid_cohort(cohort, problem):
    split_path = cohort / "data/splits/train.csv"
    rows = read_rows(split_path)
    if problem == "missing_evidence":
        (cohort / "reports/evidence/leakage_validation.txt").unlink()
    elif problem == "failed_evidence":
        (cohort / "reports/evidence/leakage_validation.txt").write_text('{"conclusion":"FAIL"}')
    elif problem == "missing_source":
        (cohort / rows[0]["dicom_relative_path"]).unlink()
    elif problem == "wrong_uid":
        path = cohort / rows[0]["dicom_relative_path"]
        ds = pydicom.dcmread(path)
        ds.SeriesInstanceUID = "9.9.9"
        ds.save_as(path, enforce_file_format=True)
    elif problem == "changed_metadata":
        with (cohort / "data/raw/CBIS-DDSM/metadata/mass_case_description_train_set.csv").open("a") as handle:
            handle.write("\n")
    else:
        if problem == "pending_mapping":
            rows[0]["mapping_status"] = "pending"
        elif problem == "leakage":
            rows[0]["participant_id"] = read_rows(cohort / "data/splits/test.csv")[0]["participant_id"]
        else:
            rows[0]["full_mammogram_reference"] = rows[0]["full_mammogram_reference"].replace("LEFT_CC/", "LEFT_CC_1/")
        write_csv(split_path, rows)
    with pytest.raises(ValueError):
        build_cache("train", cohort)
    assert not (cohort / "data/processed").exists()


def test_processing_failure_logged_not_published(cohort):
    row = read_rows(cohort / "data/splits/train.csv")[0]
    path = cohort / row["dicom_relative_path"]
    ds = pydicom.dcmread(path)
    del ds.PixelData  # Header is still valid; failure occurs only during decoding.
    ds.save_as(path, enforce_file_format=True)
    with pytest.raises(ValueError, match="no final processed manifest"):
        build_cache("train", cohort)
    assert not (cohort / "data/processed/manifests/train_processed.csv").exists()
    attempts = read_rows(cohort / "reports/evidence/preprocessing_train_attempt.csv")
    assert attempts[0]["processing_status"] == "failed" and attempts[0]["processing_error"]
    evidence = json.loads((cohort / "reports/evidence/preprocessing_train.json").read_text())
    assert evidence["failed"] == 1 and evidence["succeeded"] == 0


def test_qa_only_synthetic_validated_images(cohort):
    row = read_rows(cohort / "data/splits/train.csv")[0]
    destination = cohort / "synthetic_qa.png"
    preprocessing_qa(cohort, [row["canonical_image_id"]], destination)
    assert destination.is_file()
    with pytest.raises(ValueError, match="unvalidated"):
        preprocessing_qa(cohort, ["invented_id"], destination)


@pytest.mark.parametrize("reference", ["../external.dcm", "C:/external.dcm", "data/raw/CBIS-DDSM/images/../../external.dcm"])
def test_path_containment(tmp_path, reference):
    with pytest.raises(ValueError):
        local_path(tmp_path, reference, ("data", "raw", "CBIS-DDSM", "images"))


def test_settings_reject_unknown_and_invalid_values():
    config = copy.deepcopy(load_config())
    config["preprocessing"]["clahe"] = True
    with pytest.raises(ValueError):
        preprocessing_settings(config)
    config = copy.deepcopy(load_config())
    config["preprocessing"]["crop_margin_fraction"] = float("nan")
    with pytest.raises(ValueError):
        preprocessing_settings(config)
