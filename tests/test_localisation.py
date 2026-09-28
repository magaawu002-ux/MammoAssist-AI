"""Synthetic annotations only; never evaluate real classifier or localisation results."""

import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pydicom
import pytest

from test_preprocessing import cohort, make_dicom
from test_training import cached_cohort
from src.data.cbis_ddsm_metadata import write_csv
from src.data.provenance import sha256_file
from src.explainability.alignment import prove_full_grid, transform_mask, union_masks
from src.explainability.localisation import evaluate_localisation, lock_localisation, require_final_lock
from src.explainability.localisation_metrics import localisation_metrics
from src.explainability.roi_audit import audit_roi
from src.explainability.roi_mapping import discover_annotations, map_annotations
from src.explainability.roi_masks import decode_mask
from src.preprocessing.manifests import read_rows
from src.training.data import _load_processed
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT


def lesion(abnormality="1"):
    case = f"Mass-Training_P_00001_LEFT_CC_{abnormality}"
    return {"record_id": f"source:{abnormality}", "canonical_image_id": "image", "participant_id": "P_00001",
            "abnormality_id": abnormality, "pathology": "MALIGNANT",
            "full_mammogram_reference": "Mass-Training_P_00001_LEFT_CC/1.1/1.2/full.dcm",
            "roi_mask_reference": f"{case}/1.3/1.4/mask.dcm", "cropped_image_reference": f"{case}/1.3/1.4/crop.dcm"}


def inventory_entry(role="mask", sop="1.8", abnormality="1"):
    return {"path": f"Mass-Training_P_00001_LEFT_CC_{abnormality}/{role}_{sop}.dcm", "role": role,
            "StudyInstanceUID": "1.3", "SeriesInstanceUID": "1.4", "SOPInstanceUID": sop, "read_status": "readable"}


def test_unique_mask_and_crop_same_series():
    rows = map_annotations([lesion()], [inventory_entry(), inventory_entry("cropped", "1.9")])
    assert [r["mapping_status"] for r in rows] == ["unique", "unique"]
    assert rows[0]["SOPInstanceUID"] == "1.8" and rows[1]["SOPInstanceUID"] == "1.9"


@pytest.mark.parametrize("variant", ["two_masks", "unknown_role", "duplicate_sop"])
def test_ambiguous_mapping_not_guessed(variant):
    extra = inventory_entry("unknown" if variant == "unknown_role" else "mask", "1.8" if variant == "duplicate_sop" else "1.9")
    rows = map_annotations([lesion()], [inventory_entry(), extra])
    assert rows[0]["mapping_status"] == "ambiguous" and rows[0]["path"] == ""


def test_missing_and_multiple_lesions_preserved():
    assert map_annotations([lesion()], [])[0]["mapping_status"] == "missing"
    rows = map_annotations([lesion("1"), lesion("2")], [inventory_entry(), inventory_entry(sop="1.9", abnormality="2")])
    masks = [r for r in rows if r["role"] == "mask"]
    assert len(masks) == 2 and all(r["mapping_status"] == "unique" for r in masks)
    assert {r["record_id"] for r in masks} == {"source:1", "source:2"}


@pytest.mark.parametrize("photo", ["MONOCHROME1", "MONOCHROME2"])
def test_binary_mask_polarity(tmp_path, photo):
    expected = np.zeros((6, 8), bool); expected[2:4, 3:6] = True
    pixels = expected.astype(np.uint16)*255
    if photo == "MONOCHROME1": pixels = 255-pixels
    path = make_dicom(tmp_path / "mask.dcm", pixels, photo, SeriesDescription="ROI mask images")
    decoded = decode_mask(path, "1.2.1.3")
    np.testing.assert_array_equal(decoded.mask, expected)
    assert decoded.metadata["Rows"] == 6 and decoded.metadata["photometric"] == photo


@pytest.mark.parametrize("variant", ["empty", "grayscale", "cropped", "unknown", "wrong_sop"])
def test_invalid_mask_rejected(tmp_path, variant):
    pixels = np.zeros((6, 8), np.uint16)
    if variant != "empty": pixels[2:4, 3:6] = 255
    if variant == "grayscale": pixels[3, 4] = 70
    description = "cropped images" if variant == "cropped" else ("unknown" if variant == "unknown" else "ROI mask images")
    path = make_dicom(tmp_path / "mask.dcm", pixels, SeriesDescription=description)
    with pytest.raises(ValueError):
        decode_mask(path, "9.9" if variant == "wrong_sop" else "1.2.1.3")


def geometry():
    return {"original_height": 8, "original_width": 12, "crop_box_y0_y1_x0_x1": [1, 7, 2, 10],
            "crop_height": 6, "crop_width": 8, "resized_height": 3, "resized_width": 4,
            "padding_top": 0, "padding_bottom": 1, "padding_left": 0, "padding_right": 0,
            "padding_value": 0, "resize_scale": 0.5, "input_size": 4,
            "preprocessing_version": "canonical_v1", "reorientation": "none"}


def test_exact_nearest_neighbour_alignment_and_union():
    mask = np.zeros((8, 12), bool); mask[3:5, 4:6] = True
    result = transform_mask(mask, geometry())
    expected = np.zeros((4, 4), bool); expected[1, 1] = True
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.bool_ and set(np.unique(result)) == {False, True}
    second = expected.copy(); second[2, 2] = True
    np.testing.assert_array_equal(union_masks([result, second]), second)


@pytest.mark.parametrize("variant", ["shape", "clipped", "vanished", "padding", "orientation"])
def test_alignment_rejects_unproven_or_lost_geometry(variant):
    mask = np.zeros((8, 12), bool); mask[3:5, 4:6] = True
    config = geometry()
    if variant == "shape": mask = mask[:7]
    elif variant == "clipped": mask[0, 0] = True
    elif variant == "vanished": mask[:] = False; mask[2, 3] = True
    elif variant == "padding": config["padding_left"] = 1
    else: config["reorientation"] = "flipped"
    with pytest.raises(ValueError):
        transform_mask(mask, config)


def header(sop):
    ds = pydicom.dataset.Dataset()
    ds.Rows, ds.Columns, ds.SOPInstanceUID = 8, 12, sop
    ds.FrameOfReferenceUID = "1.20"
    ds.ImagePositionPatient, ds.ImageOrientationPatient, ds.PixelSpacing = [0,0,0], [1,0,0,0,1,0], [0.1,0.1]
    return ds


def test_grid_proof_requires_more_than_equal_shape():
    a, b = header("1.1"), header("1.2")
    assert prove_full_grid(a,b).startswith("matching_DICOM")
    del a.ImagePositionPatient
    with pytest.raises(ValueError, match="UNVERIFIED_ALIGNMENT"):
        prove_full_grid(a,b)
    review = {"method":"reviewed_full_grid_identity", "mask_sop_uid":"1.1", "full_sop_uid":"1.2",
              "reviewer":"synthetic", "reviewed_at":"2026-09-25", "evidence":"synthetic verification"}
    assert prove_full_grid(a,b,review) == "reviewed_full_grid_identity"
    a.ImagePositionPatient = [1,0,0]
    with pytest.raises(ValueError, match="Conflicting"):
        prove_full_grid(a,b,review)


def test_known_localisation_metrics():
    heat = np.array([[0.1,0.9],[0.8,0.2]])
    mask = np.array([[False, True],[False, True]])
    result = localisation_metrics(heat, mask)
    assert result["pointing_game"] and result["lesion_hit"]
    assert result["iou"] == pytest.approx(1/3) and result["dice"] == 0.5
    assert result["activation_mass_inside"] == pytest.approx(1.1/2)
    miss = localisation_metrics(np.array([[1.0,0],[0,0]]),mask)
    assert not miss["pointing_game"] and not miss["lesion_hit"] and miss["iou"] == 0


def test_ties_and_zero_maps_are_explicit():
    mask = np.array([[False,True],[False,False]])
    result = localisation_metrics(np.ones((2,2)),mask)
    assert not result["pointing_game"] and result["max_tie_count"] == 4
    zero = localisation_metrics(np.zeros((2,2)), mask)
    assert not zero["pointing_game"] and zero["activation_mass_inside"] is None
    assert zero["iou"] == zero["dice"] == 0


def test_discovery_prunes_test_and_nonselected_cases(tmp_path, monkeypatch):
    case = "Mass-Training_P_00001_LEFT_CC_1"
    make_dicom(tmp_path / case / "mask.dcm", SeriesDescription="ROI mask images")
    make_dicom(tmp_path / "Mass-Test_P_00002_LEFT_CC_1" / "never_read.dcm")
    original = pydicom.dcmread
    def guarded(path, **kwargs):
        assert "Mass-Test" not in str(path)
        assert kwargs["stop_before_pixels"] is True
        return original(path, **kwargs)
    monkeypatch.setattr(pydicom,"dcmread",guarded)
    rows = discover_annotations(tmp_path,{case})
    assert len(rows)==1 and rows[0]["role"]=="mask"


def test_absent_roi_not_ready_no_implicit_manifest_or_test_access(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    shutil.copyfile(PROJECT_ROOT / "config/localisation.yaml", tmp_path / "config/localisation.yaml")
    def forbidden(*args, **kwargs):
        pytest.fail("Absent ROI audit must not read any image manifests")
    monkeypatch.setattr("src.explainability.roi_audit.selected_images", forbidden)
    summary, rows, images = audit_roi(tmp_path)
    assert summary["status"] == "NOT_READY" and not rows and not images
    assert "mask_mapping_counts" not in summary
    with pytest.raises(ValueError, match="explicit locked"):
        audit_roi(tmp_path, partition="test")
    with pytest.raises(ValueError, match="lock-final"):
        require_final_lock(tmp_path)


def test_classifier_loader_rejects_annotation_source_even_with_consistent_logs(cached_cohort):
    root=cached_cohort
    processed=root / "data/processed/manifests/train_processed.csv"
    rows=read_rows(processed)
    split=root / "data/splits/train.csv"
    original=read_rows(split)
    bad="data/raw/CBIS-DDSM/roi_cropped/mask.dcm"
    rows[0]["dicom_relative_path"]=bad; original[0]["dicom_relative_path"]=bad
    write_csv(processed, rows); write_csv(split, original)
    write_csv(root / "reports/evidence/preprocessing_train_attempt.csv", rows)
    evidence=root / "reports/evidence/preprocessing_train.json"
    summary=json.loads(evidence.read_text()); summary["split_sha256"]["train"]=sha256_file(split)
    evidence.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="outside required"):
        _load_processed(root,"train",load_config())


def test_final_requires_lock_before_any_test_audit(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unlocked final action touched test annotations")
    monkeypatch.setattr("src.explainability.localisation.audit_roi",forbidden)
    with pytest.raises(ValueError, match="lock-final"):
        evaluate_localisation(tmp_path,final=True)


def test_lock_checks_protocol_and_repeated_final_attempt(tmp_path, monkeypatch):
    from src.explainability.localisation import write_json
    identity={"protocol":{"threshold":0.5}}
    monkeypatch.setattr("src.explainability.localisation.protocol_identity", lambda root: identity)
    run="results/explainability/validation/synthetic"
    directory=tmp_path/run
    write_json(directory/"summary.json",{"status":"COMPLETE","partition":"validation","identity":identity,"artifact_sha256":{}})
    lock_localisation(tmp_path,run)
    assert require_final_lock(tmp_path)["status"]=="LOCKED"
    identity["protocol"]["threshold"]=0.6
    with pytest.raises(ValueError,match="changed"):
        require_final_lock(tmp_path)
    identity["protocol"]["threshold"]=0.5
    write_json(tmp_path/"results/explainability/final_localisation_attempt.json",{"status":"STARTED"})
    with pytest.raises(ValueError,match="already attempted"):
        require_final_lock(tmp_path)


def test_end_to_end_synthetic_audit_validation_and_locked_final(cohort):
    """Small genuine gradient-derived maps, synthetic annotations and a toy model only."""
    from test_gradcam import toy_classifier
    from src.data.audit_cbis_ddsm import audit
    from src.preprocessing.cache import build_cache
    from src.explainability.localisation import write_json
    root = cohort
    shutil.copyfile(PROJECT_ROOT / "config/localisation.yaml",root / "config/localisation.yaml")
    for part in ("train","test"):
        source=root/f"data/raw/CBIS-DDSM/metadata/mass_case_description_{part}_set.csv"
        rows=read_rows(source)
        for row in rows:
            i=int(row["patient_id"].split("_")[1])
            full=root/f"data/raw/CBIS-DDSM/images/{i}.dcm"
            ds=pydicom.dcmread(full)
            for key in ("ImagePositionPatient","ImageOrientationPatient","PixelSpacing","FrameOfReferenceUID"):
                setattr(ds,key,getattr(header("1.1"),key))
            ds.save_as(full,enforce_file_format=True)
            case=row["image file path"].split("/")[0]+"_1"
            uid=90+i
            ref=f"{case}/1.2.{uid}.1/1.2.{uid}.2"
            row["ROI mask file path"]=ref+"/000001.dcm"
            row["cropped image file path"]=ref+"/000000.dcm"
            pixels=np.zeros((40,20),np.uint16); pixels[12:24,6:12]=255
            tags={key:getattr(ds,key) for key in ("ImagePositionPatient","ImageOrientationPatient","PixelSpacing","FrameOfReferenceUID")}
            make_dicom(root/f"data/raw/CBIS-DDSM/roi_cropped/{case}/mask.dcm", pixels, uid=uid,
                       SeriesDescription="ROI mask images",**tags)
            make_dicom(root/f"data/raw/CBIS-DDSM/roi_cropped/{case}/crop.dcm", uid=190+i,
                       StudyInstanceUID=f"1.2.{uid}.1", SeriesInstanceUID=f"1.2.{uid}.2", SeriesDescription="cropped images")
        write_csv(source,rows)
    audit(root,full=True)
    build_cache("validation",root); build_cache("test",root)
    readiness, table, images=audit_roi(root)
    assert readiness["status"]=="READY"
    assert readiness["mask_mapping_counts"]=={"unique":1}
    assert all(r["alignment_status"]=="verified" for r in table if r["role"]=="mask")
    run="results/experiments/resnet50/synthetic"
    checkpoint=root/run/"best_model.keras"
    checkpoint.parent.mkdir(parents=True)
    toy_classifier().save(checkpoint)
    write_json(root/"results/experiments/final_selection.json",{
        "status":"LOCKED_FOR_FINAL_TEST","run":run,"checkpoint_sha256":sha256_file(checkpoint),
        "split_hashes":readiness["split_hashes"],"cache_fingerprint":readiness["cache_fingerprint"]})
    result=evaluate_localisation(root)
    assert result["status"]=="COMPLETE"
    assert len(read_rows(root/result["run"]/"localisation_metrics.csv"))==2
    lock_localisation(root,result["run"])
    assert evaluate_localisation(root,final=True)["status"]=="COMPLETE"
    with pytest.raises(ValueError,match="already attempted"):
        evaluate_localisation(root,final=True)
