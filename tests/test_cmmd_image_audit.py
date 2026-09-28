"""Synthetic local-header and preprocessing guards; no real images or model."""
import ast
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pydicom
import pytest

from src.external_validation import cmmd_image_audit as a


def header(**changes):
    values = dict(PatientID='D1-0001', StudyInstanceUID='1.2', SeriesInstanceUID='1.2.3',
                  SOPInstanceUID='1.2.3.4', SOPClassUID='1.2.840.10008.5.1.4.1.1.1.2',
                  Modality='MG', ImageLaterality='L', ViewPosition='CC', ImageType=['ORIGINAL','PRIMARY'],
                  Rows=24, Columns=12, SamplesPerPixel=1, BitsStored=8)
    values.update(changes)
    return SimpleNamespace(**values)


def record(**changes):
    return dict(a.header_record(header(**changes), {('D1-0001','L'):1}, {'D1-0001'}), bytes=100)


def catalogue(**changes):
    return dict(PatientID='D1-0001', StudyInstanceUID='1.2', SeriesInstanceUID='1.2.3',
                Collection='CMMD',Modality='MG',ImageCount=1,FileSize=100, **changes)


@pytest.mark.parametrize('scheme,code,want', [('SCT','399162004','CC'),('SCT','399368009','MLO'),
                                          ('SRT','R-10242','CC'),('SRT','R-10226','MLO')])
def test_standard_coded_view_without_view_position(scheme,code,want):
    code = SimpleNamespace(CodingSchemeDesignator=scheme,CodeValue=code)
    assert a.view(header(ViewPosition='',ViewCodeSequence=[code])) == want


@pytest.mark.parametrize('changes', [dict(ViewPosition=''),dict(ViewPosition='UNKNOWN'),dict(PartialView='YES'),
    dict(ViewModifierCodeSequence=[SimpleNamespace()]),
    dict(ViewCodeSequence=[SimpleNamespace(CodingSchemeDesignator='SCT',CodeValue='399368009')]),
    dict(ViewPosition='',ViewCodeSequence=[SimpleNamespace(CodingSchemeDesignator='FAKE',CodeValue='399162004')])])
def test_missing_conflicting_modified_view_rejected(changes):
    with pytest.raises(ValueError): a.view(header(**changes))


@pytest.mark.parametrize('changes', [dict(ImageLaterality=''),dict(ImageLaterality='B'),dict(Laterality='R')])
def test_missing_ambiguous_laterality(changes):
    with pytest.raises(ValueError): a.laterality(header(**changes))


def test_series_laterality_explicit_fallback():
    assert a.laterality(header(ImageLaterality='',Laterality='R')) == 'R'


def test_clinical_id_and_contralateral_never_inherit_label():
    assert record()['label'] == 1
    assert 'contralateral_side_not_in_locked_cohort' in record(ImageLaterality='R')['reasons']
    assert record(ImageLaterality='R')['label'] == ''
    assert 'patient_not_locked' in record(PatientID='D2-0001')['reasons']


def test_strict_derived_policy_and_non_mg():
    assert 'derived_or_unspecified_image_type' in record(ImageType=['DERIVED','PRIMARY'])['reasons']
    assert 'non_mammography' in record(Modality='CT')['reasons']
    assert 'unsupported_non_full_object' in record(NumberOfFrames=2)['reasons']


def test_duplicate_sides_rejected():
    row=dict(patient_id='D1-0001',laterality='L',label='1',abnormality='mass')
    with pytest.raises(ValueError): a.clinical_map([row,row])


def test_reconciliation_counts_bytes_ids_series_and_duplicates():
    args = ([catalogue()], {'D1-0001'}, ['1.2.3'])
    result = a.reconcile([record()], *args)
    assert result['status'] == 'PASS' and result['catalogue_bytes'] == 100
    duplicate = a.reconcile([record(),record()], *args)
    assert duplicate['status']=='FAIL' and duplicate['duplicate_sop_uids']==['1.2.3.4']
    assert a.reconcile([dict(record(),bytes=99)],*args)['series_discrepancies'][0]['reasons']==['file_bytes']
    assert a.reconcile([record(PatientID='D2-0001')],*args)['extra_patients']==['D2-0001']
    assert a.reconcile([],*args)['missing_series']==['1.2.3']


def test_unreadable_and_manifest_mismatch_fail():
    assert a.reconcile([dict(header_error='InvalidDicomError')],[catalogue()],{'D1-0001'},[])['status']=='FAIL'


def test_patient_label_counts_can_overlap():
    left = record()
    right = dict(left,laterality='R',label=0,view='MLO')
    c = a.counts([left,right])
    assert c['patients']==1 and c['sides']==2 and c['patients_with_cc_and_mlo']==1
    assert c['benign']['patients']==c['malignant']['patients']==1


def test_deterministic_cohort_hash():
    assert hashlib.sha256(a.csv_bytes([record()],a.COHORT_FIELDS)).digest()==hashlib.sha256(a.csv_bytes([record()],a.COHORT_FIELDS)).digest()
    assert b'\r' not in a.csv_bytes([],a.COHORT_FIELDS)


def output():
    image=np.zeros((224,224),dtype=np.float32);image[2:-2,3:-3]=0.5
    return SimpleNamespace(image=image,metadata=dict(padding_top=2,padding_bottom=2,padding_left=3,padding_right=3))


def test_preprocessing_contract_valid_and_failures():
    result=output();assert len(a.validate_preprocessed(result))==64
    result.image[0,0]=.2
    with pytest.raises(ValueError,match='padding'): a.validate_preprocessed(result)
    for bad in (np.zeros((2,2),dtype=np.float32),np.zeros((224,224),dtype=np.float64),np.full((224,224),np.nan,dtype=np.float32)):
        with pytest.raises(ValueError): a.validate_preprocessed(SimpleNamespace(image=bad,metadata={}))


def test_actual_synthetic_dicom_through_frozen_pipeline(tmp_path):
    from src.preprocessing.canonical import preprocess_dicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
    import yaml
    path=tmp_path/'synthetic.dcm'
    meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID='1.2.840.10008.5.1.4.1.1.1.2';meta.MediaStorageSOPInstanceUID='1.2.3.4'
    ds=FileDataset(str(path),{},file_meta=meta,preamble=b'\0'*128)
    ds.Rows=24;ds.Columns=12;ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2'
    ds.BitsAllocated=8;ds.BitsStored=8;ds.HighBit=7;ds.PixelRepresentation=0
    ds.PixelData=np.arange(288,dtype=np.uint16).astype(np.uint8).tobytes()
    ds.save_as(path,enforce_file_format=True)
    config=yaml.safe_load((a.ROOT/'config/project.yaml').read_text())
    assert len(a.validate_preprocessed(preprocess_dicom(path,config)))==64


def test_audit_imports_no_inference_training_or_network():
    source=(a.ROOT/'src/external_validation/cmmd_image_audit.py').read_text()
    tree=ast.parse(source)
    prohibited={'tensorflow','keras','requests','urllib','subprocess','src.training','src.evaluation.evaluate'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): assert all(n.name not in prohibited for n in node.names)
        if isinstance(node,ast.ImportFrom): assert node.module not in prohibited
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in {'predict','fit','load_model','compile','save_weights','set_weights'}


def test_missing_images_cannot_fabricate_evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'load_locked_ids',lambda root: (_ for _ in ()).throw(FileNotFoundError('No locks')))
    with pytest.raises(FileNotFoundError): a.audit(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_blocked_lock_and_exact_contract(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'load_locked_ids',lambda root: ['D1-0001'])
    monkeypatch.setattr(a,'sha256',lambda path: 'digest')
    required={*a.BASELINE_HASHES,a.CATALOGUE,a.TCIA_MANIFEST,*a.PREPROCESSING,a.COHORT,a.PROCESSED,
              f'{a.EVIDENCE}/cmmd_dicom_headers.csv','src/external_validation/cmmd_image_audit.py'}
    lock=dict(dataset='CMMD',purpose='external_validation',status='BLOCKED',reasons=['derived images'],
        checkpoint=a.CHECKPOINT,checkpoint_sha256=a.CHECKPOINT_SHA256,classifier_threshold=a.THRESHOLD,output_root=a.RESULTS,
        training_allowed=False,fine_tuning_allowed=False,model_selection_allowed=False,calibration_allowed=False,
        threshold_tuning_allowed=False,inference_enabled=False,predictions_generated=0,
        clinical_cohort_sha256='digest',image_cohort_sha256='digest',counts=a.counts([]),bound_files={p:'digest' for p in required})
    path=tmp_path/a.IMAGE_LOCK;path.parent.mkdir(parents=True)
    monkeypatch.setattr(a,'csv_read',lambda path: [])
    path.write_text(json.dumps(lock))
    assert a.verify_image_lock(tmp_path,require_ready=False)['status']=='BLOCKED'
    with pytest.raises(ValueError,match='BLOCKED'): a.verify_image_lock(tmp_path)
    for key,value in [('classifier_threshold',.5),('checkpoint','other.keras'),('training_allowed',True),('output_root','results/experiments')]:
        changed=copy.deepcopy(lock);changed[key]=value;path.write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='lock changed'): a.verify_image_lock(tmp_path)
    changed=copy.deepcopy(lock);changed['bound_files'][a.COHORT]='changed';path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='binding changed'): a.verify_image_lock(tmp_path)
