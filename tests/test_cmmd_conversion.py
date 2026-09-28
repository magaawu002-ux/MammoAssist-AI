"""Synthetic conversion exception, generic-derived rejection and inference isolation."""
import ast
import copy
import json
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
import pytest

from src.external_validation import cmmd_conversion as c
from src.external_validation import cmmd_predict as future


def fixture():
    ds=Dataset();ds.file_meta=FileMetaDataset();ds.file_meta.TransferSyntaxUID='1.2.840.10008.1.2.1'
    values=dict(PatientID='D1-0001',StudyInstanceUID='1.2',SeriesInstanceUID='1.2.3',SOPInstanceUID='1.2.3.4',
        SOPClassUID='1.2.840.10008.5.1.4.1.1.1.2',Modality='MG',ImageLaterality='L',ViewPosition='CC',
        Rows=2294,Columns=1914,BitsAllocated=8,BitsStored=8,HighBit=7,PixelRepresentation=0,
        PhotometricInterpretation='MONOCHROME2',SamplesPerPixel=1,ImageType=['DERIVED','PRIMARY'],
        PresentationIntentType='FOR PRESENTATION',PresentationLUTShape='IDENTITY',BurnedInAnnotation='NO',
        LossyImageCompression='00',ImagerPixelSpacing=[.094090909,.094090909])
    for k,v in values.items():setattr(ds,k,v)
    sides={('D1-0001','L'):0}
    catalogue={'1.2.3':dict(Collection='CMMD',Modality='MG',PatientID='D1-0001',StudyInstanceUID='1.2')}
    original=dict(patient_id='D1-0001',study_uid='1.2',series_uid='1.2.3',sop_uid='1.2.3.4',laterality='L',view='CC',sha256='source')
    return ds,sides,catalogue,original


def test_exception_only_with_official_identity():
    ds,sides,cat,original=fixture()
    assert c.conversion_header(ds,sides,cat,original,'source')['reasons']==''
    assert c.conversion_header(ds,sides,cat,None,'source')['reasons']
    assert c.conversion_header(ds,sides,cat,original,'tampered')['reasons']
    assert c.conversion_header(ds,sides,cat,original,'source',dataset='OTHER')['reasons']
    assert c.conversion_header(ds,sides,{},original,'source')['reasons']


@pytest.mark.parametrize('key,value',[
    ('PatientID','D2-0001'),('SeriesInstanceUID','1.2.99'),('StudyInstanceUID','9.9'),('ImageLaterality','R'),
    ('Laterality','R'),('ViewPosition','MLO'),('ViewPosition',''),('Modality','CT'),
    ('SOPClassUID','1.2.840.10008.5.1.4.1.1.7'),('Rows',128),('Columns',128),
    ('SeriesDescription','ROI crop'),('StudyDescription','screenshot'),('DerivationDescription','annotation composite'),
    ('PartialView','YES'),('ImageType',['DERIVED','SECONDARY']),('ImageType',['DERIVED','PRIMARY','ROI']),
    ('ImageType',['ORIGINAL','PRIMARY']),('BurnedInAnnotation','YES'),('SamplesPerPixel',3),
    ('PhotometricInterpretation','RGB'),('BitsStored',7),('LossyImageCompression','01')])
def test_incompatible_or_ambiguous_objects_rejected(key,value):
    ds,sides,cat,original=fixture();setattr(ds,key,value)
    assert c.conversion_header(ds,sides,cat,original,'source')['reasons']


def test_contralateral_excluded_even_with_source_identity():
    ds,sides,cat,original=fixture();ds.ImageLaterality='R';original['laterality']='R'
    result=c.conversion_header(ds,sides,cat,original,'source')
    assert 'contralateral_side_not_in_locked_cohort' in result['reasons'] and result['label']==''


def test_overlay_and_presentation_objects_rejected():
    ds,sides,cat,original=fixture();ds.add_new((0x6000,0x3000),'OW',b'\0\0')
    assert 'derived_processing_or_overlay_evidence' in c.conversion_header(ds,sides,cat,original,'source')['reasons']


def test_known_16bit_encoding_not_reinterpreted_as_crop():
    ds,sides,cat,original=fixture();ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15
    assert not c.conversion_header(ds,sides,cat,original,'source')['reasons']


def test_pathology_outcome_does_not_change_image_eligibility():
    ds,sides,cat,original=fixture()
    a=c.conversion_header(ds,sides,cat,original,'source')
    b=c.conversion_header(ds,{('D1-0001','L'):1},cat,original,'source')
    assert a['reasons']==b['reasons']=='' and a['label']!=b['label']


def test_pixel_presence_and_geometry_screens():
    ds,*_=fixture()
    with pytest.raises(ValueError,match='missing_pixel_data'):c.pixel_geometry(ds)
    pixels=np.zeros((2294,1914),dtype=np.uint8)
    ds.PixelData=pixels.tobytes()
    with pytest.raises(ValueError,match='constant'):c.pixel_geometry(ds)
    pixels[100:130,100:130]=100;ds.PixelData=pixels.tobytes()
    with pytest.raises(ValueError,match='sparse'):c.pixel_geometry(ds)
    pixels[300:2000,0:1200]=np.arange(1200,dtype=np.uint16).astype(np.uint8)
    ds.PixelData=pixels.tobytes()
    assert c.pixel_geometry(ds)['pixel_data_present'] is True


def test_exact_cc_mlo_pair_requirement():
    rows=[dict(patient_id='D1-0001',laterality='L',view=v) for v in ['CC','MLO']]
    sides={('D1-0001','L'):0}
    assert c.validate_pairs(rows,sides)['complete_pairs']==1
    assert c.validate_pairs(rows+rows[:1],sides)['unexpected_or_missing_views']
    assert c.validate_pairs([] ,sides)['missing_sides']


def test_no_predictions_before_amendment(tmp_path):
    p=tmp_path/c.RESULTS/'predictions/fake.csv';p.parent.mkdir(parents=True);p.write_text('synthetic')
    with pytest.raises(ValueError,match='already exist'):c.no_external_predictions(tmp_path)


def test_no_inference_without_explicit_flag_or_passing_lock(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'verify',lambda root: (_ for _ in ()).throw(ValueError('lock changed')))
    with pytest.raises(ValueError,match='--execute'):future.execute(tmp_path)
    with pytest.raises(ValueError,match='lock changed'):future.execute(tmp_path,confirmed=True)
    assert list(tmp_path.iterdir())==[]


def test_audit_has_no_prediction_dependency_and_future_imports_are_lazy():
    source=Path(c.__file__).read_text();tree=ast.parse(source)
    assert 'tensorflow' not in source and 'load_model' not in source and 'probability' not in source
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in {'fit','predict','compile','load_model'}
    tree=ast.parse(Path(future.__file__).read_text())
    for node in tree.body:
        if isinstance(node,ast.ImportFrom):assert node.module!='tensorflow.keras.models'


@pytest.mark.parametrize('key,value',[('classifier_threshold',.5),('checkpoint','other.keras'),('training_allowed',True),
                                    ('version','other'),('output_root','results/experiments')])
def test_semantic_lock_tamper_stops_before_model(tmp_path,monkeypatch,key,value):
    monkeypatch.setattr(c,'baseline',lambda root: {})
    lock=dict(version=c.VERSION,rule=c.RULE,status=c.READY,dataset='CMMD',purpose='external_validation',
        selected_model='resnet50',checkpoint=c.CHECKPOINT,checkpoint_sha256=c.CHECKPOINT_SHA256,classifier_threshold=c.THRESHOLD,
        documented_conversion_exception=True,training_allowed=False,fine_tuning_allowed=False,calibration_allowed=False,
        model_selection_allowed=False,threshold_tuning_allowed=False,predictions_generated=0,amended_before_inference=True,
        performance_influenced_eligibility=False,output_root=c.RESULTS,execution_requires_explicit_flag=True)
    lock[key]=value
    path=tmp_path/c.PROTOCOL;path.parent.mkdir(parents=True);path.write_text(json.dumps(lock))
    with pytest.raises(ValueError,match='protocol changed'):c.verify(tmp_path)


def test_freeze_preserves_history_and_binds_versioned_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'baseline',lambda root: {})
    rows=[dict(path='synthetic.dcm',sha256='source',patient_id='D1-0001',study_uid='1.2',series_uid='1.2.3',
        sop_uid='1.2.3.'+str(i),laterality='L',view=v,label='0',bytes='100',pixel_sha256='pixels'+str(i),
        array_sha256='array'+str(i),reasons='') for i,v in enumerate(['CC','MLO'])]
    inputs=[c.old.IMAGE_LOCK,c.GEOMETRY,c.SUMMARY,c.REVIEW,c.SOURCE,c.PREDICT_SOURCE,
            c.old.CLINICAL,c.old.PROCESSED,*c.old.PREPROCESSING,c.SOURCES]
    for name in inputs:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic historical bytes')
    (tmp_path/c.GEOMETRY).write_bytes(c.old.csv_bytes(rows,list(rows[0])))
    (tmp_path/c.SOURCES).write_text(json.dumps({'sources':[{'url':'https://example.test/fixture'}]}))
    (tmp_path/c.SUMMARY).write_text(json.dumps(dict(status='REVIEW_PENDING',eligible=c.old.counts(rows),
        suspicious_objects=[],pair_check=dict(missing_sides=[],unexpected_or_missing_views=[]),
        contralateral_exclusions=1,other_exclusions={})))
    (tmp_path/c.EVIDENCE/'cmmd_conversion_visual_review.json').write_text(json.dumps(dict(status='PASS',
        geometry_csv_sha256=c.sha256(tmp_path/c.GEOMETRY))))
    historical=(tmp_path/c.old.IMAGE_LOCK).read_bytes()
    protocol=c.freeze(tmp_path)
    assert protocol['status']==c.READY and protocol['counts']['images']==2
    assert protocol['image_cohort_sha256']==c.sha256(tmp_path/c.COHORT)
    assert protocol['bound_files'][c.SUMMARY]==c.sha256(tmp_path/c.SUMMARY)
    assert json.loads((tmp_path/c.SUMMARY).read_text())['status']==c.READY
    assert (tmp_path/c.old.IMAGE_LOCK).read_bytes()==historical
    with pytest.raises(FileExistsError):c.freeze(tmp_path)
