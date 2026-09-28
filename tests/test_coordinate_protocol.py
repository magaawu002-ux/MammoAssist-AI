"""Locked coordinate admission with synthetic DICOMs; no real data/model access."""

import copy
import hashlib
import json
import shutil

import numpy as np
import pytest

from test_preprocessing import make_dicom
from src.data.cbis_ddsm_metadata import write_csv
from src.data.provenance import sha256_file
from src.explainability.coordinate_protocol import (ANCHOR, PROTOCOL, EVIDENCE, IMPLEMENTATION,
    RULES, VERSION, bound_hash, coordinate_rule, geometry_counts, load_coordinate_protocol)
from src.explainability.roi_audit import audit_roi
from src.explainability.roi_mapping import discover_annotations, map_annotations
from src.utils.paths import PROJECT_ROOT


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def seal(root,protocol):
    save(root/PROTOCOL,protocol)
    save(root/ANCHOR,dict(protocol=PROTOCOL,protocol_sha256=bound_hash(root/PROTOCOL)))


@pytest.fixture
def locked(tmp_path, monkeypatch):
    root=tmp_path
    (root/'config').mkdir();shutil.copyfile(PROJECT_ROOT/'config/localisation.yaml',root/'config/localisation.yaml')
    for name in IMPLEMENTATION:
        destination=root/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(PROJECT_ROOT/name,destination)
    reference='Mass-Training_P_00001_LEFT_CC/1.2.1.1/1.2.1.2/full.dcm'
    image_id='mass_'+hashlib.sha256(reference.encode()).hexdigest()[:24]
    full=make_dicom(root/'data/raw/CBIS-DDSM/images/full.dcm',np.arange(96,dtype=np.uint16).reshape(8,12))
    geometry=dict(original_height=8,original_width=12,crop_box_y0_y1_x0_x1=[0,8,0,12],
        crop_height=8,crop_width=12,resized_height=8,resized_width=12,padding_top=0,padding_bottom=0,
        padding_left=0,padding_right=0,padding_value=0,resize_scale=1,input_size=12,
        preprocessing_version='canonical_v1',reorientation='none',SOPInstanceUID='1.2.1.3')
    # Canonical longest-axis letterbox to 12 x 12.
    geometry.update(padding_top=2,padding_bottom=2)
    image=dict(canonical_image_id=image_id,participant_id='P_00001',official_partition='train',final_partition='validation',
        laterality='LEFT',view='CC',full_mammogram_reference=reference,mapping_method='exact_full_reference_study_series_uid',
        dicom_relative_path=full.relative_to(root).as_posix(),preprocessing_metadata_json=json.dumps(geometry))
    roi=root/'data/raw/CBIS-DDSM/roi_cropped'
    lesions=[]
    for abnormality,shape in (('1',(8,12)),('2',(4,6))):
        case='Mass-Training_P_00001_LEFT_CC_'+abnormality;uid=int(abnormality)+10
        pixels=np.zeros(shape,np.uint16);pixels[1:3,2:4]=255
        make_dicom(roi/case/'mask.dcm',pixels,uid=uid,StudyInstanceUID='1.3',SeriesInstanceUID='1.4',SeriesDescription='ROI mask images')
        make_dicom(roi/case/'crop.dcm',uid=uid+10,StudyInstanceUID='1.3',SeriesInstanceUID='1.4',SeriesDescription='cropped images')
        lesions.append(dict(record_id='lesion:'+abnormality,canonical_image_id=image_id,participant_id='P_00001',
            abnormality_id=abnormality,pathology='BENIGN',full_mammogram_reference=reference,
            roi_mask_reference=case+'/1.3/1.4/mask.dcm',cropped_image_reference=case+'/1.3/1.4/crop.dcm'))
    cases={r['roi_mask_reference'].split('/')[0] for r in lesions}
    mapped=map_annotations(lesions,discover_annotations(roi,cases))
    baseline=[dict(r,final_partition='validation') for r in mapped]
    geom=[];development=[];index=[]
    for abnormality,shape in (('1',(8,12)),('2',(4,6))):
        row=next(r for r in mapped if r['abnormality_id']==abnormality and r['role']=='mask')
        crop=next(r for r in mapped if r['abnormality_id']==abnormality and r['role']=='cropped')
        geom.append(dict(record_id=row['record_id'],canonical_image_id=image_id,participant_id='P_00001',final_partition='validation',
            full_rows=8,full_columns=12,mask_rows=shape[0],mask_columns=shape[1],mask_sha256=sha256_file(roi/row['path'])))
        development.append(dict(record_id=row['record_id'],full_sha256=sha256_file(full),crop_sha256=sha256_file(roi/crop['path'])))
        index.append(dict(panel=abnormality+'.png',canonical_image_id=image_id,abnormality_id=abnormality,
            grid_relation='equal' if abnormality=='1' else 'proportional_with_integer_rounding',
            review_status='SUPPORTED' if abnormality=='1' else 'UNRESOLVED',reviewer='synthetic',
            reviewed_at='2026-09-26',review_notes='Synthetic coordinate decision'))
    for reference in EVIDENCE:
        path=root/reference;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    write_csv(root/'reports/evidence/real_roi_train_audit.csv',baseline)
    write_csv(root/'reports/evidence/roi_alignment_geometry.csv',geom)
    write_csv(root/'reports/evidence/roi_alignment_development_checks.csv',development)
    write_csv(root/'reports/evidence/roi_alignment_review/index.csv',index)
    save(root/'reports/evidence/roi_alignment_human_review.json',dict(approved_rules=RULES,cases=index))
    for r in index:(root/'reports/evidence/roi_alignment_review'/r['panel']).write_bytes(b'synthetic panel')
    sources={}
    for name in ('data/processed/manifests/validation_processed.csv','data/raw/CBIS-DDSM/metadata/mass_case_description_train_set.csv'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic source')
        sources[name]=bound_hash(path)
    evidence=list(EVIDENCE)+['reports/evidence/roi_alignment_review/'+r['panel'] for r in index]
    protocol=dict(rule_version=VERSION,status='LOCKED_FOR_VALIDATION_ALIGNMENT',locked=True,rules=copy.deepcopy(RULES),
        hash_scheme='sha256_lf_text_v1',development_partitions=['train','validation'],
        gradcam_outputs_used=False,test_localisation_outcomes_used=False,
        implementation_sha256={p:bound_hash(root/p) for p in IMPLEMENTATION},
        evidence_sha256={p:bound_hash(root/p) for p in evidence},config_sha256=bound_hash(root/'config/localisation.yaml'),
        counts_by_partition=geometry_counts(geom),partition_source_sha256={'validation':sources})
    seal(root,protocol)
    monkeypatch.setattr('src.explainability.roi_audit.selected_images',lambda r,p:([image],{'split_hashes':{},'cache_fingerprint':'synthetic'}))
    monkeypatch.setattr('src.explainability.roi_audit.annotation_references',lambda r,images,p:lesions)
    return root,protocol


def test_locked_identity_and_exclusion_audit(locked,monkeypatch):
    root,_=locked
    from src.explainability.alignment import transform_mask
    calls=[]
    def guarded(mask,geometry):
        assert mask.shape==(8,12)  # Unequal masks must never reach the transformer.
        calls.append(mask.shape);return transform_mask(mask,geometry)
    monkeypatch.setattr('src.explainability.roi_audit.transform_mask',guarded)
    summary,table,images=audit_roi(root)
    assert summary['status']=='READY_FOR_VALIDATION_LOCALISATION'
    assert (summary['eligible_masks'],summary['excluded_masks'])==(1,1)
    assert summary['final_test_localisation_ready'] is False
    assert len(images)==1 and calls==[(8,12)]
    excluded=next(r for r in table if r['alignment_status']=='excluded')
    assert excluded['canonical_mask_pixels']=='' and excluded['reason']=='exact_spatial_transform_unproven'


@pytest.mark.parametrize('path',[PROTOCOL,'reports/evidence/roi_alignment_review/index.csv',
    'src/explainability/alignment.py','reports/evidence/roi_alignment_review/1.png','config/localisation.yaml'])
def test_protocol_review_code_and_panel_tampering_blocked(locked,path):
    root,_=locked
    with (root/path).open('ab') as f:f.write(b'\nmodified')
    with pytest.raises(ValueError,match='hash mismatch'):audit_roi(root)


def test_proportional_approval_rejected_even_if_receipt_rewritten(locked):
    root,p=locked;p['rules']['unequal_grid']='PROPORTIONAL_NEAREST_NEIGHBOUR';seal(root,p)
    with pytest.raises(ValueError,match='Unsupported coordinate'):load_coordinate_protocol(root)


def test_missing_protocol_receipt_does_not_make_validation_ready(locked):
    root,_=locked;(root/ANCHOR).unlink()
    with pytest.raises(ValueError,match='lock receipt'):audit_roi(root)


def test_coordinate_lock_cannot_authorise_test_or_access_test_metrics(locked,monkeypatch):
    root,_=locked
    def forbidden(*args,**kwargs):pytest.fail('Test manifest or annotations accessed without final lock')
    monkeypatch.setattr('src.explainability.roi_audit.selected_images',forbidden)
    with pytest.raises(ValueError,match='lock-final'):audit_roi(root,partition='test',final_authorized=True)
    assert not (root/'results').exists()


def test_changed_mapping_rejected(locked,monkeypatch):
    root,_=locked
    from src.explainability.roi_mapping import map_annotations
    def changed(*args):
        rows=map_annotations(*args);rows[0]['SOPInstanceUID']='1.99';return rows
    monkeypatch.setattr('src.explainability.roi_audit.map_annotations',changed)
    with pytest.raises(ValueError,match='mappings differ'):audit_roi(root)


def test_mask_mutation_blocks_readiness(locked):
    root,_=locked
    path=next((root/'data/raw/CBIS-DDSM/roi_cropped').rglob('mask.dcm'))
    import pydicom
    ds=pydicom.dcmread(path);ds.PatientName='synthetic mutation';ds.save_as(path,enforce_file_format=True)
    summary,_,_=audit_roi(root)
    assert summary['status']=='NOT_READY'


def test_no_proportional_transform_in_locked_rule():
    assert coordinate_rule((8,12),(8,12))=='IDENTITY'
    assert coordinate_rule((4,6),(8,12))=='EXCLUDE_UNPROVEN'


def test_lf_hash_survives_windows_checkout(tmp_path):
    path=tmp_path/'example.csv';path.write_bytes(b'a,b\n1,2\n');value=bound_hash(path)
    path.write_bytes(b'a,b\r\n1,2\r\n');assert bound_hash(path)==value


def test_locked_evidence_cannot_overwrite_human_review(locked):
    from src.explainability.coordinate_evidence import development,geometry_inventory
    root,_=locked
    before=(root/'reports/evidence/roi_alignment_review/index.csv').read_bytes()
    for action in (development,geometry_inventory):
        with pytest.raises(ValueError,match='overwrite human review'):action(root)
    assert (root/'reports/evidence/roi_alignment_review/index.csv').read_bytes()==before
