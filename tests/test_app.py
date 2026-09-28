"""Synthetic app/input/export tests; no real model or medical files required."""

from io import BytesIO
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from test_preprocessing import make_dicom
from src.app import core
from src.app.export import report_lines,png_report,pdf_report
from src.data.provenance import sha256_file
from src.explainability.coordinate_protocol import bound_hash
from src.preprocessing.canonical import preprocess_dicom
from src.preprocessing.model_inputs import transfer_input
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT


def raster(format='PNG',pixels=None):
    if pixels is None: pixels=np.arange(1200,dtype=np.uint16).reshape(40,30)%256
    out=BytesIO();Image.fromarray(pixels.astype(np.uint8)).save(out,format=format);return out.getvalue()


def synthetic_model(score=.6,fail=False):
    class Explainer:
        calls=0
        def explain(self,image):
            self.calls+=1
            if fail: raise RuntimeError('PRIVATE PATIENT DETAILS')
            assert image.shape==(224,224) and image.dtype==np.float32
            return SimpleNamespace(malignant_probability=score,target_layer=core.TARGET,
                                   heatmap=np.full(image.shape,.25,np.float32),zero_heatmap=False)
    return core.LockedModel(Explainer())


@pytest.fixture
def locked_app(tmp_path):
    def save(relative,value):
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    (tmp_path/'config').mkdir();shutil.copyfile(PROJECT_ROOT/'config/project.yaml',tmp_path/'config/project.yaml')
    paths=['src/explainability/gradcam.py','src/explainability/overlay.py']+[
        'src/preprocessing/'+name+'.py' for name in ('canonical','dicom','manifests','model_inputs','settings')]
    for path in paths:
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT_ROOT/path,p)
    selection=save('results/experiments/final_selection.json',dict(run=str(Path(core.CHECKPOINT).parent).replace('\\','/'),
        checkpoint_sha256=core.CHECKPOINT_HASH,threshold=core.THRESHOLD))
    coordinate=save('reports/evidence/roi_alignment_protocol.json',dict(implementation_sha256={p:bound_hash(tmp_path/p) for p in paths if 'preprocessing' in p}))
    from src.preprocessing.settings import preprocessing_settings
    save('reports/evidence/preprocessing_validation.json',dict(status='COMPLETE',settings=preprocessing_settings(load_config()),
        implementation={'source_sha256':{Path(p).name:sha256_file(tmp_path/p) for p in paths if 'preprocessing' in p}}))
    save('results/explainability/localisation_lock.json',dict(classifier_threshold=core.THRESHOLD,target_layer=core.TARGET,heatmap_threshold=.5,
        identity=dict(checkpoint=core.CHECKPOINT,checkpoint_sha256=core.CHECKPOINT_HASH,classifier_selection_sha256=sha256_file(selection),
                      coordinate_protocol_sha256=bound_hash(coordinate),implementation_sha256={Path(p).name:sha256_file(tmp_path/p) for p in paths if 'explainability' in p})))
    return tmp_path


def test_selected_contract(locked_app):
    c=core.contract(locked_app)
    assert c['checkpoint']==core.CHECKPOINT and c['threshold']==.3753475248813629
    assert c['input_size']==224 and c['target_layer']=='conv5_block3_out' and c['clahe'] is False


@pytest.mark.parametrize('field,value',[('classifier_threshold',.5),('target_layer','other'),('heatmap_threshold',.6)])
def test_changed_protocol_rejected(locked_app,field,value):
    path=locked_app/'results/explainability/localisation_lock.json';v=json.loads(path.read_text());v[field]=value;path.write_text(json.dumps(v))
    with pytest.raises(core.AppError): core.contract(locked_app)


def test_missing_checkpoint_fails_safely(locked_app):
    with pytest.raises(core.AppError,match='could not be verified'): core.contract(locked_app,verify_checkpoint=True)


@pytest.mark.parametrize('format,suffix',[('PNG','.png'),('JPEG','.jpg'),('JPEG','.jpeg')])
def test_raster_inputs(format,suffix):
    upload=core.prepare_upload(raster(format),'anonymous'+suffix,load_config())
    assert upload.canonical.shape==(224,224) and upload.canonical.dtype==np.float32
    assert upload.canonical.min()>=0 and upload.canonical.max()<=1
    assert any('DICOM metadata' in w for w in upload.warnings)


@pytest.mark.parametrize('photometric',['MONOCHROME1','MONOCHROME2'])
def test_dicom_matches_training_pipeline_and_redacts_metadata(tmp_path,photometric):
    path=make_dicom(tmp_path/'full.dcm',photometric=photometric,PatientName='PRIVATE_PERSON',PatientID='PRIVATE_ID')
    upload=core.prepare_upload(path.read_bytes(),'full.dcm',load_config())
    np.testing.assert_array_equal(upload.canonical,preprocess_dicom(path,load_config()).image)
    assert 'PRIVATE' not in json.dumps(upload.technical)
    assert not any('UID' in k or 'Patient' in k for k in upload.technical)
    assert upload.technical['inverted']==(photometric=='MONOCHROME1')


@pytest.mark.parametrize('filename',['roi.png','cropped-lesion.jpg','mask.dcm','Mass-Training_P_00001_LEFT_CC_1.png'])
def test_annotation_names_rejected(filename):
    with pytest.raises(core.AppError,match='ROI masks'): core.prepare_upload(raster(),filename,load_config())


@pytest.mark.parametrize('tag,value',[('SeriesDescription','ROI mask images'),('ImageComments','cropped images'),('PatientID','Mass-Test_P_00001_LEFT_CC_1')])
def test_annotation_metadata_rejected(tmp_path,tag,value):
    path=make_dicom(tmp_path/'renamed.dcm',**{tag:value})
    with pytest.raises(core.AppError,match='ROI/cropped'): core.prepare_upload(path.read_bytes(),'renamed.dcm',load_config())


def test_binary_mask_renamed_as_mammogram_rejected():
    pixels=np.zeros((40,30),np.uint8);pixels[10:20,10:20]=255
    with pytest.raises(core.AppError,match='Binary'): core.prepare_upload(raster(pixels=pixels),'full.png',load_config())


@pytest.mark.parametrize('data,name',[(b'bad','x.png'),(b'bad','x.dcm'),(b'bad','x.txt'),(b'','x.png')])
def test_invalid_upload(data,name):
    with pytest.raises(core.AppError): core.prepare_upload(data,name,load_config())


@pytest.mark.parametrize('change',['pixel_data','frames','modality'])
def test_invalid_dicom_dimensions_or_pixels(tmp_path,change):
    import pydicom
    path=make_dicom(tmp_path/'full.dcm');ds=pydicom.dcmread(path)
    if change=='pixel_data': del ds.PixelData
    elif change=='frames': ds.NumberOfFrames=2
    else: ds.Modality='CT'
    ds.save_as(path,enforce_file_format=True)
    with pytest.raises(core.AppError): core.prepare_upload(path.read_bytes(),'full.dcm',load_config())


def test_quality_warning_not_certification():
    q=core.quality_checks(np.zeros((40,30),np.float32))
    assert len(q['warnings'])==3 and 'not clinical' in q['interpretation']


def test_resnet_adapter_is_existing_adapter():
    from tensorflow.keras.applications.resnet50 import preprocess_input
    canonical=np.full((224,224),.5,np.float32)
    np.testing.assert_array_equal(transfer_input(canonical,'resnet50'),preprocess_input(np.repeat(canonical[...,None],3,axis=2)*255))


def test_genuine_explainer_entrypoint_and_same_score():
    resource=synthetic_model(); image=np.ones((224,224),np.float32)*.3
    result=core.analyse(image,resource)
    assert resource.explainer.calls==1 and result['score']==.6
    assert result['threshold']==core.THRESHOLD and result['overlay'].shape==(224,224,3)
    assert 'above' in result['decision'] and '%' not in result['decision']


def test_gradcam_failure_has_no_intensity_fallback():
    with pytest.raises(core.AppError,match='No substitute') as e: core.analyse(np.zeros((224,224),np.float32),synthetic_model(fail=True))
    assert 'PRIVATE' not in str(e.value)


@pytest.mark.parametrize('score',[float('nan'),float('inf'),-1,2])
def test_nonfinite_model_score_rejected(score):
    with pytest.raises(core.AppError): core.analyse(np.zeros((224,224),np.float32),synthetic_model(score))


def test_exports_disclaimers_and_no_identifier():
    image=np.zeros((224,224),np.float32);result=core.analyse(image,synthetic_model())
    args=(result,'upload-012345abcdef','2026-09-26T00:00:00+00:00',['Resolution warning'])
    text='\n'.join(report_lines(*args))
    assert core.SCORE_NOTE in text and core.ATTENTION_NOTE in text and core.DISCLAIMER in text
    assert str(core.THRESHOLD) in text and 'malignant-pattern score' in text
    png=png_report(result,image,*args[1:]);pdf=pdf_report(result,image,*args[1:])
    assert Image.open(BytesIO(png)).format=='PNG' and pdf.startswith(b'%PDF')
    assert b'Not a medical device' in pdf and b'Resolution warning' in pdf
    assert b'PatientName' not in pdf and b'SOPInstanceUID' not in pdf
    with pytest.raises(ValueError): report_lines(result,'Patient-Name','time',[])


def test_performance_comes_from_aggregate_evidence(tmp_path):
    from src.data.cbis_ddsm_metadata import write_csv
    base=tmp_path/'reports/evidence/model_evaluation';base.mkdir(parents=True)
    metrics={k:.123 for k in ('sensitivity','specificity','balanced_accuracy','roc_auc','pr_auc','brier_score')}
    write_csv(base/'resnet50_validation_vs_test.csv',[dict(partition='OFFICIAL HOLDOUT TEST',threshold=core.THRESHOLD,**metrics)])
    write_csv(base/'cohort_counts.csv',[dict(partition='test',images=17)])
    path=tmp_path/'reports/evidence/final_localisation/summary.json';path.parent.mkdir()
    path.write_text(json.dumps(dict(heatmap_threshold=.5,groups={'lesion/overall':{'mean_iou':.01}})))
    data=core.performance(tmp_path)
    assert data['images']==17 and data['metrics']['sensitivity']==.123


def ui_fixture(monkeypatch,root):
    details=core.contract(root)
    monkeypatch.setattr(core,'contract',lambda:details)
    monkeypatch.setattr(core,'performance',lambda:dict(images=17,metrics={'sensitivity':.1},localisation={'mean_iou':.01,'pointing_game_rate':.02}))
    import src.app.dashboard as dashboard
    monkeypatch.setattr(dashboard,'validation_evidence',lambda:dict(flip_min=.02,flip_max=.21,most_disruptive='contrast_down',
        any_flip_count=8,images=20,sanity_conclusion='The Validation occlusion sanity check did not provide strong support for the expected Grad-CAM faithfulness behaviour.'))
    monkeypatch.setattr(dashboard,'dashboard_evidence',lambda:dict(images=17,participants=9,
        metrics={k:.123 for k in ('sensitivity','specificity','balanced_accuracy','roc_auc','pr_auc','brier_score')},
        confidence_intervals={k:(.01,.3) for k in ('sensitivity','specificity','balanced_accuracy','roc_auc')},
        false_negatives=2,false_negative_rate=.2,localisation=dict(mean_iou=.01,mean_dice=.02,pointing_game_rate=.03,lesion_hit_rate=.4)))
    import src.utils.paths
    monkeypatch.setattr(src.utils.paths,'PROJECT_ROOT',root)
    checkpoint=root/core.CHECKPOINT;checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_bytes(b'synthetic')


def test_streamlit_startup_has_no_model_or_evaluation(monkeypatch,locked_app):
    from streamlit.testing.v1 import AppTest
    import src.app.runtime as runtime
    ui_fixture(monkeypatch,locked_app)
    def forbidden(*args,**kwargs): pytest.fail('No model load/inference on startup')
    monkeypatch.setattr(runtime,'model_resource',forbidden)
    app=AppTest.from_file(str(PROJECT_ROOT/'src/app/streamlit_app.py'),default_timeout=15).run()
    assert not app.exception
    assert app.title[0].value==core.TITLE
    assert any('not a calibrated clinical probability' in t.value for t in app.markdown)
    assert not app.get('download_button')


def test_streamlit_synthetic_upload_result_and_clear(monkeypatch,locked_app):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    import src.app.runtime as runtime
    ui_fixture(monkeypatch,locked_app)
    # Between the rounded display value and the exact threshold: must stay below.
    data=raster(); resource=synthetic_model(score=.37532)
    scientific_files={p:p.read_bytes() for folder in ('config','reports','results')
                      for p in (locked_app/folder).rglob('*') if p.is_file()}
    monkeypatch.setattr(st,'file_uploader',lambda *a,**k:SimpleNamespace(name='full.png',getvalue=lambda:data))
    monkeypatch.setattr(runtime,'model_resource',lambda *a:resource)
    app=AppTest.from_file(str(PROJECT_ROOT/'src/app/streamlit_app.py'),default_timeout=15).run()
    assert not app.exception
    assert app.button[0].label=='Analyse full mammogram' or any(b.label=='Analyse full mammogram' for b in app.button)
    app.checkbox[0].check().run()
    next(b for b in app.button if b.label=='Analyse full mammogram').click().run()
    assert not app.exception and resource.explainer.calls==1
    assert any(m.label=='Model output: malignant-pattern score' for m in app.metric)
    assert next(m.value for m in app.metric if m.label=='Research decision threshold')=='0.3753'
    assert app.session_state['result']['threshold']==0.3753475248813629
    assert app.session_state['result']['decision']=='Output is below the prespecified research threshold.'
    assert any(str(core.THRESHOLD) in str(j.value) for j in app.json)
    assert len(app.get('download_button'))==2
    monkeypatch.setattr(st,'file_uploader',lambda *a,**k:None)
    next(b for b in app.button if b.label.startswith('Clear uploaded')).click().run()
    assert not app.exception and 'result' not in app.session_state
    assert scientific_files=={p:p.read_bytes() for folder in ('config','reports','results')
                             for p in (locked_app/folder).rglob('*') if p.is_file()}


def test_decision_boundary_uses_exact_locked_threshold():
    assert core.THRESHOLD==0.3753475248813629
    assert 'below' in core.score_wording(np.nextafter(core.THRESHOLD,-np.inf))
    assert 'above or equal to' in core.score_wording(core.THRESHOLD)


def test_model_load_is_resource_cached(monkeypatch):
    import src.app.runtime as runtime
    calls=[]
    monkeypatch.setattr(runtime,'load_model',lambda:calls.append(1) or synthetic_model())
    runtime.model_resource.clear()
    a=runtime.model_resource(core.CHECKPOINT_HASH,1);b=runtime.model_resource(core.CHECKPOINT_HASH,1)
    assert a is b and len(calls)==1
    runtime.model_resource.clear()
