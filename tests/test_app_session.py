"""Synthetic workflow/evidence tests. No real model, Test pixels or evaluation."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_app import locked_app,ui_fixture,raster,synthetic_model
from src.app import core
from src.app.session import add_case,clear_history,comparison,near_threshold,NEAR_THRESHOLD_MARGIN,COMPARISON_NOTE
from src.app.dashboard import dashboard_evidence
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT


@pytest.mark.parametrize('score,expected',[
    (core.THRESHOLD,True),(core.THRESHOLD-.02,True),(core.THRESHOLD+.02,True),
    (core.THRESHOLD-.05,True),(core.THRESHOLD+.05,True),
    (np.nextafter(core.THRESHOLD-.05,-np.inf),False),
    (np.nextafter(core.THRESHOLD+.05,np.inf),False),(.1,False),(.9,False)])
def test_caution_never_changes_exact_decision(score,expected):
    before=core.score_wording(score)
    assert NEAR_THRESHOLD_MARGIN==.05
    assert near_threshold(score)==expected
    assert core.score_wording(score)==before
    assert ('above or equal' in before)==(score>=.3753475248813629)


def case(state,score=.4):
    upload=core.prepare_upload(raster(),'anonymous.png',load_config())
    upload.technical.update(PatientName='PRIVATE',PatientID='PRIVATE',path='PRIVATE')
    result=core.analyse(upload.canonical,synthetic_model(score))
    return add_case(state,upload,result,'2026-09-26T00:00:00+00:00'),upload,result


def test_history_allowlist_copies_minimal_views_without_persistence(tmp_path,monkeypatch):
    state={};monkeypatch.chdir(tmp_path)
    first,upload,result=case(state);second,_,_=case(state,.8)
    assert [r['case_number'] for r in state['case_history']]==[1,2]
    assert set(first)=={'case_number','timestamp','format','width','height','score','threshold','decision','near_threshold','quality_warning_count','gradcam_available'}
    assert first['near_threshold'] is True and second['near_threshold'] is False
    assert 'PRIVATE' not in json.dumps(state['case_history'])
    assert set(state['case_views'][1])=={'canonical','overlay','warnings','brightness','contrast','zero_heatmap'}
    np.testing.assert_array_equal(state['case_views'][1]['canonical'],upload.canonical)
    assert not np.shares_memory(state['case_views'][1]['canonical'],upload.canonical)
    assert list(tmp_path.iterdir())==[]
    clear_history(state)
    assert state=={}


def test_comparison_selection_difference_and_reuse():
    state={}
    with pytest.raises(ValueError,match='at least two'):comparison(state,1,2)
    case(state)
    with pytest.raises(ValueError,match='at least two'):comparison(state,1,2)
    case(state,.8)
    selected=comparison(state,2,1)
    assert selected['records'][0]['score']==.8
    assert selected['absolute_score_difference']==pytest.approx(.4)
    assert selected['views'][0] is state['case_views'][2]
    assert COMPARISON_NOTE=='Score differences between images do not indicate disease progression, treatment response or a change in clinical diagnosis.'
    with pytest.raises(ValueError,match='different'):comparison(state,1,1)
    with pytest.raises(ValueError,match='unavailable'):comparison(state,1,3)


def test_dashboard_reads_only_saved_aggregate_evidence(tmp_path,monkeypatch):
    from src.data.cbis_ddsm_metadata import write_csv
    base=tmp_path/'reports/evidence/model_evaluation';base.mkdir(parents=True)
    keys=('sensitivity','specificity','balanced_accuracy','roc_auc','pr_auc','brier_score')
    write_csv(base/'resnet50_validation_vs_test.csv',[
        dict(partition=part,threshold=core.THRESHOLD,false_negatives=3,false_negative_rate=.25,**{k:v for k in keys})
        for part,v in [('VALIDATION',.9),('OFFICIAL HOLDOUT TEST',.234)]])
    write_csv(base/'cohort_counts.csv',[dict(partition='test',images=19,participants=11)])
    write_csv(base/'test_confidence_intervals.csv',[dict(metric=k,point_estimate=.234,lower_95=.1,upper_95=.4) for k in keys[:4]])
    path=tmp_path/'reports/evidence/final_localisation/summary.json';path.parent.mkdir()
    path.write_text(json.dumps(dict(heatmap_threshold=.5,groups={'lesion/overall':dict(mean_iou=.012,mean_dice=.024,pointing_game_rate=.02,lesion_hit_rate=.3)})))
    from pathlib import Path
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    original_open=Path.open
    def guarded(path,*args,**kwargs):
        assert path in before,'Unexpected read outside synthetic aggregate evidence'
        assert not any(c in str(args[0] if args else kwargs.get('mode','r')) for c in ('w','a','+'))
        return original_open(path,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'open',guarded)
        evidence=dashboard_evidence(tmp_path)
    assert evidence['images']==19 and evidence['participants']==11
    assert evidence['metrics']['roc_auc']==.234 and evidence['false_negatives']==3
    assert evidence['confidence_intervals']['sensitivity']==(.1,.4)
    assert evidence['localisation']['mean_dice']==.024
    assert before=={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    write_csv(base/'test_confidence_intervals.csv',[dict(metric=k,point_estimate=.9,lower_95=.1,upper_95=.4) for k in keys[:4]])
    with pytest.raises(core.AppError,match='inconsistent'):dashboard_evidence(tmp_path)


def test_live_session_workflow_reuses_results_and_clears(monkeypatch,locked_app):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    import src.app.runtime as runtime
    ui_fixture(monkeypatch,locked_app)
    models=[synthetic_model(.37532),synthetic_model(.8)];active=[0]
    monkeypatch.setattr(st,'file_uploader',lambda *a,**k:SimpleNamespace(name='full.png',
        getvalue=lambda:raster(pixels=np.arange(1200).reshape(40,30)+active[0])))
    monkeypatch.setattr(runtime,'model_resource',lambda *a:models[active[0]])
    app=AppTest.from_file(str(PROJECT_ROOT/'src/app/streamlit_app.py'),default_timeout=20).run()
    assert not app.exception
    assert any('Analyse at least two' in m.value for m in app.markdown)
    assert next(m.value for m in app.metric if m.label=='ROC-AUC')=='0.123'
    assert any('overlap was low' in m.value for m in app.markdown)
    app.checkbox[0].check().run()
    def click(label):
        next(b for b in app.button if b.label==label).click().run()
        assert not app.exception
    click('Analyse full mammogram')
    assert len(app.session_state['case_history'])==1 and models[0].explainer.calls==1
    assert any(w.value=='Near research decision threshold' for w in app.warning)
    assert 'below' in app.session_state['result']['decision']
    baseline=app.session_state['result']['heatmap'].copy()
    original_overlay=app.session_state['result']['overlay'].copy()
    original_exports=app.session_state['exports']
    app.slider(key='attention_opacity').set_value(.8).run()
    for mode in ('Original','Heatmap','Overlay'):
        app.radio(key='attention_mode').set_value(mode).run()
        assert not app.exception
    assert models[0].explainer.calls==1 and app.session_state['result']['score']==.37532
    np.testing.assert_array_equal(baseline,app.session_state['result']['heatmap'])
    np.testing.assert_array_equal(original_overlay,app.session_state['result']['overlay'])
    assert original_exports==app.session_state['exports']
    assert next(m.value for m in app.metric if m.label=='Robustness flip-rate range')=='2.0%–21.0%'
    assert any('did not provide strong support' in m.value for m in app.markdown)
    active[0]=1;app.run()  # Replace upload; retain previous compact case, not its raw image.
    assert len(app.session_state['case_history'])==1 and 'result' not in app.session_state
    assert not app.checkbox[0].value and models[1].explainer.calls==0
    app.checkbox[0].check().run();click('Analyse full mammogram')
    assert len(app.session_state['case_history'])==2 and models[1].explainer.calls==1
    assert next(m.value for m in app.metric if m.label=='Absolute difference in model score')=='0.4247'
    assert any(COMPARISON_NOTE==info.value for info in app.info)
    app.selectbox(key='compare_left').select(2).run()
    app.selectbox(key='compare_right').select(1).run()
    assert not app.exception and sum(m.explainer.calls for m in models)==2
    assert len(app.get('download_button'))==2
    assert b'0.375320' not in app.session_state['exports'][1]  # Other case score is not exported.
    click('Clear session history')
    assert 'case_history' not in app.session_state and 'case_views' not in app.session_state
    assert 'result' in app.session_state  # History-only clear preserves current single analysis.
    click('Analyse full mammogram')
    monkeypatch.setattr(st,'file_uploader',lambda *a,**k:None)
    click('Clear uploaded image and results')
    assert all(k not in app.session_state for k in ('case_history','case_views','result','prepared','exports','timestamp','attention_mode','attention_opacity'))


def test_attention_rendering_does_not_mutate_scientific_outputs():
    from src.app.display import attention_view
    _,upload,result=case({})
    canonical=upload.canonical.copy();heatmap=result['heatmap'].copy();overlay=result['overlay'].copy();score=result['score']
    transparent=attention_view(canonical,result,'Overlay',0)
    opaque=attention_view(canonical,result,'Overlay',1)
    assert not np.array_equal(transparent,opaque)
    np.testing.assert_array_equal(transparent,np.repeat(canonical[...,None],3,axis=-1))
    np.testing.assert_array_equal(attention_view(canonical,result,'Original',.5),canonical)
    np.testing.assert_array_equal(attention_view(canonical,result,'Heatmap',.5),heatmap)
    np.testing.assert_array_equal(heatmap,result['heatmap']);np.testing.assert_array_equal(overlay,result['overlay'])
    np.testing.assert_array_equal(upload.canonical,canonical)
    assert result['score']==score and result['threshold']==core.THRESHOLD
    with pytest.raises(ValueError):attention_view(canonical,result,'Overlay',float('nan'))


def test_validation_dashboard_uses_saved_summaries_only(tmp_path,monkeypatch):
    from src.app.dashboard import validation_evidence
    from src.data.cbis_ddsm_metadata import write_csv
    from pathlib import Path
    base=tmp_path/'reports/evidence';(base/'robustness').mkdir(parents=True);(base/'gradcam_sanity').mkdir()
    write_csv(base/'robustness/robustness_summary.csv',[
        dict(perturbation=name,threshold=core.THRESHOLD,n=20,flip_rate=rate)
        for name,rate in [('original',0),('brightness_down',.05),('noise',.35)]])
    (base/'robustness/stability_summary.json').write_text(json.dumps(dict(any_flip_count=9)))
    path=base/'gradcam_sanity/summary.json'
    path.write_text(json.dumps(dict(partition='validation',status='COMPLETE',mean_paired_difference=-.1,high_larger_drop_fraction=.3)))
    original_open=Path.open;allowed={p for p in base.rglob('*') if p.is_file()}
    def guarded(p,*a,**kw):
        assert p in allowed and not any(x in str(a[0] if a else kw.get('mode','r')) for x in ('w','a','+'))
        return original_open(p,*a,**kw)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'open',guarded);data=validation_evidence(tmp_path)
    assert data['flip_min']==.05 and data['flip_max']==.35 and data['most_disruptive']=='noise'
    assert data['any_flip_count']==9 and data['images']==20
    assert 'did not provide strong support' in data['sanity_conclusion']
    path.write_text(json.dumps(dict(partition='test',status='COMPLETE')))
    with pytest.raises(core.AppError):validation_evidence(tmp_path)
