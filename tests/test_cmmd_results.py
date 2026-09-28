"""Saved-score evaluation tests. No real predictions, model, DICOM or official Test evaluation."""
import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from src.external_validation import cmmd_results as r
from src.evaluation.bootstrap import bootstrap_intervals
from src.evaluation.metrics import evaluate_metrics


def predictions():
    rows=[]
    for i,(label,scores) in enumerate([(0,[.1,.5]),(1,[.2,.8])]):
        for j,(view,p) in enumerate(zip(['CC','MLO'],scores)):
            rows.append(dict(patient_id=f'P{i}',laterality='L',view=view,label=label,probability=p,
                predicted_label=int(p>=r.THRESHOLD),threshold=r.THRESHOLD,SOPInstanceUID=f'1.2.{i}.{j}'))
    return rows


def cohort(rows):
    return [dict(patient_id=x['patient_id'],laterality=x['laterality'],view=x['view'],label=x['label'],
                 sop_uid=x['SOPInstanceUID']) for x in rows]


def test_complete_identity_and_locked_decisions():
    rows=predictions()
    assert r.validate_predictions(rows,cohort(rows))==rows
    result=r.report_metrics(rows)
    assert [result[k] for k in ('tn','fp','fn','tp')]==[1,1,1,1]
    assert result['number_images']==4 and result['number_patients']==2


@pytest.mark.parametrize('field,value',[('patient_id','extra'),('laterality','R'),('view','UNKNOWN'),('label',1),
    ('threshold',.5),('probability',float('nan')),('probability',float('inf')),('probability',1.1),('predicted_label',1),
    ('SOPInstanceUID','9.9')])
def test_bad_prediction_rejected(field,value):
    rows=predictions();locked=cohort(rows);rows[0][field]=value
    with pytest.raises(ValueError):r.validate_predictions(rows,locked)


def test_missing_duplicate_and_extra_scores_rejected():
    rows=predictions();locked=cohort(rows)
    for bad in [rows[:-1],rows+rows[:1],[rows[0],rows[0],*rows[2:]]]:
        with pytest.raises(ValueError):r.validate_predictions(bad,locked)


def test_calibration_bins_cover_endpoints_and_empty_bins_without_fitting():
    rows=predictions();rows[0]['probability']=0.;rows[-1]['probability']=1.
    cal=r.calibration_summary(rows)
    assert len(cal['bins'])==10 and sum(b['count'] for b in cal['bins'])==4
    assert cal['bins'][0]['count']==1 and cal['bins'][-1]['count']==1
    assert cal['recalibration_fitted'] is False
    assert cal['brier_score']==pytest.approx(np.mean([(x['probability']-x['label'])**2 for x in rows]))
    assert any(b['mean_probability'] is None for b in cal['bins'])


def test_fn_descriptions_and_nested_threshold_bands():
    rows=predictions()
    rows[2]['probability']=r.THRESHOLD-.01
    fn=r.false_negatives(rows)
    assert fn['false_negative_images']==1 and fn['unique_false_negative_patients']==1
    assert fn['by_view']['CC']['false_negative_rate']==1
    assert fn['near_threshold']=={'0.01':1,'0.025':1,'0.05':1,'0.1':1}
    rows[2]['probability']=r.THRESHOLD
    assert r.false_negatives(rows)['false_negative_images']==0


def test_cluster_bootstrap_same_helper_fixed_seed_threshold_and_invalid_counts():
    rows=predictions();y=np.array([x['label'] for x in rows]);p=np.array([x['probability'] for x in rows]);ids=[x['patient_id'] for x in rows]
    a=bootstrap_intervals(y,p,ids,r.THRESHOLD,seed=42,iterations=30)
    assert a==bootstrap_intervals(y,p,ids,r.THRESHOLD,seed=42,iterations=30)
    assert a['threshold_fixed']==r.THRESHOLD
    for ci in a['intervals'].values():assert ci['valid_replicates']+ci['invalid_replicates']==30
    assert a['intervals']['roc_auc']['invalid_replicates']>0


def test_cluster_resampling_keeps_both_views_together(monkeypatch):
    import src.evaluation.bootstrap as b
    observed=[]
    real=b.evaluate_metrics
    def capture(y,p,threshold):
        observed.append(tuple(p));return real(y,p,threshold)
    monkeypatch.setattr(b,'evaluate_metrics',capture)
    b.bootstrap_intervals(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]),['A','A','B','B'],r.THRESHOLD,seed=42,iterations=10)
    for sample in observed[1:]:
        assert sample.count(.1)==sample.count(.2) and sample.count(.8)==sample.count(.9)


def test_secondary_mean_is_fixed_and_primary_unchanged():
    rows=predictions();before=copy.deepcopy(rows)
    m,sides=r.secondary_sides(rows)
    assert [x['mean_probability'] for x in sides]==pytest.approx([.3,.5])
    assert m['threshold']==r.THRESHOLD and m['analysis_unit']=='SECONDARY / SIDE-LEVEL'
    assert rows==before and m['tn']==m['tp']==1
    with pytest.raises(ValueError):r.secondary_sides(rows[:-1])


def test_comparison_reports_preserve_saved_estimates_and_ci():
    rows=predictions();m=r.report_metrics(rows);cbis=dict(m,patients=2)
    ci=bootstrap_intervals(np.array([x['label'] for x in rows]),np.array([x['probability'] for x in rows]),
        [x['patient_id'] for x in rows],r.THRESHOLD,seed=42,iterations=30)
    comp=r.comparison_rows(cbis,m,ci,ci)
    assert len(comp)==14
    fn=next(x for x in comp if x['metric']=='false_negative_rate')
    assert fn['ci95_lower']==1-ci['intervals']['sensitivity']['upper']
    primary,comparison=r.render_reports(m,ci,r.calibration_summary(rows),r.false_negatives(rows),cbis,comp,r.secondary_sides(rows)[0])
    assert 'Secondary side-level' in primary and 'No CMMD fitting' in primary
    assert 'no' in comparison.lower() and 'descriptive' in comparison


def test_report_module_cannot_run_prediction_or_training():
    tree=ast.parse(Path(r.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            assert node.module not in {'tensorflow','tensorflow.keras.models','src.evaluation.evaluate','src.training.train'}
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in {'execute','fit','predict','load_model','train_on_batch','compile'}


def test_completed_evaluation_refuses_repeat_before_other_io(tmp_path):
    p=tmp_path/r.COMPLETE;p.parent.mkdir(parents=True);p.write_text('{}')
    with pytest.raises(FileExistsError,match='already complete'):r.run(tmp_path)


def test_all_report_figures_are_isolated_and_do_not_overwrite(tmp_path):
    rows=predictions();m=r.report_metrics(rows)
    ci=bootstrap_intervals(np.array([x['label'] for x in rows]),np.array([x['probability'] for x in rows]),
        [x['patient_id'] for x in rows],r.THRESHOLD,seed=42,iterations=30)
    comparison=r.comparison_rows(m,m,ci,ci)
    outputs=r.figures(rows,m,r.calibration_summary(rows),comparison,tmp_path)
    assert len(outputs)==10 and all(p.startswith(r.RESULTS+'/figures/') for p in outputs)
    assert all((tmp_path/p).stat().st_size>100 for p in outputs)
    with pytest.raises(FileExistsError):r.figures(rows,m,r.calibration_summary(rows),comparison,tmp_path)
