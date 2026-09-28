"""Synthetic regression checks for the independent saved-score audit."""
import ast
from pathlib import Path

import numpy as np
import pytest

from src.integrity_audit import THRESHOLD, calibration, cluster_bootstrap, disagreement, join_unique, metrics


def test_confusion_and_exact_threshold():
    m=metrics([0,0,1,1],[.1,.9,.2,THRESHOLD])
    assert [m[k] for k in ('tn','fp','fn','tp')]==[1,1,1,1]
    assert m['sensitivity']==m['specificity']==m['balanced_accuracy']==.5
    assert m['brier_score']==np.mean((np.array([.1,.9,.2,THRESHOLD])-np.array([0,0,1,1]))**2)


@pytest.mark.parametrize('scores',[[float('nan')],[float('inf')],[-.1],[1.1]])
def test_invalid_scores_rejected(scores):
    with pytest.raises(ValueError):metrics([1],scores)


def test_join_uses_identity_not_position():
    manifest=[dict(id='a',label='0'),dict(id='b',label='1')]
    prediction=[dict(sop='b',label='1'),dict(sop='a',label='0')]
    assert join_unique(prediction,manifest,'sop','id',{'label':'label'})==manifest[::-1]


@pytest.mark.parametrize('prediction',[[dict(sop='a',label='0')]*2,[dict(sop='b',label='0')],[dict(sop='z',label='1')]])
def test_ambiguous_or_wrong_linkage_rejected(prediction):
    with pytest.raises(ValueError):join_unique(prediction,[dict(id='b',label='1')],'sop','id',{'label':'label'})


def test_calibration_endpoints_and_empty_bins():
    bins=calibration([0,1],[0.,1.])
    assert bins[0]['n']==bins[-1]['n']==1
    assert bins[1]['mean_score'] is None and sum(b['n'] for b in bins)==2


def test_cluster_bootstrap_matches_existing_helper():
    from src.evaluation.bootstrap import bootstrap_intervals
    y=np.array([0,0,1,1]);p=np.array([.2,.6,.4,.3]);ids=['a','a','b','b']
    expected=bootstrap_intervals(y,p,ids,THRESHOLD,seed=42,iterations=40)['intervals']
    actual=cluster_bootstrap(y,p,ids,iterations=40)
    assert not disagreement(actual,expected)
    assert actual['roc_auc']['invalid_replicates']>0


def test_audit_never_loads_or_calls_model():
    source=Path('src/integrity_audit.py').read_text()
    tree=ast.parse(source)
    calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
    assert not calls & {'predict','fit','evaluate_final','load_model','preprocess_dicom','explain'}
    assert 'tensorflow' not in source


def test_disagreement_detects_material_change():
    assert disagreement({'metrics':{'sensitivity':.8}},{'metrics':{'sensitivity':.7}})==['metrics.sensitivity']
