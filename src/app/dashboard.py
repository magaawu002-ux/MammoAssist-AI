"""Aggregate evidence only; no Test images, predictions or evaluation entrypoints."""
from math import isfinite
import json

from src.app.core import performance,THRESHOLD,AppError
from src.preprocessing.manifests import read_rows
from src.utils.paths import PROJECT_ROOT


def dashboard_evidence(root=PROJECT_ROOT):
    result=performance(root)
    base=root/'reports/evidence/model_evaluation'
    row=next(r for r in read_rows(base/'resnet50_validation_vs_test.csv') if r['partition']=='OFFICIAL HOLDOUT TEST')
    counts=next(r for r in read_rows(base/'cohort_counts.csv') if r['partition']=='test')
    result['participants']=int(counts['participants'])
    result['false_negatives']=int(row['false_negatives'])
    result['false_negative_rate']=float(row['false_negative_rate'])
    intervals={r['metric']:r for r in read_rows(base/'test_confidence_intervals.csv')}
    result['confidence_intervals']={}
    for key in ('sensitivity','specificity','balanced_accuracy','roc_auc'):
        ci=intervals[key];point=float(ci['point_estimate']);lo=float(ci['lower_95']);hi=float(ci['upper_95'])
        if not all(isfinite(v) and 0<=v<=1 for v in (point,lo,hi)) or lo>hi or abs(point-result['metrics'][key])>1e-12:
            raise AppError('Frozen confidence interval evidence is inconsistent.')
        result['confidence_intervals'][key]=(lo,hi)
    if float(row['threshold'])!=THRESHOLD or result['images']<=0 or not 0<=result['false_negative_rate']<=1:
        raise AppError('Frozen cohort evidence is inconsistent.')
    return result


def validation_evidence(root=PROJECT_ROOT):
    """Load small saved extension summaries, never per-image data or evaluators."""
    base=root/'reports/evidence'
    rows=read_rows(base/'robustness/robustness_summary.csv')
    stresses=[r for r in rows if r['perturbation']!='original']
    with (base/'robustness/stability_summary.json').open() as handle: stability=json.load(handle)
    with (base/'gradcam_sanity/summary.json').open() as handle: sanity=json.load(handle)
    if not stresses or sanity['partition']!='validation' or sanity['status']!='COMPLETE':
        raise AppError('Extended Validation evidence is unavailable.')
    for row in rows:
        rate=float(row['flip_rate'])
        if float(row['threshold'])!=THRESHOLD or not isfinite(rate) or not 0<=rate<=1:
            raise AppError('Extended Validation evidence is inconsistent.')
    worst=min(stresses,key=lambda r:(-float(r['flip_rate']),r['perturbation']))
    difference=float(sanity['mean_paired_difference']);fraction=float(sanity['high_larger_drop_fraction'])
    if not isfinite(difference) or not isfinite(fraction) or not 0<=fraction<=1:
        raise AppError('Sanity evidence is inconsistent.')
    return dict(flip_min=min(float(r['flip_rate']) for r in stresses),flip_max=float(worst['flip_rate']),
        most_disruptive=worst['perturbation'],any_flip_count=int(stability['any_flip_count']),images=int(rows[0]['n']),
        sanity_conclusion=('The Validation occlusion sanity check did not provide strong support for the expected Grad-CAM faithfulness behaviour.'
            if difference<=0 or fraction<=.5 else 'The saved occlusion comparison favoured high-attention masking under this specific intervention; it does not establish general faithfulness.'))
