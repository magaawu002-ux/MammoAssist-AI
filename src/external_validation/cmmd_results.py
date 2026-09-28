"""Report one completed locked CMMD prediction file; never load a model or DICOM pixels."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from src.evaluation.metrics import evaluate_metrics
from src.evaluation.bootstrap import bootstrap_intervals
from src.evaluation.calibration import calibration_curve
from . import cmmd_conversion as protocol
from .cmmd_image_audit import csv_read, csv_bytes
from .cmmd_metrics import report_metrics
from .cmmd_predict import PREDICTIONS, RECEIPT
from .cmmd_provenance import ROOT, RESULTS, EVIDENCE, THRESHOLD, CHECKPOINT_SHA256, sha256, write_new, isolated_path

PLAN = f'{EVIDENCE}/cmmd_analysis_plan.json'
COMPLETE = f'{RESULTS}/metrics/cmmd_external_validation_complete.json'
PROTOCOL_HASH = '4c031fcd627d499f1478445ba32d41dd14282114d2257ee54f646acd568d84bf'
COHORT_HASH = '26c1ed97b03eb704ab8158d4c831e1ebba8b3b919a077e38f8fd6a9ae4e0205d'
CBIS = 'results/experiments/resnet50/20260925T083032_895231Z/test'
METRICS = ['sensitivity','specificity','balanced_accuracy','roc_auc','pr_auc','brier_score','false_negative_rate']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_predictions(raw, cohort):
    """Require exact SOP membership, side/view/label identity and locked decisions."""
    expected={r['sop_uid']:r for r in cohort}
    if not raw or len(raw)!=len(cohort) or len(expected)!=len(cohort):raise ValueError('Prediction/cohort count mismatch')
    seen=set(); rows=[]
    for r in raw:
        uid=r['SOPInstanceUID']
        if uid not in expected or uid in seen:raise ValueError('Extra or duplicate prediction SOP')
        seen.add(uid); original=expected[uid]
        if any(str(r[k])!=str(original[k]) for k in ('patient_id','laterality','view','label')):
            raise ValueError('Patient/side/view/label differs from cohort')
        p=float(r['probability']);threshold=float(r['threshold']);label=int(r['label'])
        if not np.isfinite(p) or not 0<=p<=1:raise ValueError('Nonfinite/out-of-range score')
        if threshold!=THRESHOLD or int(r['predicted_label'])!=int(p>=THRESHOLD):raise ValueError('Threshold/decision changed')
        rows.append(dict(r,probability=p,threshold=threshold,label=label,predicted_label=int(r['predicted_label'])))
    if seen!=set(expected):raise ValueError('Missing cohort SOP')
    return rows


def calibration_summary(rows):
    y=np.array([r['label'] for r in rows]);p=np.array([r['probability'] for r in rows])
    result=calibration_curve(y,p,bins=10)
    result.update(binning='10 equal-width bins; final includes 1',recalibration_fitted=False,
        mean_score=float(p.mean()),malignant_prevalence=float(y.mean()),
        mean_score_minus_prevalence=float(p.mean()-y.mean()),
        caveat='Descriptive transfer only. Brier reflects discrimination, calibration and case mix; no recalibration fitted.')
    return result


def false_negatives(rows):
    malignant=[r for r in rows if r['label']==1]
    fn=[r for r in malignant if r['probability']<THRESHOLD]
    groups={}
    for field,values in [('view',['CC','MLO']),('laterality',['L','R'])]:
        groups[field]={}
        for value in values:
            subset=[r for r in malignant if r[field]==value]
            failures=[r for r in subset if r['probability']<THRESHOLD]
            groups[field][value]=dict(malignant_images=len(subset),false_negative_images=len(failures),
                false_negative_rate=len(failures)/len(subset) if subset else None)
    scores=np.array([r['probability'] for r in malignant])
    distribution={str(q):float(np.quantile(scores,q)) for q in [0,.05,.25,.5,.75,.95,1]} if len(scores) else {}
    return dict(threshold=THRESHOLD,false_negative_images=len(fn),unique_false_negative_patients=len({r['patient_id'] for r in fn}),
        malignant_images=len(malignant),false_negative_rate=len(fn)/len(malignant) if malignant else None,
        by_view=groups['view'],by_laterality=groups['laterality'],malignant_score_quantiles=distribution,
        malignant_mean_score=float(scores.mean()) if len(scores) else None,
        near_threshold={str(b):sum(THRESHOLD-b<=r['probability']<THRESHOLD for r in fn) for b in [.01,.025,.05,.10]},
        near_threshold_rule='Nested inclusive bands below the unchanged threshold; not threshold optimization',
        interpretation='Image-level rates conditional on malignancy; patients may contribute several correlated images')


def secondary_sides(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['patient_id'],r['laterality'])].append(r)
    sides=[]
    for (patient,side),group in sorted(groups.items()):
        if len(group)!=2 or {r['view'] for r in group}!={'CC','MLO'} or len({r['label'] for r in group})!=1:
            raise ValueError('Secondary summary requires one consistent CC/MLO pair')
        score=float(np.mean([r['probability'] for r in group]))
        sides.append(dict(participant_id=patient,laterality=side,label=group[0]['label'],mean_probability=score,
                          threshold=THRESHOLD,predicted_label=int(score>=THRESHOLD)))
    result=evaluate_metrics(np.array([r['label'] for r in sides]),np.array([r['mean_probability'] for r in sides]),THRESHOLD)
    result.update(analysis_unit='SECONDARY / SIDE-LEVEL',aggregation='Arithmetic mean of exactly CC and MLO',
                  patients=len({r['participant_id'] for r in sides}),benign_sides=sum(r['label']==0 for r in sides),
                  malignant_sides=sum(r['label']==1 for r in sides),false_negative_rate=result['fn']/sum(r['label']==1 for r in sides),
                  primary_analysis_unchanged=True,intervals='Not specified for this descriptive secondary analysis')
    return result,sides


def comparison_rows(cbis,external,cbis_ci,external_ci):
    if cbis['threshold']!=THRESHOLD or external['threshold']!=THRESHOLD:raise ValueError('Comparison threshold mismatch')
    result=[]
    for dataset,metrics,ci in [('CBIS-DDSM Official Test',cbis,cbis_ci),('CMMD External Validation',external,external_ci)]:
        for key in METRICS:
            interval=ci['intervals'].get(key,{})
            # FNR is the complement of sensitivity; reverse endpoints, no resampling.
            if key=='false_negative_rate':
                sensitivity=ci['intervals']['sensitivity']
                interval=dict(lower=1-sensitivity['upper'],upper=1-sensitivity['lower'])
            result.append(dict(dataset=dataset,metric=key,value=metrics[key],
                               ci95_lower=interval.get('lower',''),ci95_upper=interval.get('upper','')))
    return result


def figures(rows,metrics,calibration,comparison,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    outputs=[]
    def save(fig,name):
        for extension in ('png','svg'):
            relative=f'{RESULTS}/figures/{name}.{extension}'
            target=isolated_path(relative,root)
            if target.exists():raise FileExistsError(target)
            target.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(target,dpi=300,bbox_inches='tight')
            outputs.append(relative)
        plt.close(fig)
    y=np.array([r['label'] for r in rows]);p=np.array([r['probability'] for r in rows])
    fig,ax=plt.subplots(figsize=(5,4))
    matrix=np.array([[metrics['tn'],metrics['fp']],[metrics['fn'],metrics['tp']]])
    ax.imshow(matrix,cmap='Blues')
    for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>matrix.max()/2 else 'black',fontsize=14)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Benign','Malignant'],yticklabels=['Benign','Malignant'],
           xlabel='Predicted label',ylabel='True side-specific label',title='CMMD: image-level confusion matrix')
    fig.subplots_adjust(bottom=.24)
    fig.text(.5,.025,f'Frozen threshold = {THRESHOLD}',ha='center',fontsize=8)
    save(fig,'cmmd_confusion_matrix')
    fpr,tpr,_=roc_curve(y,p);fig,ax=plt.subplots(figsize=(5,4))
    ax.plot(fpr,tpr,label=f'ROC-AUC {metrics["roc_auc"]:.3f}');ax.plot([0,1],[0,1],'--',color='gray')
    ax.scatter([1-metrics['specificity']],[metrics['sensitivity']],label='Frozen-threshold operating point',s=25)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='False-positive rate',ylabel='Sensitivity',title='CMMD: ROC');ax.legend(fontsize=8)
    save(fig,'cmmd_roc')
    precision,recall,_=precision_recall_curve(y,p);fig,ax=plt.subplots(figsize=(5,4))
    ax.plot(recall,precision,label=f'Average precision {metrics["pr_auc"]:.3f}')
    ax.axhline(y.mean(),linestyle='--',color='gray',label=f'Malignant prevalence {y.mean():.3f}')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Recall',ylabel='Precision',title='CMMD: precision–recall');ax.legend(fontsize=8)
    save(fig,'cmmd_precision_recall')
    fig,(ax,hist)=plt.subplots(2,1,figsize=(5,5),gridspec_kw={'height_ratios':[3,1]},sharex=True)
    bins=[b for b in calibration['bins'] if b['count']]
    ax.plot([b['mean_probability'] for b in bins],[b['positive_fraction'] for b in bins],'o-')
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(ylim=(0,1),ylabel='Observed malignant fraction',title=f'CMMD: calibration (Brier {metrics["brier_score"]:.3f})')
    hist.bar([(b['lower']+b['upper'])/2 for b in calibration['bins']],[b['count'] for b in calibration['bins']],width=.09)
    hist.set(xlim=(0,1),xlabel='Uncalibrated frozen model score',ylabel='Images')
    save(fig,'cmmd_calibration')
    labels=['Sensitivity','Specificity','Balanced accuracy','ROC-AUC']
    keys=['sensitivity','specificity','balanced_accuracy','roc_auc']
    fig,ax=plt.subplots(figsize=(7,4))
    for offset,(dataset,color) in zip([-.12,.12],[('CBIS-DDSM Official Test','#386cb0'),('CMMD External Validation','#d95f02')]):
        values=[next(r for r in comparison if r['dataset']==dataset and r['metric']==key) for key in keys]
        estimates=np.array([r['value'] for r in values]);errors=np.array([[r['value']-r['ci95_lower'] for r in values],[r['ci95_upper']-r['value'] for r in values]])
        ax.errorbar(estimates,np.arange(4)+offset,xerr=errors,fmt='o',capsize=3,color=color,label=dataset)
    ax.set(yticks=np.arange(4),yticklabels=labels,xlim=(0,1),xlabel='Estimate and participant-cluster 95% CI',title='Frozen ResNet50: cross-dataset comparison');ax.invert_yaxis();ax.legend(fontsize=8)
    save(fig,'cbis_vs_cmmd_comparison')
    return outputs


def render_reports(metrics,ci,calibration,fn,cbis,comparison,secondary):
    intervals=ci['intervals']
    fmt=lambda x:f'{x:.4f}' if isinstance(x,(int,float)) else 'Undefined' if x is None else str(x)
    lines=['# CMMD external validation: completed','',
        f'Primary image-level analysis: {metrics["number_images"]} full mammograms, {metrics["number_breast_sides"]} eligible sides, {metrics["number_patients"]} participants.',
        f'Benign/malignant images: {metrics["benign_images"]}/{metrics["malignant_images"]}. Fixed threshold: {THRESHOLD}.',
        'Images are correlated within participants; uncertainty resamples participants with all their images. No CMMD fitting or selection occurred.','',
        '| Metric | Estimate | 95% participant-cluster CI |','| --- | ---: | --- |']
    for key in ['accuracy',*METRICS,'precision','f1']:
        interval=intervals.get(key)
        bounds=f'{interval["lower"]:.4f}–{interval["upper"]:.4f}' if interval else 'Not estimated'
        lines.append(f'| {key} | {fmt(metrics[key])} | {bounds} |')
    lines += ['',f'TN / FP / FN / TP: {metrics["tn"]} / {metrics["fp"]} / {metrics["fn"]} / {metrics["tp"]}.',
        'ROC-AUC is threshold-free; PR-AUC is average precision, not trapezoidal PR area. Bootstrap: 2,000 participant-cluster percentile replicates, seed 42, threshold fixed.',
        'Invalid replicates: '+', '.join(f'{k}={v["invalid_replicates"]}' for k,v in intervals.items())+'.','',
        '## Calibration transfer','',
        f'Mean score {calibration["mean_score"]:.4f}; malignant prevalence {calibration["malignant_prevalence"]:.4f}; mean-score minus prevalence {calibration["mean_score_minus_prevalence"]:+.4f}. '
        f'Brier score {calibration["brier_score"]:.4f}. Ten equal-width reliability bins and their image counts are saved.',
        'This observes transfer of the original scores. No calibration model was fitted. Brier depends on discrimination, calibration and prevalence; these selected clinical cohorts do not establish screening calibration.','',
        '## False negatives','',f'{fn["false_negative_images"]} malignant images were missed across {fn["unique_false_negative_patients"]} participants.',
        '| Group | Malignant images | FN images | FN rate |','| --- | ---: | ---: | ---: |']
    for group in [fn['by_view'],fn['by_laterality']]:
        for key,v in group.items():lines.append(f'| {key} | {v["malignant_images"]} | {v["false_negative_images"]} | {fmt(v["false_negative_rate"])} |')
    lines += ['',f'Malignant score quantiles: {fn["malignant_score_quantiles"]}.',
        'FN counts within nested distances below the fixed threshold: '+', '.join(f'{k}: {v}' for k,v in fn['near_threshold'].items())+'.',
        'These are descriptive; no threshold search, case removal or subtype/lesion localization was performed.','',
        '## Secondary side-level summary','',
        'Prespecified here before scores were viewed: arithmetic mean of exactly CC and MLO, then the same fixed threshold. This does not replace the primary image-level analysis.',
        f'{secondary["n"]} sides; TN/FP/FN/TP {secondary["tn"]}/{secondary["fp"]}/{secondary["fn"]}/{secondary["tp"]}; '
        f'sensitivity {secondary["sensitivity"]:.4f}, specificity {secondary["specificity"]:.4f}, balanced accuracy {secondary["balanced_accuracy"]:.4f}, ROC-AUC {secondary["roc_auc"]:.4f}.',
        'All basic secondary metrics are saved separately; secondary confidence intervals were not specified.','',
        '## Provenance','',
        'The completion receipt binds predictions, clinical/image cohort, protocol, checkpoint, preprocessing, analysis plan, code and outputs. '
        'The original STARTED and inference-complete receipts remain immutable. Predictions were generated once. '
        'No training, fine-tuning, recalibration, threshold change or CBIS-DDSM evaluation rerun occurred.','']
    comp=['# CBIS-DDSM Official Test versus CMMD external validation','',
        'Same frozen ResNet50, canonical preprocessing and threshold; primary unit is the full mammogram. '
        'CBIS-DDSM figures are read from existing official Test artifacts. No internal inference, evaluation or bootstrap was rerun.','',
        '| Cohort | Images | Participants | Benign / malignant images |','| --- | ---: | ---: | --- |',
        f'| CBIS-DDSM Official Test: MASS full mammograms | {cbis["n"]} | {cbis["patients"]} | {cbis["tn"]+cbis["fp"]} / {cbis["fn"]+cbis["tp"]} |',
        f'| CMMD external: locked MASS-only affected sides, CC/MLO | {metrics["n"]} | {metrics["number_patients"]} | {metrics["benign_images"]} / {metrics["malignant_images"]} |','',
        '| Metric | CBIS-DDSM | CMMD | CMMD minus CBIS |','| --- | --- | --- | ---: |']
    for key in METRICS:
        a,b=[next(r for r in comparison if r['dataset']==d and r['metric']==key) for d in ['CBIS-DDSM Official Test','CMMD External Validation']]
        display=lambda r:f'{r["value"]:.4f}'+(f' [{r["ci95_lower"]:.4f}, {r["ci95_upper"]:.4f}]' if r['ci95_lower']!='' else '')
        comp.append(f'| {key} | {display(a)} | {display(b)} | {b["value"]-a["value"]:+.4f} |')
    comp += ['', 'Brackets are within-cohort 95% participant-cluster percentile intervals; FNR intervals complement sensitivity. '
        'Intervals for AP and Brier were not prespecified. Differences are descriptive, not paired tests or confidence intervals for the between-cohort difference.',
        'Population, acquisition/encoding and case-mix/prevalence differences can contribute to observed domain shift. '
        'These data do not identify a causal mechanism. AP and Brier are prevalence-sensitive; point estimates alone do not establish that a dataset is easier or a model clinically adequate.',
        'CMMD findings must not select a new model, threshold or calibration. The comparison concerns generalisation of the already-frozen model.','']
    interpretation = [
        '## Interpretation','',
        ('Mixed generalisation, with important domain-shift limitations.'
         if metrics['sensitivity']>cbis['sensitivity'] and metrics['specificity']<cbis['specificity'] and metrics['roc_auc']<cbis['roc_auc']
         else 'Generalisation requires joint interpretation of discrimination, operating-point performance and calibration.'),
        f'The fixed operating point yields sensitivity {metrics["sensitivity"]:.4f} versus {cbis["sensitivity"]:.4f} internally, '
        f'but specificity {metrics["specificity"]:.4f} versus {cbis["specificity"]:.4f}; ROC-AUC is {metrics["roc_auc"]:.4f} versus {cbis["roc_auc"]:.4f}. '
        'These point-estimate changes do not establish a causal effect or a formal between-cohort significance test.',
        f'Average precision is {metrics["pr_auc"]:.4f} versus {cbis["pr_auc"]:.4f}, but malignant prevalence differs '
        f'({calibration["malignant_prevalence"]:.4f} versus {(cbis["tp"]+cbis["fn"])/cbis["n"]:.4f}); higher AP alone is not evidence of stronger transfer.',
        f'Brier score is {calibration["brier_score"]:.4f} versus {cbis["brier_score"]:.4f}. The reliability bins show '
        'whether low scores understate and high scores overstate the observed malignant fraction; mean calibration alone can conceal these opposing errors.',
        f'The CC/MLO false-negative rates are {fn["by_view"]["CC"]["false_negative_rate"]:.4f} and '
        f'{fn["by_view"]["MLO"]["false_negative_rate"]:.4f}. This descriptive view difference does not establish its mechanism. '
        'All images remain in the analysis, and neither threshold nor preprocessing was adjusted.',
        'The external discrimination and operating-point specificity limit a claim of reliable cross-dataset performance. '
        'Acquisition, population and case mix are plausible contributors, not demonstrated causes. '
        'These results do not establish clinical utility or screening-population calibration.','']
    return '\n'.join(lines+interpretation),'\n'.join(comp+interpretation)


def run(root=ROOT):
    root=Path(root).resolve();started=time.perf_counter()
    if (root/COMPLETE).exists():raise FileExistsError('External validation already complete; do not repeat')
    if sha256(root/protocol.PROTOCOL)!=PROTOCOL_HASH or sha256(root/protocol.COHORT)!=COHORT_HASH:
        raise ValueError('Frozen v2 protocol/cohort changed')
    lock=protocol.verify(root)
    attempt=read_json(root/RECEIPT)
    completion_path=f'{RESULTS}/predictions/cmmd_inference_complete_v2.json'
    completion=read_json(root/completion_path)
    if (completion['status']!='COMPLETE' or completion['images']!=2282 or completion['threshold']!=THRESHOLD or
        attempt['protocol_sha256']!=PROTOCOL_HASH or attempt['checkpoint_sha256']!=CHECKPOINT_SHA256 or attempt['threshold']!=THRESHOLD or
        completion['predictions_sha256']!=sha256(root/PREDICTIONS)):
        raise ValueError('Incomplete or inconsistent inference receipts')
    plan=read_json(root/PLAN)
    if (plan['classifier_threshold']!=THRESHOLD or plan['bootstrap']['seed']!=42 or plan['bootstrap']['iterations']!=2000 or
        plan['calibration']['bins']!=10 or plan['secondary']['threshold']!=THRESHOLD):raise ValueError('Analysis plan changed')
    rows=validate_predictions(csv_read(root/PREDICTIONS),csv_read(root/protocol.COHORT))
    if len(rows)!=2282:raise ValueError('Require all 2282 frozen images')
    metrics=report_metrics(rows)
    if (metrics['number_breast_sides'],metrics['number_patients'])!=(1141,1090):raise ValueError('Cohort totals changed')
    y=np.array([r['label'] for r in rows]);p=np.array([r['probability'] for r in rows])
    ci=bootstrap_intervals(y,p,[r['patient_id'] for r in rows],THRESHOLD,seed=42,iterations=2000)
    cal=calibration_summary(rows);fn=false_negatives(rows);secondary,side_rows=secondary_sides(rows)
    cbis=read_json(root/CBIS/'metrics.json');cbis_ci=read_json(root/CBIS/'bootstrap_intervals.json')
    cbis_rows=csv_read(root/CBIS/'predictions.csv')
    cbis['patients']=len({r['participant_id'] for r in cbis_rows});cbis['false_negative_rate']=cbis['fn']/(cbis['fn']+cbis['tp'])
    comparison=comparison_rows(cbis,metrics,cbis_ci,ci)
    inputs=[PREDICTIONS,RECEIPT,completion_path,protocol.PROTOCOL,protocol.COHORT,protocol.old.CLINICAL,PLAN,
            f'{CBIS}/metrics.json',f'{CBIS}/bootstrap_intervals.json',f'{CBIS}/predictions.csv',
            f'{EVIDENCE}/cmmd_inference_timing.json','src/external_validation/cmmd_results.py',
            'src/evaluation/metrics.py','src/evaluation/bootstrap.py','src/evaluation/calibration.py']
    inputs_hashes={path:sha256(root/path) for path in inputs}
    objects={'cmmd_metrics.json':metrics,'cmmd_bootstrap_ci.json':ci,'cmmd_calibration.json':cal,
             'cmmd_false_negative_summary.json':fn,'cmmd_secondary_side_metrics.json':secondary}
    outputs=[]
    for name,obj in objects.items():
        relative=f'{RESULTS}/metrics/{name}';write_new(relative,json.dumps(obj,indent=2,sort_keys=True)+'\n',root);outputs.append(relative)
    csv_outputs={f'{RESULTS}/metrics/cbis_vs_cmmd_comparison.csv':comparison,
                 f'{RESULTS}/metrics/cmmd_calibration_bins.csv':cal['bins'],
                 f'{RESULTS}/predictions/cmmd_secondary_side_scores.csv':side_rows}
    for path,values in csv_outputs.items():
        write_new(path,csv_bytes(values,list(values[0])).decode(),root);outputs.append(path)
    reports=render_reports(metrics,ci,cal,fn,cbis,comparison,secondary)
    for name,content in zip(['cmmd_external_results.md','cbis_vs_cmmd_comparison.md'],reports):
        relative=f'{EVIDENCE}/{name}';write_new(relative,content,root,evidence=True);outputs.append(relative)
    outputs+=figures(rows,metrics,cal,comparison,root)
    protocol.verify(root,check_pixels=False)
    if any(sha256(root/path)!=digest for path,digest in inputs_hashes.items()):raise ValueError('Reporting input changed')
    receipt=dict(status='EXTERNAL VALIDATION COMPLETE',training=False,fine_tuning=False,threshold_changed=False,
        calibration_fitting=False,cbis_evaluation_rerun=False,inference_invocations=1,
        checkpoint_sha256=CHECKPOINT_SHA256,image_cohort_sha256=COHORT_HASH,protocol_sha256=PROTOCOL_HASH,
        clinical_cohort_sha256=lock['clinical_cohort_sha256'],threshold=THRESHOLD,preprocessing_version='canonical_v1',
        preprocessing_hashes=lock['preprocessing_hashes'],inference_timing=read_json(root/f'{EVIDENCE}/cmmd_inference_timing.json'),
        reporting_elapsed_seconds=time.perf_counter()-started,analysis_unit='image',cluster_unit='participant',
        input_hashes=inputs_hashes,output_hashes={path:sha256(root/path) for path in outputs})
    write_new(COMPLETE,json.dumps(receipt,indent=2,sort_keys=True)+'\n',root)
    return dict(metrics=metrics,bootstrap=ci,calibration=cal,false_negatives=fn,secondary=secondary,cbis=cbis)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(),indent=2))
