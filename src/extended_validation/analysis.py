"""Fixed in-memory stresses and matched occlusion on Validation only."""

import argparse
from collections import Counter
from datetime import datetime,timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import cv2
import numpy as np
from PIL import Image

from src.app.core import contract,CHECKPOINT,THRESHOLD,TARGET
from src.data.cbis_ddsm_metadata import write_csv
from src.data.provenance import sha256_file
from src.evaluation.metrics import evaluate_metrics
from src.preprocessing.manifests import read_rows,local_path
from src.preprocessing.model_inputs import transfer_input
from src.utils.paths import PROJECT_ROOT

SEED=20260926
STRESSES=('original','brightness_down','brightness_up','contrast_down','contrast_up','noise','blur','jpeg')
PARAMETERS=dict(brightness_factors=[.9,1.1],contrast_factors=[.9,1.1],noise_sd=.01,
                blur_kernel=5,blur_sigma=.5,jpeg_quality=90,occlusion_size=32,occlusion_stride=16,
                foreground_threshold=.05,minimum_foreground_fraction=.5,matching_tolerance=.1,
                bootstrap_iterations=2000,bootstrap_confidence=.95,baseline_absolute_tolerance=1e-5)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def validate_rows(rows):
    if not rows or any(r.get('final_partition')!='validation' or r.get('official_partition')!='train' for r in rows):
        raise ValueError('Only official-Training Validation rows are permitted')
    if len({r['canonical_image_id'] for r in rows})!=len(rows): raise ValueError('Duplicate Validation images')
    return rows


def validation_images(root):
    """No partition argument: a Test manifest cannot be requested."""
    details=contract(root,verify_checkpoint=True)
    manifest=root/'data/processed/manifests/validation_processed.csv'
    rows=validate_rows(read_rows(manifest))
    if rows!=read_rows(root/'reports/evidence/preprocessing_validation_attempt.csv'):
        raise ValueError('Validation manifest changed')
    meta=json.loads((root/'reports/evidence/preprocessing_validation.json').read_text())
    if sha256_file(root/'data/splits/validation.csv')!=meta['split_sha256']['validation']:
        raise ValueError('Validation split changed')
    predictions=validate_rows(read_rows(root/Path(CHECKPOINT).parent/'validation/predictions.csv'))
    by_id={r['canonical_image_id']:r for r in predictions}
    if {r['canonical_image_id'] for r in rows}!=set(by_id): raise ValueError('Saved prediction cohort mismatch')
    images=[]; rows=sorted(rows,key=lambda r:r['canonical_image_id'])
    for row in rows:
        p=by_id[row['canonical_image_id']]
        if any(p.get(k)!=v for k,v in row.items()) or float(p['threshold'])!=THRESHOLD:
            raise ValueError('Frozen Validation prediction traceability/threshold changed')
        geometry=json.loads(row['preprocessing_metadata_json'])
        if geometry['clahe'] is not False or geometry['preprocessing_version']!='canonical_v1':
            raise ValueError('Canonical non-CLAHE input required')
        cache=local_path(root,row['cache_relative_path'],('data','processed','cache'))
        if sha256_file(cache)!=row['cache_sha256']: raise ValueError('Canonical cache changed')
        image=np.load(cache,allow_pickle=False)
        check_image(image); images.append(image)
    return rows,images,by_id,details


def check_image(image):
    if image.shape!=(224,224) or image.dtype!=np.float32 or not np.isfinite(image).all() or np.any((image<0)|(image>1)):
        raise ValueError('Expected finite canonical float32 image')


def perturb(image,name,image_id):
    check_image(image)
    if name not in STRESSES: raise ValueError('Unknown prespecified stress')
    out=image.copy();support=image>0
    if name.startswith('brightness'):
        out*=.9 if name.endswith('down') else 1.1
    elif name.startswith('contrast'):
        centre=float(image[support].mean()) if support.any() else 0
        out=(image-centre)*(.9 if name.endswith('down') else 1.1)+centre
    elif name=='noise':
        seed=int.from_bytes(hashlib.sha256(f'{SEED}:{image_id}:noise'.encode()).digest()[:8],'little')
        out+=np.random.default_rng(seed).normal(0,.01,image.shape).astype(np.float32)
    elif name=='blur': out=cv2.GaussianBlur(image,(5,5),.5,borderType=cv2.BORDER_REFLECT_101)
    elif name=='jpeg':
        buffer=BytesIO();Image.fromarray(np.uint8(np.rint(image*255))).save(buffer,format='JPEG',quality=90,subsampling=0)
        buffer.seek(0)
        with Image.open(buffer) as reconstructed: out=np.asarray(reconstructed,dtype=np.float32)/255
    out[~support]=0  # Preserve original zero background/padding for photometric stresses.
    return np.clip(out,0,1).astype(np.float32)


def robustness_metrics(labels,probabilities,baseline):
    metrics=evaluate_metrics(np.asarray(labels),np.asarray(probabilities),THRESHOLD)
    delta=np.asarray(probabilities)-np.asarray(baseline)
    metrics.update(mean_probability_change=float(delta.mean()),mean_absolute_probability_change=float(np.abs(delta).mean()),
        median_absolute_probability_change=float(np.median(np.abs(delta))),p95_absolute_probability_change=float(np.quantile(np.abs(delta),.95)),
        flip_count=int(np.sum((np.asarray(probabilities)>=THRESHOLD)!=(np.asarray(baseline)>=THRESHOLD))))
    metrics['flip_rate']=metrics['flip_count']/len(delta)
    return {k:v for k,v in metrics.items() if not isinstance(v,list)}


def occlusion_pair(image,heatmap):
    """Equal area, disjoint patches; row-major deterministic ties, no pathology input."""
    check_image(image);check_image(heatmap)
    if not heatmap.any(): return None,'zero_heatmap'
    patches=[]
    for y in range(0,193,16):
        for x in range(0,193,16):
            region=image[y:y+32,x:x+32]
            support=float((region>.05).mean())
            if support>=.5:
                patches.append(dict(y=y,x=x,attention=float(heatmap[y:y+32,x:x+32].mean()),
                                    brightness=float(region.mean()),foreground=support))
    if not patches:return None,'no_foreground_patch'
    high=min(patches,key=lambda p:(-p['attention'],p['y'],p['x']))
    candidates=[p for p in patches if (abs(p['y']-high['y'])>=32 or abs(p['x']-high['x'])>=32)
        and abs(p['foreground']-high['foreground'])<=.1 and abs(p['brightness']-high['brightness'])<=.1
        and p['attention']<high['attention']]
    if not candidates:return None,'no_disjoint_matched_lower_attention_patch'
    return (high,min(candidates,key=lambda p:(p['attention'],p['y'],p['x']))),'eligible'


def occlude(image,patch):
    out=image.copy();y,x=patch['y'],patch['x']
    out[y:y+32,x:x+32]=image[y:y+32,x:x+32].mean()
    return out


def paired_cluster_ci(rows):
    ids=sorted({r['participant_id'] for r in rows})
    if len(ids)<2:return None
    groups={p:[r for r in rows if r['participant_id']==p] for p in ids};rng=np.random.default_rng(SEED)
    values=[]
    for _ in range(2000):
        sample=[r for p in rng.choice(ids,len(ids),replace=True) for r in groups[p]]
        values.append([np.mean([r['drop_high'] for r in sample]),np.mean([r['drop_low'] for r in sample]),
                       np.mean([r['drop_high']-r['drop_low'] for r in sample]),np.mean([r['drop_high']>r['drop_low'] for r in sample])])
    return {k:dict(lower=float(np.quantile(np.array(values)[:,i],.025)),upper=float(np.quantile(np.array(values)[:,i],.975)))
            for i,k in enumerate(('mean_drop_high','mean_drop_low','mean_paired_difference','high_larger_drop_fraction'))}


def predict(model,images):
    outputs=[]
    for start in range(0,len(images),8):
        inputs=np.stack([transfer_input(image,'resnet50') for image in images[start:start+8]])
        outputs.extend(model(inputs,training=False).numpy().reshape(-1).tolist())
    result=np.array(outputs)
    if not np.isfinite(result).all() or np.any((result<0)|(result>1)):raise ValueError('Invalid prediction')
    return result


def run(root=PROJECT_ROOT):
    robustness=root/'reports/evidence/robustness';sanity=root/'reports/evidence/gradcam_sanity'
    if robustness.exists() or sanity.exists(): raise ValueError('Extended evidence already exists; do not overwrite or tune')
    rows,images,saved,details=validation_images(root)
    frozen={p:sha256_file(root/p) for p in ('results/experiments/final_selection.json','results/experiments/final_test_attempt.json',
        'results/explainability/localisation_lock.json','results/explainability/final_localisation_attempt.json',CHECKPOINT)}
    # Publish the fixed plan before obtaining any new predictions.
    config=dict(status='PRESPECIFIED_BEFORE_EXTENDED_INFERENCE',version='extended_validation_v1',partition='validation',
        exploratory_posthoc_extension=True,external_validation=False,seed=SEED,parameters=PARAMETERS,stresses=STRESSES,
        classifier_threshold=THRESHOLD,target_layer=TARGET,checkpoint_sha256=details['checkpoint_sha256'],
        baseline='fresh unperturbed Validation inference, checked against saved Validation probabilities within 1e-5',
        background='Original exact-zero pixels remain zero; contrast centred on nonzero mean; clipping to [0,1].',
        jpeg='8-bit quantisation plus JPEG quality 90 reconstruction, not a pure compression-only test.',
        translation='Omitted to avoid removing anatomy from already-filled canonical frames.',
        occlusion='All Validation images; 32x32 patches on stride16 grid; foreground=image>0.05 fraction>=0.5. Highest mean Grad-CAM patch vs disjoint lower-attention minimum, matched foreground fraction and mean brightness within0.1. Replace patch by its own mean; no ROI data.',
        baseline_consistency_tolerance=1e-5,tuning_allowed=False,no_test_pixels_or_predictions=True,
        source_sha256=sha256_file(Path(__file__)),git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        prescribed_at_utc=datetime.now(timezone.utc).isoformat(),frozen_before_sha256=frozen)
    save(robustness/'robustness_config.json',config);save(sanity/'protocol.json',config)
    import tensorflow as tf
    from src.explainability.gradcam import ResNet50GradCAM
    tf.keras.utils.set_random_seed(SEED);tf.config.experimental.enable_op_determinism()
    explainer=ResNet50GradCAM.from_checkpoint(root/CHECKPOINT)
    baseline=predict(explainer.model,images)
    expected=np.array([float(saved[r['canonical_image_id']]['predicted_probability']) for r in rows])
    if np.max(np.abs(baseline-expected))>1e-5:raise ValueError('Unperturbed Validation differs from frozen predictions')
    labels=[int(r['binary_label']) for r in rows];per_image=[];summary=[]
    for name in STRESSES:
        probability=baseline if name=='original' else predict(explainer.model,[perturb(im,name,r['canonical_image_id']) for r,im in zip(rows,images)])
        summary.append(dict(perturbation=name,**robustness_metrics(labels,probability,baseline)))
        for row,p,b in zip(rows,probability,baseline):
            per_image.append(dict(canonical_image_id=row['canonical_image_id'],participant_id=row['participant_id'],partition='validation',
                pathology=row['pathology'],true_label=int(row['binary_label']),perturbation=name,baseline_probability=float(b),
                probability=float(p),probability_change=float(p-b),absolute_probability_change=float(abs(p-b)),
                baseline_class=int(b>=THRESHOLD),perturbed_class=int(p>=THRESHOLD),flip=bool((p>=THRESHOLD)!=(b>=THRESHOLD)),threshold=THRESHOLD))
        print('Completed Validation robustness:',name,flush=True)
    write_csv(robustness/'robustness_per_image.csv',per_image);write_csv(robustness/'robustness_summary.csv',summary)
    cases=[]
    for index,(row,image,b) in enumerate(zip(rows,images,baseline)):
        explanation=explainer.explain(image)
        if abs(explanation.malignant_probability-b)>1e-5:raise ValueError('Grad-CAM baseline differs')
        pair,reason=occlusion_pair(image,explanation.heatmap)
        record=dict(canonical_image_id=row['canonical_image_id'],participant_id=row['participant_id'],partition='validation',
            true_label=int(row['binary_label']),baseline_probability=float(b),status=reason,drop_high=None,drop_low=None)
        if pair:
            high,low=pair;values=predict(explainer.model,[occlude(image,high),occlude(image,low)])
            record.update(high_patch_json=json.dumps(high),low_patch_json=json.dumps(low),probability_high=float(values[0]),
                          probability_low=float(values[1]),drop_high=float(b-values[0]),drop_low=float(b-values[1]),
                          high_larger_drop=bool(values[0]<values[1]))
        cases.append(record)
        if (index+1)%25==0:print('Completed Validation occlusion:',index+1,'/',len(rows),flush=True)
    write_csv(sanity/'per_case.csv',cases,sorted({k for r in cases for k in r}))
    eligible=[r for r in cases if r['status']=='eligible']
    if not eligible:raise ValueError('No matched occlusion cases; retain truthful exclusion evidence')
    result=dict(status='COMPLETE',partition='validation',method='fixed_matched_patch_mean_replacement',n_total=len(cases),n_eligible=len(eligible),
        participants=len({r['participant_id'] for r in eligible}),status_counts=dict(Counter(r['status'] for r in cases)),
        mean_drop_high=float(np.mean([r['drop_high'] for r in eligible])),mean_drop_low=float(np.mean([r['drop_low'] for r in eligible])),
        mean_paired_difference=float(np.mean([r['drop_high']-r['drop_low'] for r in eligible])),
        high_larger_drop_fraction=float(np.mean([r['high_larger_drop'] for r in eligible])),
        high_score_increase_count=sum(r['drop_high']<0 for r in eligible),low_score_increase_count=sum(r['drop_low']<0 for r in eligible),
        paired_participant_cluster_95ci=paired_cluster_ci(eligible),bootstrap_iterations=2000,
        caveat='Exploratory intervention sensitivity, not a proof of faithfulness or clinical localisation. Occlusion is distribution-shifting; matching is approximate. No tuning.',
        no_test_pixels_or_predictions=True)
    save(sanity/'summary.json',result)
    for row in rows:
        if sha256_file(root/row['cache_relative_path'])!=row['cache_sha256']:raise ValueError('Canonical cache changed')
    if frozen!={p:sha256_file(root/p) for p in frozen}:raise ValueError('Frozen scientific state changed')
    save(robustness/'execution.json',dict(status='COMPLETE',images=len(rows),participants=len({r['participant_id'] for r in rows}),
         baseline_max_absolute_difference=float(np.max(abs(baseline-expected))),python=platform.python_version(),tensorflow=tf.__version__,
         numpy=np.__version__,no_test_pixels_or_predictions=True,cache_and_frozen_state_unchanged=True,
         completed_at_utc=datetime.now(timezone.utc).isoformat()))
    render(root,per_image,summary,cases,result)


def render(root,per_image,summary,cases,sanity_summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    robust=root/'reports/evidence/robustness';sanity=root/'reports/evidence/gradcam_sanity'
    lines=['# Extended Validation-only robustness analysis','',
           'Exploratory in-memory stresses; not external validation. Threshold, model and cache remain unchanged. No Test inputs or post-hoc tuning.','',
           '| Stress | Flip rate | Mean absolute score change | ROC-AUC | Balanced accuracy | Sensitivity | Specificity | Brier |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in summary:lines.append('| '+r['perturbation']+' | '+' | '.join(f"{r[k]:.5f}" for k in ('flip_rate','mean_absolute_probability_change','roc_auc','balanced_accuracy','sensitivity','specificity','brier_score'))+' |')
    lines+=['','JPEG combines 8-bit quantisation and compression. Exact-zero background is preserved. Perturbations are controlled digital approximations, not validated scanner simulations. Shared participants and multiple stresses make rows dependent.']
    (robust/'robustness_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    stress=summary[1:];fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,key,title in zip(axes,('flip_rate','mean_absolute_probability_change'),('Classification flip rate','Mean absolute score change')):
        ax.barh([r['perturbation'] for r in stress],[r[key] for r in stress],color='#286a86');ax.set_title(title);ax.set_xlabel('Fraction / score units')
    fig.suptitle('Extended Validation-only robustness • fixed model and threshold');fig.tight_layout();fig.savefig(robust/'robustness_figure.png',dpi=180);plt.close(fig)
    ids=sorted({r['canonical_image_id'] for r in per_image});stability=[]
    for key in ids:
        group=[r for r in per_image if r['canonical_image_id']==key and r['perturbation']!='original']; first=group[0]
        category=('false_negative' if first['baseline_class']==0 else 'true_positive') if first['true_label']==1 else ('false_positive' if first['baseline_class'] else 'true_negative')
        stability.append(dict(canonical_image_id=key,participant_id=first['participant_id'],classification=category,baseline_probability=first['baseline_probability'],
            threshold_distance=abs(first['baseline_probability']-THRESHOLD),flip_count=sum(r['flip'] for r in group),
            any_flip=any(r['flip'] for r in group),mean_absolute_change=float(np.mean([r['absolute_probability_change'] for r in group])),
            max_absolute_change=max(r['absolute_probability_change'] for r in group)))
    write_csv(robust/'prediction_stability.csv',stability)
    groups={category:dict(n=len(g),any_flip_count=sum(r['any_flip'] for r in g),any_flip_rate=float(np.mean([r['any_flip'] for r in g])))
            for category in sorted({r['classification'] for r in stability}) if (g:=[r for r in stability if r['classification']==category])}
    save(robust/'stability_summary.json',dict(groups=groups,any_flip_count=sum(r['any_flip'] for r in stability),
        closest_to_threshold=sorted(stability,key=lambda r:(r['threshold_distance'],r['canonical_image_id']))[:10],
        caveat='Conditioning on baseline classification and threshold margin; descriptive association, not causal vulnerability or statistical independence.'))
    fig,axes=plt.subplots(1,2,figsize=(12,5));axes[0].scatter([r['threshold_distance'] for r in stability],[r['max_absolute_change'] for r in stability],c=[r['any_flip'] for r in stability],cmap='coolwarm',s=18)
    axes[0].set(xlabel='Distance to fixed threshold',ylabel='Maximum absolute score change',title='Red: flips under at least one stress')
    axes[1].boxplot([[r['probability_change'] for r in per_image if r['perturbation']==name] for name in STRESSES[1:]],tick_labels=STRESSES[1:],vert=False)
    axes[1].axvline(0,color='gray',lw=.7);axes[1].set(xlabel='Signed score change',title='Validation perturbation distributions')
    fig.tight_layout();fig.savefig(robust/'prediction_stability.png',dpi=180);plt.close(fig)
    eligible=[r for r in cases if r['status']=='eligible'];fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].scatter([r['drop_low'] for r in eligible],[r['drop_high'] for r in eligible],s=12,alpha=.7)
    lim=max(abs(v) for r in eligible for v in (r['drop_low'],r['drop_high']));axes[0].plot([-lim,lim],[-lim,lim],'k--',lw=.8)
    axes[0].set(xlabel='Low-attention occlusion score drop',ylabel='High-attention occlusion score drop')
    axes[1].hist([r['drop_high']-r['drop_low'] for r in eligible],bins=25,color='#286a86');axes[1].axvline(0,color='black',lw=.8)
    axes[1].set(xlabel='High minus low score drop',ylabel='Validation images');fig.suptitle('Exploratory matched-occlusion sanity check');fig.tight_layout();fig.savefig(sanity/'figure.png',dpi=180);plt.close(fig)
    (sanity/'summary.md').write_text('# Validation-only Grad-CAM sanity analysis\n\n'+json.dumps(sanity_summary,indent=2)+'\n\nPositive and negative drops are retained; negative drop means the score increased after occlusion. No ROI masks, Test inputs, model changes or tuning.\n',encoding='utf-8')


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
