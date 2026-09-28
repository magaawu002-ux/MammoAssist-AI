"""Validation-only descriptive reporting; never select or tune a protocol."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.cbis_ddsm_metadata import write_csv
from src.data.provenance import sha256_file
from src.explainability.coordinate_protocol import PROTOCOL, bound_hash, load_coordinate_protocol
from src.explainability.localisation import protocol_identity, write_json
from src.preprocessing.manifests import local_path, read_rows
from src.utils.paths import PROJECT_ROOT

RUN = 'results/experiments/resnet50/20260925T083032_895231Z'
CHECKPOINT = '0076686617cb29ad5e5682f3d1eff89fae9cad2cebffa86e2b8be2b537934384'
COORDINATES = '674f32236b55eae2d831509e5b6fb3206eb14c9a6a0a605876e0ea8d9977429e'
THRESHOLD = 0.3753475248813629
FIGURE_RULE = ('Highest/lowest image-union IoU, ties by canonical ID; first canonical ID '
               'for true-positive malignant, false-negative malignant, benign and multiple-lesion. '
               'Categories may select the same image. Extremes illustrate range, not typical performance.')


def fixed_protocol(root):
    identity = protocol_identity(root)
    selection = json.loads((root/'results/experiments/final_selection.json').read_text())
    coordinate = load_coordinate_protocol(root)
    if (identity['checkpoint'] != RUN+'/best_model.keras' or identity['checkpoint_sha256'] != CHECKPOINT
            or selection['threshold'] != THRESHOLD or identity['protocol']['heatmap_threshold'] != 0.5
            or identity['protocol']['target_layer'] != 'conv5_block3_out'
            or bound_hash(root/PROTOCOL) != COORDINATES or not coordinate):
        raise ValueError('Fixed checkpoint, thresholds, layer or alignment protocol changed')
    return identity, coordinate


def validation_rows(rows):
    for row in rows:
        partition = row.get('partition', row.get('final_partition'))
        if partition != 'validation' or row.get('official_partition', 'train') != 'train':
            raise ValueError('Only Validation rows are permitted')
    return rows


def boolean(value):
    if str(value) not in ('True','False'):
        raise ValueError('Invalid boolean metric')
    return str(value) == 'True'


def enrich(metrics, predictions, manifest, mapping):
    validation_rows(metrics); validation_rows(predictions); validation_rows(manifest)
    by_id = {r['canonical_image_id']: r for r in predictions}
    images = {r['canonical_image_id']: r for r in manifest}
    if len(by_id)!=len(predictions) or len(images)!=len(manifest) or set(by_id)!=set(images):
        raise ValueError('Duplicate or incomplete Validation prediction cohort')
    for key, p in by_id.items():
        image = images[key]
        if any(p.get(k)!=v for k,v in image.items()):
            raise ValueError('Existing predictions do not match Validation manifest')
        prob=float(p['predicted_probability']); label=int(p['true_label'])
        if (float(p['threshold'])!=THRESHOLD or not np.isfinite(prob) or not 0<=prob<=1
                or int(p['predicted_class'])!=int(prob>=THRESHOLD) or label!=int(image['binary_label'])
                or boolean(p['correct'])!=(int(p['predicted_class'])==label)):
            raise ValueError('Existing Validation prediction threshold/decision mismatch')
    eligible={}; excluded=set()
    for row in mapping:
        if not row['reference'].startswith('Mass-Training_'):
            raise ValueError('Test mapping rows are forbidden')
        if row['role']!='mask':
            continue
        key=(row['canonical_image_id'],row['record_id'])
        if row['alignment_status']=='verified' and row['alignment_method']=='locked_equal_grid_identity':
            if key in eligible: raise ValueError('Duplicate eligible lesion')
            eligible[key]=row
        elif row['alignment_status']=='excluded' and row['alignment_method']=='EXCLUDE_UNPROVEN':
            excluded.add(key)
        else:
            raise ValueError('Unproven mask cannot enter scoring')
    counts=Counter(key[0] for key in eligible)
    actual=set(); unions=set(); output=[]
    for row in metrics:
        key=row['canonical_image_id']; p=by_id[key]
        if key not in counts or float(row['threshold'])!=0.5:
            raise ValueError('Excluded image or changed heatmap threshold')
        if row['participant_id']!=p['participant_id']: raise ValueError('Participant mismatch')
        if row['scope']=='lesion':
            pair=(key,row['lesion_id'])
            if pair not in eligible or pair in excluded or pair in actual:
                raise ValueError('Excluded, duplicate or unmapped lesion entered scoring')
            if row['mask_sha256']!=eligible[pair]['mask_sha256']:
                raise ValueError('Scored mask hash changed')
            actual.add(pair)
        elif row['scope']=='image_union' and key not in unions:
            unions.add(key)
        else: raise ValueError('Unknown or duplicate metric scope')
        result=dict(row, view=p['view'],laterality=p['laterality'],image_pathology=p['pathology'],
                    true_label=int(p['true_label']),predicted_probability=float(p['predicted_probability']),
                    classifier_threshold=THRESHOLD,predicted_class=int(p['predicted_class']),
                    classification=p['error_type'],eligible_lesions_in_image=counts[key])
        for name in ('pointing_game','lesion_hit','zero_heatmap'): result[name]=boolean(row[name])
        for name in ('iou','dice','activation_mass_inside'):
            result[name]=float(row[name]) if row[name] not in ('',None) else None
            if result[name] is not None and (not np.isfinite(result[name]) or not 0<=result[name]<=1):
                raise ValueError('Nonfinite or invalid localisation metric')
        if result['iou'] is None or result['dice'] is None or (result['activation_mass_inside'] is None)!=result['zero_heatmap']:
            raise ValueError('Invalid missing metric')
        output.append(result)
    if actual!=set(eligible) or unions!=set(counts): raise ValueError('Incomplete eligible metric cohort')
    return output


def aggregate(rows):
    n=len(rows)
    result=dict(n=n,images=len({r['canonical_image_id'] for r in rows}),
                participants=len({r['participant_id'] for r in rows}))
    for key in ('pointing_game','lesion_hit'):
        result[key+'_hits']=sum(r[key] for r in rows)
        result[key+'_rate']=result[key+'_hits']/n if n else None
    for key in ('iou','dice','activation_mass_inside'):
        values=[r[key] for r in rows if r[key] is not None]
        result[key+'_defined_n']=len(values)
        result['mean_'+key]=float(np.mean(values)) if values else None
        result['median_'+key]=float(np.median(values)) if values else None
    result['zero_heatmaps']=sum(r['zero_heatmap'] for r in rows)
    return result


def breakdowns(rows):
    output={}
    for scope in ('lesion','image_union'):
        subset=[r for r in rows if r['scope']==scope]
        output[scope+'/overall']=aggregate(subset)
        for field in ('pathology','view','laterality'):
            for value in sorted({r[field] for r in subset}):
                output[f'{scope}/{field}/{value}']=aggregate([r for r in subset if r[field]==value])
        for value in ('true_positive','false_negative'):
            output[f'{scope}/malignant/{value}']=aggregate([r for r in subset if r['pathology']=='MALIGNANT' and r['classification']==value])
        output[scope+'/multiple_lesion']=aggregate([r for r in subset if r['eligible_lesions_in_image']>1])
    return output


def representatives(rows):
    images=sorted((r for r in rows if r['scope']=='image_union'),key=lambda r:r['canonical_image_id'])
    if not images: raise ValueError('No eligible images')
    choices=[('highest_iou',min(images,key=lambda r:(-r['iou'],r['canonical_image_id']))),
             ('lowest_iou',min(images,key=lambda r:(r['iou'],r['canonical_image_id'])))]
    for category,predicate in (
        ('correct_malignant',lambda r:r['classification']=='true_positive'),
        ('false_negative_malignant',lambda r:r['classification']=='false_negative'),
        ('benign',lambda r:r['true_label']==0),
        ('multiple_lesions',lambda r:r['eligible_lesions_in_image']>1)):
        match=next((r for r in images if predicate(r)),None)
        if match is not None: choices.append((category,match))
    return choices


def figures(root, directory, rows, manifest, mapping):
    import tensorflow as tf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.explainability.gradcam import ResNet50GradCAM
    from src.explainability.alignment import transform_mask,union_masks
    from src.explainability.roi_masks import decode_mask
    from src.explainability.localisation_metrics import localisation_metrics
    from src.explainability.overlay import overlay_heatmap
    from src.training.data import load_canonical
    from src.utils.config import load_config
    tf.keras.utils.set_random_seed(load_config(root/'config/project.yaml')['random_seed'])
    tf.config.experimental.enable_op_determinism()
    model=ResNet50GradCAM.from_checkpoint(root/RUN/'best_model.keras')
    images={r['canonical_image_id']:r for r in manifest}; index=[]
    for category,row in representatives(rows):
        image_id=row['canonical_image_id']; source=images[image_id]
        cache=local_path(root,source['cache_relative_path'],('data','processed','cache'))
        if sha256_file(cache)!=source['cache_sha256']: raise ValueError('Cache changed')
        geometry=json.loads(source['preprocessing_metadata_json']); masks=[]
        for lesion in mapping:
            if lesion['role']!='mask' or lesion['canonical_image_id']!=image_id: continue
            if lesion['alignment_status']!='verified' or lesion['alignment_method']!='locked_equal_grid_identity':
                raise ValueError('Unproven figure annotation')
            path=local_path(root,'data/raw/CBIS-DDSM/roi_cropped/'+lesion['path'],('data','raw','CBIS-DDSM','roi_cropped'))
            if sha256_file(path)!=lesion['mask_sha256']: raise ValueError('Mask changed')
            masks.append(transform_mask(decode_mask(path,lesion['SOPInstanceUID']).mask,geometry))
        union=union_masks(masks); image=load_canonical(cache,geometry['input_size']); result=model.explain(image)
        check=localisation_metrics(result.heatmap,union,0.5)
        for key in ('iou','dice','activation_mass_inside','pointing_game','lesion_hit'):
            if check[key] is None and row[key] is None: continue
            if check[key] is None or row[key] is None or not np.isclose(check[key],row[key],atol=1e-7,rtol=1e-6):
                raise ValueError('Figure regeneration differs from evaluated metrics')
        if not np.isclose(result.malignant_probability,row['predicted_probability'],atol=1e-5,rtol=1e-5):
            raise ValueError('Figure probability differs from existing Validation prediction')
        fig,axes=plt.subplots(1,3,figsize=(12,5))
        axes[0].imshow(image,cmap='gray',vmin=0,vmax=1)
        hm=axes[1].imshow(result.heatmap,cmap='jet',vmin=0,vmax=1)
        fig.colorbar(hm,ax=axes[1],fraction=.046,pad=.04)
        axes[2].imshow(overlay_heatmap(image,result.heatmap))
        for ax,title in zip(axes,('Full mammogram + ROI','Malignant Grad-CAM + ROI','Overlay + ROI (green)')):
            ax.contour(union.astype(float),levels=[.5],colors=['lime'],linewidths=.8)
            ax.set_title(title); ax.set_axis_off()
        mass='undefined' if row['activation_mass_inside'] is None else f"{row['activation_mass_inside']:.4f}"
        fig.suptitle(f"{category}: {image_id} | {row['pathology']} {row['view']} {row['laterality']}\n"
                     f"P(malignant)={row['predicted_probability']:.6f}; classifier threshold={THRESHOLD}; {row['classification']}\n"
                     f"Image union ({len(masks)} lesions): pointing={int(row['pointing_game'])}, hit={int(row['lesion_hit'])}, "
                     f"IoU={row['iou']:.4f}, Dice={row['dice']:.4f}, activation mass={mass}; heatmap threshold=0.5",fontsize=10)
        fig.tight_layout(rect=(0,0,1,.80)); name=category+'_'+image_id+'.png'
        fig.savefig(directory/name,dpi=160); plt.close(fig)
        index.append(dict(category=category,canonical_image_id=image_id,filename=name,
                          selection_rule=FIGURE_RULE,sha256=sha256_file(directory/name)))
    return index


def report(root, run):
    identity,coordinate=fixed_protocol(root)
    directory=local_path(root,run,('results','explainability','validation'))
    summary=json.loads((directory/'summary.json').read_text())
    if summary['status']!='COMPLETE' or summary['partition']!='validation' or summary['identity']!=identity:
        raise ValueError('Completed current-protocol Validation run required')
    for name,value in summary['artifact_sha256'].items():
        if sha256_file(directory/name)!=value: raise ValueError('Evaluation artifact changed')
    predictions_path=root/RUN/'validation/predictions.csv'
    manifest=validation_rows(read_rows(root/'data/processed/manifests/validation_processed.csv'))
    mapping=read_rows(directory/'mapping_audit.csv')
    rows=enrich(read_rows(directory/'localisation_metrics.csv'),read_rows(predictions_path),manifest,mapping)
    output=directory/'report'; output.mkdir(exist_ok=False)
    write_csv(output/'per_case_and_lesion.csv',rows)
    groups=breakdowns(rows)
    write_csv(output/'descriptive_summary.csv',[dict(group=k,**v) for k,v in groups.items()])
    index=figures(root,output,rows,manifest,mapping); write_csv(output/'figures.csv',index)
    details=dict(status='COMPLETE',partition='validation',validation_run=run,groups=groups,
        eligible_lesions=summary['audit']['eligible_masks'],excluded_lesions=summary['audit']['excluded_masks'],
        classifier_threshold=THRESHOLD,heatmap_threshold=0.5,heatmap_threshold_prespecified=True,
        threshold_optimisation_performed=False,test_localisation_accessed=False,
        identity=identity,coordinate_rule_version=coordinate['rule_version'],coordinate_rules=coordinate['rules'],
        existing_validation_predictions_sha256=sha256_file(predictions_path),
        reporting_source_sha256=sha256_file(Path(__file__)),figure_selection=FIGURE_RULE,figures=index,
        interpretation='Descriptive only; correlated lesions/views/participants. Malignant-probability Grad-CAM also shown for benign annotations. Undefined activation mass on zero maps is excluded from that mean/median and counted explicitly. No tuning or significance tests.')
    write_json(output/'summary.json',details)
    lines=['# Validation Grad-CAM localisation','',
        'Prespecified heatmap threshold **0.5**; no threshold search or tuning. Classifier threshold **'+str(THRESHOLD)+'**.',
        '249 eligible lesions; nine unequal-grid annotations excluded. Test localisation was not accessed.','',
        '| Group | n | Pointing hits/rate | Lesion hits/rate | IoU mean/median | Dice mean/median | Activation mass mean/median |',
        '| --- | ---: | --- | --- | --- | --- | --- |']
    def number(v): return 'undefined' if v is None else f'{v:.6f}'
    for group,g in groups.items():
        lines.append(f"| {group} | {g['n']} | {g['pointing_game_hits']} / {number(g['pointing_game_rate'])} | "
                     f"{g['lesion_hit_hits']} / {number(g['lesion_hit_rate'])} | {number(g['mean_iou'])} / {number(g['median_iou'])} | "
                     f"{number(g['mean_dice'])} / {number(g['median_dice'])} | {number(g['mean_activation_mass_inside'])} / {number(g['median_activation_mass_inside'])} |")
    lines+=['',details['interpretation'],'',FIGURE_RULE,'',
            'Figures regenerate only the selected Validation heatmaps and verify their metrics against the completed run. They do not alter the scored cohort or definitions.','',
            'Green contours are reference annotations, not clinically validated predictions. Canonical full mammograms are displayed at model resolution.','']
    lines += [f"- [{r['category']}]({r['filename']})" for r in index]
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Preserve the original completed evaluation summary before attaching reporting hashes.
    shutil.copyfile(directory/'summary.json',output/'evaluation_summary_before_reporting.json')
    summary['artifact_sha256'].update({'report/'+p.name:sha256_file(p) for p in sorted(output.iterdir())})
    summary['descriptive_report']='report/summary.json'
    fixed_protocol(root)
    write_json(directory/'summary.json',summary)
    destination=root/'reports/evidence/validation_localisation'
    shutil.copytree(output,destination)
    return details


def finalise(root,run):
    identity,coordinate=fixed_protocol(root)
    directory=local_path(root,run,('results','explainability','validation'))
    summary=json.loads((directory/'summary.json').read_text())
    if summary.get('status')!='COMPLETE' or summary.get('partition')!='validation' or summary.get('identity')!=identity:
        raise ValueError('Current completed Validation protocol required')
    if summary.get('descriptive_report')!='report/summary.json': raise ValueError('Validation report required')
    for name,expected in summary['artifact_sha256'].items():
        if sha256_file(directory/name)!=expected: raise ValueError('Validation result/report changed')
    lock=dict(status='LOCKED',identity=identity,validation_run=run,
        validation_summary_sha256=sha256_file(directory/'summary.json'),
        classifier_threshold=THRESHOLD,heatmap_threshold=0.5,target_layer='conv5_block3_out',
        coordinate_rule_version=coordinate['rule_version'],coordinate_rules=coordinate['rules'],
        validation_artifact_sha256=summary['artifact_sha256'],
        metric_definitions={'pointing_game':'First row-major global maximum inside mask; zero map is a miss.',
            'lesion_hit':'Any intersection of H >= 0.5 and reference mask.',
            'iou':'Intersection / union of H >= 0.5 and reference mask.',
            'dice':'2 * intersection / (predicted area + reference area).',
            'activation_mass_inside':'Sum H inside mask / sum H; undefined for zero heatmap.',
            'scopes':'Individual eligible lesions and per-image OR-union of eligible lesions.'},
        threshold_prespecified=True,tuning_after_validation=False,test_localisation_accessed=False,
        reporting_source_sha256=sha256_file(Path(__file__)))
    write_json(root/'results/explainability/localisation_lock.json',lock,exclusive=True)
    write_json(root/'reports/evidence/validation_localisation/final_protocol_lock.json',lock,exclusive=True)
    return lock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run'); parser.add_argument('--lock',action='store_true'); args=parser.parse_args()
    result=finalise(PROJECT_ROOT,args.run) if args.lock else report(PROJECT_ROOT,args.run)
    print(json.dumps({'status':result['status'],'run':args.run}))


if __name__=='__main__': main()
