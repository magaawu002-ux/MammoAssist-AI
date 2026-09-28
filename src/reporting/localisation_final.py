"""Summarise a completed one-time Test run; no inference or protocol mutation."""

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.cbis_ddsm_metadata import write_csv
from src.data.provenance import sha256_file
from src.explainability.localisation import protocol_identity, require_final_lock, write_json
from src.preprocessing.manifests import local_path, read_rows
from src.reporting.localisation_validation import RUN, THRESHOLD, boolean, breakdowns, fixed_protocol
from src.utils.paths import PROJECT_ROOT


def completed_run(root):
    """Verify the immutable final receipt before opening metrics or predictions."""
    receipt=json.loads((root/'results/explainability/final_localisation_attempt.json').read_text())
    lock=json.loads((root/'results/explainability/localisation_lock.json').read_text())
    identity,coordinate=fixed_protocol(root)
    if receipt.get('status')!='COMPLETE' or lock['identity']!=identity:
        raise ValueError('Completed final receipt and unchanged lock required')
    directory=local_path(root,receipt['run'],('results','explainability','test'))
    if sha256_file(directory/'summary.json')!=receipt['summary_sha256']:
        raise ValueError('Final summary differs from receipt')
    summary=json.loads((directory/'summary.json').read_text())
    if summary['status']!='COMPLETE' or summary['partition']!='test' or summary['identity']!=identity:
        raise ValueError('Invalid completed Test run')
    for name,expected in summary['artifact_sha256'].items():
        if sha256_file(directory/name)!=expected: raise ValueError('Final artifact changed')
    try:
        require_final_lock(root)  # Read-only guard check; never call evaluate_localisation.
    except ValueError as exc:
        if 'already attempted' not in str(exc): raise
    else:
        raise ValueError('Repeated final attempt protection failed')
    return directory,summary,lock,coordinate


def enrich_test(metrics,predictions,manifest,mapping):
    """Join frozen Test probabilities without classifying any image again."""
    for rows in (metrics,predictions,manifest):
        for row in rows:
            if row.get('partition',row.get('final_partition'))!='test' or row.get('official_partition','test')!='test':
                raise ValueError('Only official Test rows permitted')
    images={r['canonical_image_id']:r for r in manifest}
    predictions_by_id={r['canonical_image_id']:r for r in predictions}
    if len(images)!=len(manifest) or len(predictions_by_id)!=len(predictions) or set(images)!=set(predictions_by_id):
        raise ValueError('Incomplete or duplicate frozen prediction cohort')
    for key,p in predictions_by_id.items():
        if any(p.get(k)!=v for k,v in images[key].items()): raise ValueError('Prediction/manifest mismatch')
        probability=float(p['predicted_probability']); label=int(p['true_label']); predicted=int(p['predicted_class'])
        category={(1,1):'true_positive',(1,0):'false_negative',(0,0):'true_negative',(0,1):'false_positive'}.get((label,predicted))
        if (float(p['threshold'])!=THRESHOLD or not np.isfinite(probability) or not 0<=probability<=1
                or predicted!=int(probability>=THRESHOLD) or label!=int(images[key]['binary_label'])
                or boolean(p['correct'])!=(predicted==label) or p['error_type']!=category):
            raise ValueError('Frozen classifier decision/threshold mismatch')
    eligible={}; excluded=set()
    for row in mapping:
        if not row['reference'].startswith('Mass-Test_'): raise ValueError('Non-Test mapping')
        if row['role']!='mask': continue
        key=(row['canonical_image_id'],row['record_id'])
        if key in eligible or key in excluded: raise ValueError('Duplicate mask')
        if row['alignment_status']=='verified' and row['alignment_method']=='locked_equal_grid_identity':
            eligible[key]=row
        elif (row['alignment_status']=='excluded' and row['alignment_method']=='EXCLUDE_UNPROVEN'
              and row['reason']=='exact_spatial_transform_unproven' and row['canonical_mask_pixels']==''):
            excluded.add(key)
        else: raise ValueError('Unproven geometry or transformed excluded mask')
    counts=Counter(k[0] for k in eligible); seen=set(); unions=set(); output=[]
    for row in metrics:
        image_id=row['canonical_image_id']; p=predictions_by_id[image_id]
        if image_id not in counts or float(row['threshold'])!=.5 or row['participant_id']!=p['participant_id']:
            raise ValueError('Invalid scored cohort or heatmap threshold')
        if row['scope']=='lesion':
            key=(image_id,row['lesion_id'])
            if key not in eligible or key in seen or key in excluded: raise ValueError('Excluded/duplicate lesion scored')
            if row['mask_sha256']!=eligible[key]['mask_sha256']: raise ValueError('Mask hash changed')
            seen.add(key)
        elif row['scope']=='image_union' and image_id not in unions:
            unions.add(image_id)
        else: raise ValueError('Duplicate or invalid metric scope')
        result=dict(row,view=p['view'],laterality=p['laterality'],image_pathology=p['pathology'],
                    true_label=int(p['true_label']),predicted_probability=float(p['predicted_probability']),
                    classifier_threshold=THRESHOLD,predicted_class=int(p['predicted_class']),
                    classification=p['error_type'],eligible_lesions_in_image=counts[image_id])
        for name in ('pointing_game','lesion_hit','zero_heatmap'): result[name]=boolean(row[name])
        for name in ('iou','dice','activation_mass_inside'):
            result[name]=float(row[name]) if row[name] not in ('',None) else None
            if result[name] is not None and (not np.isfinite(result[name]) or not 0<=result[name]<=1):
                raise ValueError('Invalid/nonfinite metric')
        if result['iou'] is None or result['dice'] is None or (result['activation_mass_inside'] is None)!=result['zero_heatmap']:
            raise ValueError('Invalid undefined metric')
        output.append(result)
    if seen!=set(eligible) or unions!=set(counts): raise ValueError('Incomplete scored cohort')
    return output,len(excluded)


def markdown_table(groups):
    lines=['| Group | n | Pointing hits/rate | Lesion hits/rate | IoU mean/median | Dice mean/median | Activation mass mean/median |',
           '| --- | ---: | --- | --- | --- | --- | --- |']
    def number(value): return 'undefined' if value is None else f'{value:.6f}'
    for key,g in groups.items():
        lines.append(f"| {key} | {g['n']} | {g['pointing_game_hits']} / {number(g['pointing_game_rate'])} | "
            f"{g['lesion_hit_hits']} / {number(g['lesion_hit_rate'])} | {number(g['mean_iou'])} / {number(g['median_iou'])} | "
            f"{number(g['mean_dice'])} / {number(g['median_dice'])} | {number(g['mean_activation_mass_inside'])} / {number(g['median_activation_mass_inside'])} |")
    return '\n'.join(lines)


def report(root):
    directory,summary,lock,coordinate=completed_run(root)
    predictions=root/RUN/'test/predictions.csv'
    # Snapshot hashes of existing classifier result files; only predictions are parsed.
    frozen={p.relative_to(root).as_posix():sha256_file(p) for p in sorted(predictions.parent.rglob('*')) if p.is_file()}
    rows,excluded=enrich_test(read_rows(directory/'localisation_metrics.csv'),read_rows(predictions),
        read_rows(root/'data/processed/manifests/test_processed.csv'),read_rows(directory/'mapping_audit.csv'))
    groups=breakdowns(rows); audit=summary['audit']
    if (len([r for r in rows if r['scope']=='lesion'])!=365 or excluded!=13
            or audit['eligible_masks']!=365 or audit['excluded_masks']!=13):
        raise ValueError('Locked Test eligibility inventory mismatch')
    output=directory/'report'; output.mkdir(exist_ok=False)
    for scope,name in (('lesion','per_lesion.csv'),('image_union','image_union.csv')):
        write_csv(output/name,[r for r in rows if r['scope']==scope])
    write_csv(output/'breakdowns.csv',[dict(group=k,**v) for k,v in groups.items()])
    image_rows=sorted((r for r in rows if r['scope']=='image_union'),key=lambda r:r['canonical_image_id'])
    figures=[]
    for row in image_rows[:lock['identity']['protocol']['overlay_limit']]:
        filename=row['canonical_image_id']+'.png'; shutil.copyfile(directory/filename,output/filename)
        figures.append(dict(row,filename=filename,sha256=sha256_file(output/filename),
                            selection='First canonical IDs; unchanged locked evaluator overlay_limit=6'))
    write_csv(output/'figures.csv',figures)
    details=dict(status='COMPLETE',partition='test',label='OFFICIAL TEST LOCALISATION',
        run=directory.relative_to(root).as_posix(),groups=groups,excluded_unequal_grid_masks=excluded,
        multiple_lesion_images=sum(r['eligible_lesions_in_image']>1 for r in image_rows),
        identity=summary['identity'],coordinate_rules=lock['coordinate_rules'],metric_definitions=lock['metric_definitions'],
        classifier_threshold=THRESHOLD,heatmap_threshold=.5,thresholds_prespecified=True,tuning=False,
        final_receipt_sha256=sha256_file(root/'results/explainability/final_localisation_attempt.json'),
        final_lock_sha256=sha256_file(root/'results/explainability/localisation_lock.json'),
        evaluation_summary_sha256=sha256_file(directory/'summary.json'),
        evaluation_artifact_sha256=summary['artifact_sha256'],classifier_results_sha256=frozen,
        reporter_sha256=sha256_file(Path(__file__)),aggregation_source_sha256=lock['reporting_source_sha256'],
        repeated_final_attempt_blocked=True,classifier_evaluation_rerun=False,additional_inference_for_reporting=False,
        figures=figures,figure_limitation='Category-based Validation figure generation is Validation-only and reruns inference. It was not extended to Test. Retained the six canonical-ID overlays generated by the locked evaluator, with probabilities/metrics in the figure index and captions.')
    write_json(output/'summary.json',details)
    validation=json.loads((root/lock['validation_run']/'report/summary.json').read_text())
    comparison={label+'/'+scope:source['groups'][scope+'/overall'] for label,source in
                (('VALIDATION',validation),('OFFICIAL TEST LOCALISATION',details)) for scope in ('lesion','image_union')}
    write_csv(output/'validation_vs_test.csv',[dict(cohort_scope=k,**v) for k,v in comparison.items()])
    (output/'validation_vs_test.md').write_text('# Frozen Validation vs official Test localisation\n\n'+markdown_table(comparison)+
        '\n\nSame fixed thresholds, metric definitions and identity/exclusion rules. Separate cohorts; descriptive comparison only. No Test-driven selection or tuning.\n',encoding='utf-8')
    lines=['# OFFICIAL TEST LOCALISATION','',markdown_table(groups),'',
           f"Zero heatmaps: {groups['image_union/overall']['zero_heatmaps']} images; {groups['lesion/overall']['zero_heatmaps']} lesion rows. Activation mass is undefined for zero maps; its defined denominator is retained in JSON/CSV.",
           '',f"Multiple-lesion images: {details['multiple_lesion_images']}. Individual lesions and eligible OR-unions are separate. Thirteen unequal-grid masks remain excluded.",
           '',f'Classifier threshold {THRESHOLD}; prespecified Grad-CAM threshold 0.5; conv5_block3_out. No tuning.',
           '', 'Descriptive only: correlated lesions/views/participants; no significance tests. Malignant-probability explanations are also reported for benign lesions. These overlap results do not establish clinical contour validity.',
           '',details['figure_limitation'],'']
    for r in figures:
        lines += [f"### {r['canonical_image_id']}", '',
            f"{r['pathology']} / {r['view']} / {r['laterality']}; {r['classification']}; saved P(malignant)={r['predicted_probability']:.6f}; classifier threshold={THRESHOLD}; heatmap threshold=0.5.",
            f"Image union: pointing={r['pointing_game']}, hit={r['lesion_hit']}, IoU={r['iou']:.6f}, Dice={r['dice']:.6f}, activation mass={r['activation_mass_inside']}. Green contour is the reference ROI.",
            '',f"![Full mammogram and malignant Grad-CAM with ROI contour]({r['filename']})",'']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    if frozen!={p:sha256_file(root/p) for p in frozen}: raise ValueError('Existing classifier results changed')
    completed_run(root)  # Recheck receipt, validation/final artifacts and repeat guard.
    write_json(output/'artifact_hashes.json',{p.name:sha256_file(p) for p in sorted(output.iterdir())})
    shutil.copytree(output,root/'reports/evidence/final_localisation')
    return details


if __name__=='__main__':
    result=report(PROJECT_ROOT)
    print(json.dumps({'status':result['status'],'run':result['run'],'overall':result['groups']['lesion/overall']}))
