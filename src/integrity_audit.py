"""Read-only saved-data checks. Never imports a model or an inference entrypoint.

Only the final_integrity_audit evidence directory is writable by this command.
Metrics are independently recomputed; existing result artifacts are not edited.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pydicom
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = 'reports/evidence/final_integrity_audit'
THRESHOLD = 0.3753475248813629
RUN = 'results/experiments/resnet50/20260925T083032_895231Z'
EXT = 'results/external_validation/CMMD'
EVIDENCE = 'reports/evidence/external_validation/CMMD'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def metrics(y, p, threshold=THRESHOLD):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if y.shape != p.shape or not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid labels/scores')
    pred = p >= threshold
    tn, fp, fn, tp = [int(v.sum()) for v in ((y == 0) & ~pred, (y == 0) & pred, (y == 1) & ~pred, (y == 1) & pred)]
    div = lambda a, b: a / b if b else None
    sens, spec = div(tp, tp+fn), div(tn, tn+fp)
    return dict(n=len(y), tn=tn, fp=fp, fn=fn, tp=tp, threshold=threshold,
                benign=tn+fp, malignant=tp+fn, accuracy=(tn+tp)/len(y), sensitivity=sens,
                specificity=spec, precision=div(tp, tp+fp), f1=div(2*tp, 2*tp+fp+fn),
                balanced_accuracy=(sens+spec)/2 if sens is not None and spec is not None else None,
                roc_auc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                pr_auc=float(average_precision_score(y, p)) if y.sum() else None,
                brier_score=float(np.mean((p-y)**2)), false_negative_rate=div(fn, tp+fn))


def join_unique(predictions, manifest, prediction_key, manifest_key, fields):
    """Strict set join: shuffled rows are safe; duplicates/missing identities fail."""
    index = {r[manifest_key]: r for r in manifest}
    ids = [r[prediction_key] for r in predictions]
    if len(index) != len(manifest) or len(set(ids)) != len(ids) or set(ids) != set(index):
        raise ValueError('Duplicate or unequal identity sets')
    for row in predictions:
        source = index[row[prediction_key]]
        if any(str(row[a]) != str(source[b]) for a, b in fields.items()):
            raise ValueError('Prediction linkage mismatch')
    return [index[r[prediction_key]] for r in predictions]


def cluster_bootstrap(y, p, ids, iterations=2000, seed=42):
    y, p, ids = np.array(y), np.array(p), np.array(ids)
    unique = np.unique(ids)
    groups = {pid: np.where(ids == pid)[0] for pid in unique}
    rng = np.random.default_rng(seed)
    values = {k: [] for k in ['sensitivity', 'specificity', 'balanced_accuracy', 'roc_auc']}
    for _ in range(iterations):
        index = np.concatenate([groups[pid] for pid in rng.choice(unique, len(unique), replace=True)])
        m = metrics(y[index], p[index])
        for k in values:
            if m[k] is not None:
                values[k].append(m[k])
    return {k: dict(lower=float(np.quantile(v, (1-.95)/2)) if len(v)>1 else None,
                    upper=float(np.quantile(v, 1-(1-.95)/2)) if len(v)>1 else None,
                    valid_replicates=len(v), invalid_replicates=iterations-len(v)) for k, v in values.items()}


def calibration(y, p):
    y, p = np.array(y), np.array(p)
    edges = np.linspace(0, 1, 11)
    result = []
    for i in range(10):
        use = (p >= edges[i]) & (p <= edges[i+1] if i == 9 else p < edges[i+1])
        result.append(dict(lower=float(edges[i]), upper=float(edges[i+1]), n=int(use.sum()),
                           mean_score=float(p[use].mean()) if use.any() else None,
                           malignant_fraction=float(y[use].mean()) if use.any() else None))
    return result


def disagreement(a, b, prefix=''):
    failures = []
    for k, value in a.items():
        if k not in b:
            continue
        expected = b[k]
        if isinstance(value, dict) and isinstance(expected, dict):
            failures += disagreement(value, expected, prefix+k+'.')
        elif value is None or expected is None:
            if value != expected: failures.append(prefix+k)
        elif isinstance(value, (int, float)) and isinstance(expected, (int, float)):
            if not np.isclose(value, expected, atol=1e-12, rtol=1e-12): failures.append(prefix+k)
        elif value != expected:
            failures.append(prefix+k)
    return failures


def run(root=ROOT):
    root = Path(root)
    output = root/OUT
    output.mkdir(parents=True, exist_ok=True)
    evidence_hashes = {}
    def read(path):
        evidence_hashes[str(path).replace('\\', '/')] = digest(root/path)
        return json.loads((root/path).read_text(encoding='utf-8-sig'))
    def csv_read(path):
        evidence_hashes[str(path).replace('\\', '/')] = digest(root/path)
        return rows(root/path)
    result = dict(inference_executed=False, frozen_outputs_modified=False, findings=[])
    splits = {part: csv_read(f'data/splits/{part}.csv') for part in ('train','validation','test')}
    processed = {part: csv_read(f'data/processed/manifests/{part}_processed.csv') for part in splits}
    from src.data.cbis_ddsm_metadata import read_metadata, aggregate_images
    source_rows = []
    for part in ('train','test'):
        path = f'data/raw/CBIS-DDSM/metadata/mass_case_description_{part}_set.csv'
        evidence_hashes[path] = digest(root/path)
        records, issues, _ = read_metadata(root/path, part, root)
        assert not issues, issues
        source_rows.extend(records)
    images, issues = aggregate_images(source_rows)
    assert not issues, issues
    originals = {r['canonical_image_id']: r for r in images}
    identities = defaultdict(lambda: defaultdict(set))
    counts, headers = {}, []
    for part, cohort in processed.items():
        join_unique(cohort, splits[part], 'canonical_image_id','canonical_image_id', {k:k for k in splits[part][0]})
        assert len(cohort) == len(splits[part])
        for r in cohort:
            original = originals[r['canonical_image_id']]
            for k in ('participant_id','official_partition','laterality','view','full_mammogram_reference','binary_label','lesion_row_count'):
                assert str(r[k]) == str(original[k]), (r['canonical_image_id'], k)
            assert r['official_partition'] == ('test' if part == 'test' else 'train')
            assert r['participant_id'] == r['original_participant_id']
            assert r['mapping_status'] == 'uniquely_mapped'
            meta = json.loads(r['preprocessing_metadata_json'])
            assert meta['clahe'] is False and meta['reorientation']=='none'
            assert meta['preprocessing_version']=='canonical_v1' and meta['input_size']==224
            ds = pydicom.dcmread(root/r['dicom_relative_path'], stop_before_pixels=True)
            for tag in ('SOPInstanceUID','StudyInstanceUID','SeriesInstanceUID'):
                assert str(getattr(ds,tag)) == meta[tag]
            assert str(ds.SeriesInstanceUID) == r['full_mammogram_reference'].split('/')[2]
            assert 'full' in str(getattr(ds,'SeriesDescription','')).lower()
            array = np.load(root/r['cache_relative_path'], allow_pickle=False)
            assert array.shape==(224,224) and array.dtype==np.float32 and np.isfinite(array).all()
            assert array.min()>=0 and array.max()<=1
            assert digest(root/r['cache_relative_path']) == r['cache_sha256']
            for k,v in dict(participant_id=r['participant_id'], canonical_image_id=r['canonical_image_id'],
                            file=r['dicom_relative_path'], sop=meta['SOPInstanceUID'],
                            cache=r['cache_sha256'], canonical_pixels=hashlib.sha256(array.tobytes()).hexdigest()).items():
                identities[k][v].add(part)
            headers.append(dict(partition=part, view=r['view'], photometric=meta['photometric_interpretation'],
                                bits=str(ds.BitsStored), rows=int(ds.Rows), columns=int(ds.Columns)))
        counts[part] = dict(images=len(cohort), participants=len({r['participant_id'] for r in cohort}),
                            benign=sum(r['binary_label']=='0' for r in cohort),malignant=sum(r['binary_label']=='1' for r in cohort),
                            mixed_pathology=sum(r['mixed_pathology']=='True' for r in cohort))
        print('Checked saved canonical arrays and DICOM headers:',part,flush=True)
    overlaps = {k:{v:sorted(parts) for v,parts in groups.items() if len(parts)>1} for k,groups in identities.items()}
    result['cbis_data'] = dict(counts=counts,source_rows=len(source_rows),unique_full_mammograms=len(images),
                              collapsed_rows=len(source_rows)-len(images),cross_partition_overlaps=overlaps,
                              canonical_exact_duplicate_detection=True,raw_pixel_equivalence_not_exhaustively_tested=True)
    if any(overlaps.values()): result['findings'].append('CRITICAL cross-partition identity overlap')
    comparison = []
    for path in sorted((root/'results/experiments').glob('*/*/validation/metrics.json')):
        rel=path.relative_to(root).as_posix();base=path.parent.parent.relative_to(root).as_posix()
        saved=read(rel);pred=csv_read(base+'/validation/predictions.csv');meta=read(base+'/run_metadata.json')
        join_unique(pred,processed['validation'],'canonical_image_id','canonical_image_id',
                    {k:k for k in ('participant_id','binary_label','view','laterality')})
        for part,h in meta['processed_manifest_sha256'].items(): assert digest(root/f'data/processed/manifests/{part}_processed.csv')==h
        for part,h in meta['split_hashes'].items(): assert digest(root/f'data/splits/{part}.csv')==h
        m=metrics([int(r['true_label']) for r in pred],[float(r['predicted_probability']) for r in pred],saved['threshold'])
        comparison.append(dict(model=meta['model'],run=base,metrics=m,metric_mismatches=disagreement(m,saved)))
    result['model_comparison']=comparison
    def prediction_audit(pred, label, score, participant, saved):
        y=[int(r[label]) for r in pred];p=[float(r[score]) for r in pred]
        assert all(float(r['threshold'])==THRESHOLD for r in pred)
        m=metrics(y,p)
        fn=[r for r in pred if int(r[label])==1 and float(r[score])<THRESHOLD]
        return dict(metrics=m,metric_mismatches=disagreement(m,saved),patients=len({r[participant] for r in pred}),
                    false_negative_patients=len({r[participant] for r in fn}),
                    false_negative_bands={str(b):sum(THRESHOLD-float(r[score])<=b for r in fn) for b in [.01,.025,.05,.10]},
                    by_view={v:metrics([int(r[label]) for r in pred if r['view']==v],[float(r[score]) for r in pred if r['view']==v]) for v in ('CC','MLO')},
                    by_laterality={v:metrics([int(r[label]) for r in pred if r['laterality']==v],[float(r[score]) for r in pred if r['laterality']==v]) for v in sorted({r['laterality'] for r in pred})},
                    calibration_bins=calibration(y,p), bootstrap=cluster_bootstrap(y,p,[r[participant] for r in pred]))
    pred=csv_read(RUN+'/test/predictions.csv')
    join_unique(pred,processed['test'],'canonical_image_id','canonical_image_id',{k:k for k in ('participant_id','binary_label','view','laterality')})
    assert all(r['true_label']==r['binary_label'] and int(r['predicted_class'])==int(float(r['predicted_probability'])>=THRESHOLD) for r in pred)
    result['cbis_test']=prediction_audit(pred,'true_label','predicted_probability','participant_id',read(RUN+'/test/metrics.json'))
    result['cbis_test']['bootstrap_mismatches']=disagreement(result['cbis_test']['bootstrap'],read(RUN+'/test/bootstrap_intervals.json')['intervals'])
    print('Recomputed CBIS saved-score metrics and bootstrap',flush=True)
    cohort=csv_read('data/external/CMMD/manifests/cmmd_external_image_cohort_v2.csv')
    pred=csv_read(EXT+'/predictions/cmmd_image_predictions_v2.csv')
    matched=join_unique(pred,cohort,'SOPInstanceUID','sop_uid',dict(patient_id='patient_id',laterality='laterality',view='view',label='label'))
    assert all(int(r['predicted_label'])==int(float(r['probability'])>=THRESHOLD) for r in pred)
    result['cmmd']=prediction_audit(pred,'label','probability','patient_id',read(EXT+'/metrics/cmmd_metrics.json'))
    result['cmmd']['bootstrap_mismatches']=disagreement(result['cmmd']['bootstrap'],read(EXT+'/metrics/cmmd_bootstrap_ci.json')['intervals'])
    grouped=defaultdict(list)
    for r in pred: grouped[(r['patient_id'],r['laterality'])].append(r)
    assert all(sorted(r['view'] for r in group)==['CC','MLO'] and len({r['label'] for r in group})==1 for group in grouped.values())
    result['cmmd']['sides']=len(grouped)
    result['cmmd']['secondary']=metrics([int(g[0]['label']) for g in grouped.values()],[np.mean([float(r['probability']) for r in g]) for g in grouped.values()])
    result['cmmd']['secondary_mismatches']=disagreement(result['cmmd']['secondary'],read(EXT+'/metrics/cmmd_secondary_side_metrics.json'))
    geometry=csv_read(EVIDENCE+'/cmmd_format_conversion_objects.csv')
    join_unique(geometry,cohort,'sop_uid','sop_uid',dict(patient_id='patient_id',laterality='laterality',view='view',label='label',sha256='sha256',array_sha256='array_sha256'))
    fields=['SOPClassUID','Rows','Columns','BitsAllocated','BitsStored','PhotometricInterpretation','SamplesPerPixel','ImageType','PatientOrientation','view_codes','crop_reason','preprocessing_status']
    result['cmmd']['headers_by_view']={view:{f:dict(Counter(r[f] for r in geometry if r['view']==view)) for f in fields} for view in ('CC','MLO')}
    downloaded=csv_read(EVIDENCE+'/cmmd_dicom_headers.csv')
    assert len({r['sop_uid'] for r in downloaded})==len(downloaded)
    result['cmmd']['download']=dict(objects=len(downloaded),patients=len({r['patient_id'] for r in downloaded}),series=len({r['series_uid'] for r in downloaded}),unreadable=sum(bool(r['header_error']) for r in downloaded),contralateral_excluded=sum('contralateral_side_not_in_locked_cohort' in r['reasons'] for r in downloaded))
    print('Recomputed CMMD saved-score metrics, bootstrap and stable-ID joins',flush=True)
    for part,folder in [('validation','validation_localisation'),('test','final_localisation')]:
        summary=read(f'reports/evidence/{folder}/summary.json')
        if part=='test': local=csv_read(f'reports/evidence/{folder}/per_lesion.csv')+csv_read(f'reports/evidence/{folder}/image_union.csv')
        else: local=csv_read(f'reports/evidence/{folder}/per_case_and_lesion.csv')
        recomputed={}
        for scope in ('lesion','image_union'):
            rr=[r for r in local if r['scope']==scope]
            stats=dict(n=len(rr),images=len({r['canonical_image_id'] for r in rr}),participants=len({r['participant_id'] for r in rr}),
                       pointing_game_hits=sum(r['pointing_game']=='True' for r in rr),lesion_hit_hits=sum(r['lesion_hit']=='True' for r in rr),zero_heatmaps=sum(r['zero_heatmap']=='True' for r in rr))
            for k in ('iou','dice','activation_mass_inside'):
                values=[float(r[k]) for r in rr if r[k] not in ('','None')]
                stats['mean_'+k]=float(np.mean(values));stats['median_'+k]=float(np.median(values));stats[k+'_defined_n']=len(values)
            stats['pointing_game_rate']=stats['pointing_game_hits']/len(rr);stats['lesion_hit_rate']=stats['lesion_hit_hits']/len(rr)
            recomputed[scope]=dict(metrics=stats,mismatches=disagreement(stats,summary['groups'][scope+'/overall']))
        result[part+'_localisation']=recomputed
    robust=csv_read('reports/evidence/robustness/robustness_per_image.csv')
    assert all(r['partition']=='validation' and float(r['threshold'])==THRESHOLD for r in robust)
    valid_ids={r['canonical_image_id'] for r in splits['validation']}
    assert {r['canonical_image_id'] for r in robust}==valid_ids
    flip_counts={}
    per_image=defaultdict(list)
    for r in robust:
        flip=(float(r['probability'])>=THRESHOLD)!=(float(r['baseline_probability'])>=THRESHOLD)
        assert (r['flip']=='True')==flip
        flip_counts[r['perturbation']]=flip_counts.get(r['perturbation'],0)+int(flip)
        per_image[r['canonical_image_id']].append(r)
    baseline=[group[0] for group in per_image.values()]
    fns=[r['canonical_image_id'] for r in baseline if r['true_label']=='1' and float(r['baseline_probability'])<THRESHOLD]
    result['robustness']=dict(flip_counts=flip_counts,images=len(per_image),any_flip=sum(any(r['flip']=='True' for r in g) for g in per_image.values()),false_negatives=len(fns),false_negatives_any_flip=sum(any(r['flip']=='True' for r in per_image[k]) for k in fns))
    occ=csv_read('reports/evidence/gradcam_sanity/per_case.csv')
    assert {r['canonical_image_id'] for r in occ}==valid_ids and all(r['partition']=='validation' for r in occ)
    eligible=[r for r in occ if r['status']=='eligible']
    for r in eligible:
        assert np.isclose(float(r['drop_high']),float(r['baseline_probability'])-float(r['probability_high']))
        assert np.isclose(float(r['drop_low']),float(r['baseline_probability'])-float(r['probability_low']))
        high,low=[json.loads(r[k+'_patch_json']) for k in ('high','low')]
        assert abs(high['y']-low['y'])>=32 or abs(high['x']-low['x'])>=32
        assert abs(high['foreground']-low['foreground'])<=.1 and abs(high['brightness']-low['brightness'])<=.1
    result['occlusion']=dict(n_total=len(occ),n_eligible=len(eligible),status_counts=dict(Counter(r['status'] for r in occ)),
                             mean_drop_high=float(np.mean([float(r['drop_high']) for r in eligible])),mean_drop_low=float(np.mean([float(r['drop_low']) for r in eligible])),
                             mean_paired_difference=float(np.mean([float(r['drop_high'])-float(r['drop_low']) for r in eligible])),high_larger_drop_fraction=float(np.mean([float(r['drop_high'])>float(r['drop_low']) for r in eligible])))
    result['occlusion']['mismatches']=disagreement(result['occlusion'],read('reports/evidence/gradcam_sanity/summary.json'))
    result['evidence_sha256']=evidence_hashes
    (output/'computed_checks.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print('Saved read-only recomputation:',output/'computed_checks.json',flush=True)
    return result


if __name__ == '__main__':
    run()
