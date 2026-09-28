"""Read-only integrity verification. Standard library only; never executes inference."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
MANIFEST='reports/evidence/reproducibility_manifest.json'
THRESHOLD=0.3753475248813629
CHECKPOINT='results/experiments/resnet50/20260925T083032_895231Z/best_model.keras'
CHECKPOINT_HASH='0076686617cb29ad5e5682f3d1eff89fae9cad2cebffa86e2b8be2b537934384'


def digest(path,mode='bytes'):
    if mode=='lf': return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    if mode!='bytes': raise ValueError('Unknown hash mode')
    with path.open('rb') as handle: return hashlib.file_digest(handle,'sha256').hexdigest()


def safe_path(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError('Path escapes project')
    return path


def check_hashes(root,bindings):
    failures=[]
    for relative,binding in bindings.items():
        try:
            if digest(safe_path(root,relative),binding['mode'])!=binding['sha256']:
                failures.append('Hash mismatch: '+relative)
        except (OSError,ValueError): failures.append('Missing or unsafe: '+relative)
    return failures


def read(root,relative):
    return json.loads(safe_path(root,relative).read_text(encoding='utf-8'))


def check_locks(root):
    """Cross-check semantic locks and their original bindings, not regenerated values."""
    selection=read(root,'results/experiments/final_selection.json')
    lock=read(root,'results/explainability/localisation_lock.json')
    classifier=read(root,'results/experiments/final_test_attempt.json')
    localisation=read(root,'results/explainability/final_localisation_attempt.json')
    identity=lock['identity']; errors=[]
    if not (selection['status']=='LOCKED_FOR_FINAL_TEST' and selection['run']+'/best_model.keras'==CHECKPOINT
            and selection['threshold']==THRESHOLD and selection['checkpoint_sha256']==CHECKPOINT_HASH
            and lock['status']=='LOCKED' and lock['classifier_threshold']==THRESHOLD
            and lock['heatmap_threshold']==.5 and lock['target_layer']=='conv5_block3_out'
            and identity['checkpoint']==CHECKPOINT and identity['checkpoint_sha256']==CHECKPOINT_HASH
            and lock['coordinate_rules']['equal_grid']=='IDENTITY'
            and lock['coordinate_rules']['unequal_grid']=='EXCLUDE_UNPROVEN'
            and lock['threshold_prespecified'] is True and lock['tuning_after_validation'] is False):
        errors.append('Locked scientific identity changed')
    if classifier['status']!='COMPLETE' or localisation['status']!='COMPLETE':
        errors.append('Final Test completion receipts missing/incomplete')
    bindings={CHECKPOINT:dict(mode='bytes',sha256=CHECKPOINT_HASH),
        'results/experiments/final_selection.json':dict(mode='bytes',sha256=identity['classifier_selection_sha256']),
        'reports/evidence/roi_alignment_protocol.json':dict(mode='lf',sha256=identity['coordinate_protocol_sha256']),
        lock['validation_run']+'/summary.json':dict(mode='bytes',sha256=lock['validation_summary_sha256']),
        classifier['run']+'/test/metrics.json':dict(mode='bytes',sha256=classifier['metrics_sha256']),
        localisation['run']+'/summary.json':dict(mode='bytes',sha256=localisation['summary_sha256']),
        'src/reporting/localisation_validation.py':dict(mode='bytes',sha256=lock['reporting_source_sha256'])}
    for name,value in identity['implementation_sha256'].items():
        bindings['src/explainability/'+name]=dict(mode='bytes',sha256=value)
    for name,value in lock['validation_artifact_sha256'].items():
        bindings[lock['validation_run']+'/'+name]=dict(mode='bytes',sha256=value)
    for partition,value in selection['split_hashes'].items():
        bindings['data/splits/'+partition+'.csv']=dict(mode='bytes',sha256=value)
    for field,name in [('threshold_sha256','threshold.json'),('config_sha256','config_used.yaml'),('run_metadata_sha256','run_metadata.json')]:
        bindings[selection['run']+'/'+name]=dict(mode='bytes',sha256=selection[field])
    errors+=check_hashes(root,bindings)
    return errors


def verify(root=ROOT):
    errors=[]; count=0
    try:
        manifest=read(root,MANIFEST);bindings=manifest['files'];count=len(bindings)
        required={'results/experiments/final_selection.json','results/experiments/final_test_attempt.json',
                  'results/explainability/localisation_lock.json','results/explainability/final_localisation_attempt.json',
                  'reports/evidence/model_card.md','reports/evidence/robustness/robustness_config.json',
                  'reports/evidence/gradcam_sanity/summary.json',CHECKPOINT}
        if not required.issubset(bindings): errors.append('Required evidence omitted from manifest')
        errors+=check_hashes(root,bindings)
        errors+=check_locks(root)
        plan=read(root,'reports/evidence/robustness/robustness_config.json')
        if (plan['partition']!='validation' or plan['classifier_threshold']!=THRESHOLD
                or plan['target_layer']!='conv5_block3_out' or plan['tuning_allowed'] is not False
                or plan['no_test_pixels_or_predictions'] is not True): errors.append('Extended experiment scope changed')
        if digest(root/'src/extended_validation/analysis.py')!=plan['source_sha256']:
            errors.append('Extended experiment implementation differs from prespecified source')
    except (OSError,KeyError,ValueError,TypeError) as exc:
        errors.append('Invalid or missing evidence: '+str(exc))
    return dict(status='FAIL' if errors else 'PASS',files_checked=count,errors=errors,
                inference_executed=False,test_pixels_opened=False,
                note='Hashing saved Test artifacts is read-only. No evaluation or model deserialisation. Receipts remain present to block repeats.')


if __name__=='__main__':
    result=verify();print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='PASS' else 1)
