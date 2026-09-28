"""Immutable scientific bindings and narrowly scoped write destinations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = 'results/experiments/resnet50/20260925T083032_895231Z/best_model.keras'
CHECKPOINT_SHA256 = '0076686617cb29ad5e5682f3d1eff89fae9cad2cebffa86e2b8be2b537934384'
THRESHOLD = 0.3753475248813629
RESULTS = 'results/external_validation/CMMD'
EVIDENCE = 'reports/evidence/external_validation/CMMD'
WAITING = 'WAITING_FOR_CMMD_CLINICAL_METADATA'
LOCK = 'config/external_validation/cmmd_lock.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config):
    expected = dict(dataset='CMMD', mode='external_validation', training_allowed=False,
                    fine_tuning_allowed=False, threshold_tuning_allowed=False,
                    model_selection_allowed=False, calibration_allowed=False,
                    selected_model='resnet50', checkpoint=CHECKPOINT,
                    checkpoint_sha256=CHECKPOINT_SHA256, classifier_threshold=THRESHOLD,
                    input_size=224, output_root=RESULTS, inference_enabled=False,
                    official_test_evaluation_allowed=False)
    for key, value in expected.items():
        if type(config.get(key)) is not type(value) or config[key] != value:
            raise ValueError(f'Frozen external-validation binding changed: {key}')
    return config


def verify_frozen(root=ROOT):
    import yaml
    root = Path(root)
    validate_config(yaml.safe_load((root / 'config/external_validation/cmmd.yaml').read_text()))
    lock = json.loads((root / LOCK).read_text())
    if lock['checkpoint_sha256'] != CHECKPOINT_SHA256 or lock['classifier_threshold'] != THRESHOLD:
        raise ValueError('Invalid provenance lock')
    if sha256(root / CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError('Selected checkpoint changed')
    for name, digest in lock['frozen_files'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'Frozen file changed: {name}')
    return lock


def isolated_path(relative, root=ROOT, *, evidence=False, manifests=False):
    """Resolve junctions/symlinks and reject absolute paths and traversal."""
    relative = Path(relative)
    if relative.is_absolute() or relative.drive or '..' in relative.parts:
        raise ValueError('Output must be a relative external path')
    root = Path(root).resolve()
    if evidence and manifests:
        raise ValueError('Choose one isolated output destination')
    base = root / ('data/external/CMMD/manifests' if manifests else EVIDENCE if evidence else RESULTS)
    if not base.resolve().is_relative_to(root) or base.resolve() != base:
        raise ValueError('Redirected output root')
    target = (root / relative).resolve()
    if not target.is_relative_to(base) or target == base:
        raise ValueError('Output outside isolated CMMD destination')
    return target


def write_new(relative, text, root=ROOT, *, evidence=False, manifests=False):
    target = isolated_path(relative, root, evidence=evidence, manifests=manifests)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8', newline='') as stream:
        stream.write(text)


def run_inference(*args, **kwargs):
    raise RuntimeError('Inference disabled: metadata, DICOM mapping and harmonisation review required')
