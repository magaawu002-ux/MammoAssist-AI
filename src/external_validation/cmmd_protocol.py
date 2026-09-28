"""Freeze an audited clinical cohort and plan selective transfer; no network/I/O of pixels."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

from .cmmd_audit import csv_text
from .cmmd_cohort import build_cohort
from .cmmd_metadata import inspect_workbook
from .cmmd_provenance import (ROOT, EVIDENCE, RESULTS, CHECKPOINT, CHECKPOINT_SHA256,
                              THRESHOLD, sha256, verify_frozen, write_new, isolated_path)

MANIFESTS = 'data/external/CMMD/manifests'
PROTOCOL = f'{EVIDENCE}/cmmd_external_protocol.json'
READY = 'READY_FOR_CMMD_IMAGE_DOWNLOAD'


def download_plan(eligible, catalogue):
    """Use complete series for eligible patients; catalogue has no per-image side tags."""
    if not eligible or {r['label'] for r in eligible} != {0, 1}:
        raise ValueError('Require a nonempty benign/malignant clinical cohort')
    by_patient = {}
    for row in eligible:
        if row['reasons'] or row['abnormality'] != 'mass' or row['laterality'] not in ('L', 'R'):
            raise ValueError('Invalid eligible side')
        sides = by_patient.setdefault(row['patient_id'], {})
        if row['laterality'] in sides:
            raise ValueError('Duplicate eligible side')
        sides[row['laterality']] = row['label']
    found, uids, plan = set(), set(), []
    for series in catalogue:
        patient = series['PatientID']
        if patient not in by_patient:
            continue
        uid = series['SeriesInstanceUID']
        if series['Collection'] != 'CMMD' or series['Modality'] != 'MG' or not uid or uid in uids:
            raise ValueError('Invalid or duplicate catalogue series')
        if not series['StudyInstanceUID'] or int(series['ImageCount']) < 1 or int(series['FileSize']) < 1:
            raise ValueError('Incomplete catalogue metadata')
        uids.add(uid); found.add(patient)
        sides = by_patient[patient]
        plan.append(dict(patient_id=patient, eligible_sides=';'.join(sorted(sides)),
                         left_label=sides.get('L', ''), right_label=sides.get('R', ''),
                         study_uid=series['StudyInstanceUID'], series_uid=uid,
                         catalogue_dicom_count=int(series['ImageCount']), catalogue_bytes=int(series['FileSize']),
                         download_status='NOT_DOWNLOADED', header_verification='PENDING'))
    if found != set(by_patient):
        raise ValueError('Eligible PatientIDs missing from public catalogue')
    return sorted(plan, key=lambda r: (r['patient_id'], r['series_uid']))


def plan_csv(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def freeze_protocol(workbook, mapping_path, catalogue_path, catalogue_source_path, root=ROOT):
    root = Path(root)
    verify_frozen(root)
    mapping = json.loads(Path(mapping_path).read_text(encoding='utf-8'))
    sheets = inspect_workbook(workbook)
    eligible, excluded = build_cohort(sheets, mapping['columns'], mapping['side_specific_evidence'])
    # Bind original sources, both CSVs and the real audit. Refuse post-audit drift.
    audit_summary = json.loads((root / EVIDENCE / 'cmmd_metadata_summary.json').read_text())
    if audit_summary['spreadsheet_sha256'] != sha256(workbook) or audit_summary['mapping'] != mapping:
        raise ValueError('Workbook/mapping changed after audit')
    for name, rows in (('cmmd_eligible_sides.csv', eligible), ('cmmd_exclusions.csv', excluded)):
        if (root / EVIDENCE / name).read_bytes() != csv_text(rows).encode('utf-8'):
            raise ValueError('Deterministic audited cohort mismatch')
    with Path(catalogue_path).open(encoding='utf-8', newline='') as stream:
        catalogue = list(csv.DictReader(stream))
    source = json.loads(Path(catalogue_source_path).read_text())
    if source['catalogue_csv_sha256'] != sha256(catalogue_path):
        raise ValueError('Catalogue source hash mismatch')
    plan = download_plan(eligible, catalogue)
    ids = sorted({r['patient_id'] for r in eligible})
    files = {f'{MANIFESTS}/cmmd_eligible_patient_ids.txt': '\n'.join(ids) + '\n',
             f'{MANIFESTS}/cmmd_download_plan.csv': plan_csv(plan)}
    bound = {str(Path(p).resolve().relative_to(root.resolve())).replace('\\', '/'): sha256(p)
             for p in (workbook, mapping_path, catalogue_path, catalogue_source_path)}
    for name in ('cmmd_eligible_sides.csv', 'cmmd_exclusions.csv', 'cmmd_metadata_summary.json', 'cmmd_metadata_audit.md'):
        bound[f'{EVIDENCE}/{name}'] = sha256(root / EVIDENCE / name)
    for path, content in files.items():
        bound[path] = hashlib.sha256(content.encode('utf-8')).hexdigest()
    for file in (root / 'src/external_validation').glob('*.py'):
        bound[file.relative_to(root).as_posix()] = sha256(file)
    protocol = dict(dataset='CMMD', purpose='external_validation', status=READY,
                    training_allowed=False, fine_tuning_allowed=False, model_selection_allowed=False,
                    threshold_tuning_allowed=False, calibration_allowed=False, inference_enabled=False,
                    selected_model='resnet50', checkpoint=CHECKPOINT, checkpoint_sha256=CHECKPOINT_SHA256,
                    classifier_threshold=THRESHOLD, input_size=224, output_root=RESULTS,
                    spreadsheet_sha256=sha256(workbook), cohort_csv_sha256=bound[f'{EVIDENCE}/cmmd_eligible_sides.csv'],
                    cohort_definition='Explicit single-side benign/malignant MASS only, unambiguous catalogue-matched imaging PatientID.',
                    inclusion_rules=['Explicit classification', 'Explicit affected side', 'Mass only', 'Safe PatientID mapping'],
                    exclusion_rules=sorted({reason for row in excluded for reason in row['reasons']}),
                    additional_prohibitions=['No unlabelled contralateral-side inclusion', 'No clinical label propagation',
                                            'No official CBIS-DDSM Test rerun', 'No preprocessing change', 'No inference before DICOM review'],
                    eligible_patient_ids=ids, eligible_patients=len(ids), eligible_sides=len(eligible),
                    benign_sides=sum(r['label'] == 0 for r in eligible), malignant_sides=sum(r['label'] == 1 for r in eligible),
                    expected_preprocessing=dict(config='config/project.yaml', version='canonical_v1', input_size=224,
                        canonical_function='src.preprocessing.canonical.preprocess_dicom',
                        adapter='src.preprocessing.model_inputs.transfer_input', family='resnet50',
                        harmonisation='Pending label-independent DICOM compatibility review'),
                    download=dict(method='TCIA NBIA patient selection, complete MG series', expected_series=len(plan),
                        catalogue_dicom_count=sum(r['catalogue_dicom_count'] for r in plan),
                        catalogue_bytes=sum(r['catalogue_bytes'] for r in plan),
                        downloaded_images=0, eligible_image_count=None,
                        note='Transfer estimates only; contralateral/noneligible objects require header filtering.'),
                    bound_files=bound)
    files[PROTOCOL] = json.dumps(protocol, indent=2) + '\n'
    for path in files:
        if isolated_path(path, root, evidence=path == PROTOCOL, manifests=path != PROTOCOL).exists():
            raise FileExistsError('Existing cohort lock/download plan must not be overwritten')
    for path, content in files.items():
        write_new(path, content, root, evidence=path == PROTOCOL, manifests=path != PROTOCOL)
    return protocol


def verify_protocol(root=ROOT):
    root = Path(root)
    verify_frozen(root)
    protocol = json.loads((root / PROTOCOL).read_text())
    for key, expected in dict(dataset='CMMD', purpose='external_validation', status=READY,
                             checkpoint=CHECKPOINT, checkpoint_sha256=CHECKPOINT_SHA256,
                             classifier_threshold=THRESHOLD, selected_model='resnet50', input_size=224, output_root=RESULTS).items():
        if protocol.get(key) != expected:
            raise ValueError(f'Protocol binding changed: {key}')
    for key in ('training_allowed', 'fine_tuning_allowed', 'model_selection_allowed',
                'threshold_tuning_allowed', 'calibration_allowed', 'inference_enabled'):
        if protocol.get(key) is not False:
            raise ValueError(f'Prohibited protocol permission: {key}')
    for relative, expected in protocol['bound_files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError(f'Protocol source changed: {relative}')
    return protocol
