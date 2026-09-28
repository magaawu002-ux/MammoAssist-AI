"""Verify the human-authorised identity/exclusion rule, never model outputs."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

from src.preprocessing.manifests import local_path, read_rows

PROTOCOL = 'reports/evidence/roi_alignment_protocol.json'
ANCHOR = 'config/roi_alignment_lock.json'
VERSION = 'cbis_equal_grid_identity_v1'
RULES = {'equal_grid': 'IDENTITY', 'unequal_grid': 'EXCLUDE_UNPROVEN',
         'exclusion_reason': 'exact_spatial_transform_unproven'}
IMPLEMENTATION = tuple('src/explainability/'+name+'.py' for name in
    ('coordinate_protocol','coordinate_checks','coordinate_evidence','alignment',
     'roi_audit','roi_mapping','roi_masks','localisation')) + tuple(
    'src/preprocessing/'+name+'.py' for name in ('canonical','dicom','manifests'))
EVIDENCE = tuple('reports/evidence/'+name for name in (
    'roi_alignment_review/index.csv','roi_alignment_human_review.json',
    'roi_alignment_geometry.csv','roi_alignment_geometry_summary.json',
    'roi_alignment_development_checks.csv','roi_alignment_equal_grid_validation.json',
    'roi_alignment_validation_contact_sheet.png','real_roi_train_audit.csv','real_roi_test_audit.csv'))


def bound_hash(path: Path) -> str:
    """SHA256 with LF-normalised text so Git's Windows checkout cannot break a lock."""
    content = path.read_bytes()
    if path.suffix in ('.py','.json','.csv','.yaml','.md'):
        content = content.replace(b'\r\n',b'\n')
    return hashlib.sha256(content).hexdigest()


def coordinate_rule(mask_shape: tuple[int, int], full_shape: tuple[int, int]) -> str:
    if any(type(v) is not int or v <= 0 for v in (*mask_shape,*full_shape)) or len(mask_shape)!=2 or len(full_shape)!=2:
        raise ValueError('Invalid coordinate dimensions')
    return 'IDENTITY' if mask_shape == full_shape else 'EXCLUDE_UNPROVEN'


def geometry_counts(rows: list[dict]) -> dict:
    result = {}
    for partition in ('train','validation','test'):
        subset = [r for r in rows if r['final_partition']==partition]
        eligible = [r for r in subset if coordinate_rule((int(r['mask_rows']),int(r['mask_columns'])),
                    (int(r['full_rows']),int(r['full_columns'])))=='IDENTITY']
        counts = Counter(r['canonical_image_id'] for r in eligible)
        result[partition] = dict(total_masks=len(subset),eligible_masks=len(eligible),
            excluded_masks=len(subset)-len(eligible),eligible_mammograms=len(counts),
            eligible_participants=len({r['participant_id'] for r in eligible}),
            eligible_multiple_lesion_mammograms=sum(n>1 for n in counts.values()))
    return result


def load_coordinate_protocol(root: Path) -> dict | None:
    """Fail closed on any present-but-invalid lock; absence retains strict DICOM proof."""
    if not (root/PROTOCOL).exists() and not (root/ANCHOR).exists():
        return None
    if not (root/ANCHOR).is_file() or not (root/PROTOCOL).is_file():
        raise ValueError('Coordinate protocol requires its independent lock receipt')
    anchor = json.loads((root/ANCHOR).read_text(encoding='utf-8'))
    if anchor.get('protocol') != PROTOCOL or anchor.get('protocol_sha256') != bound_hash(root/PROTOCOL):
        raise ValueError('Coordinate protocol hash mismatch')
    protocol = json.loads((root/PROTOCOL).read_text(encoding='utf-8'))
    if (protocol.get('rule_version') != VERSION or protocol.get('status') != 'LOCKED_FOR_VALIDATION_ALIGNMENT'
            or protocol.get('locked') is not True or protocol.get('rules') != RULES
            or protocol.get('hash_scheme') != 'sha256_lf_text_v1'
            or protocol.get('development_partitions') != ['train','validation']
            or protocol.get('gradcam_outputs_used') is not False
            or protocol.get('test_localisation_outcomes_used') is not False):
        raise ValueError('Unsupported coordinate protocol/rules or development scope')
    implementation = protocol.get('implementation_sha256',{})
    evidence = protocol.get('evidence_sha256',{})
    if set(implementation) != set(IMPLEMENTATION) or not set(EVIDENCE) <= set(evidence):
        raise ValueError('Incomplete coordinate protocol hash bindings')
    if protocol.get('config_sha256') != bound_hash(root/'config/localisation.yaml'):
        raise ValueError('Coordinate protocol config hash mismatch')
    for bindings, prefix in ((implementation,('src',)),(evidence,('reports','evidence'))):
        for reference, expected in bindings.items():
            if bound_hash(local_path(root,reference,prefix)) != expected:
                raise ValueError(f'Coordinate evidence/implementation hash mismatch: {reference}')
    decisions = json.loads((root/'reports/evidence/roi_alignment_human_review.json').read_text(encoding='utf-8'))
    index = read_rows(root/'reports/evidence/roi_alignment_review/index.csv')
    by_panel = {r['panel']:r for r in index}
    if len(by_panel)!=len(index) or decisions.get('approved_rules') != RULES:
        raise ValueError('Duplicate reviews or missing human rule authorisation')
    reviewed = set()
    for decision in decisions['cases']:
        row = by_panel.get(decision['panel'])
        if row is None or decision['panel'] in reviewed or any(row.get(k)!=decision.get(k) for k in
                ('canonical_image_id','abnormality_id','review_status','reviewer','reviewed_at','review_notes')):
            raise ValueError('Human review does not match locked index')
        if row['review_status'] not in ('SUPPORTED','UNRESOLVED'):
            raise ValueError('Unsupported human review outcome')
        if row['review_status']=='SUPPORTED' and row['grid_relation']!='equal':
            raise ValueError('Proportional transformation has no human approval')
        reviewed.add(row['panel'])
    if not reviewed or any(r['review_status']!='PENDING_HUMAN_REVIEW' for r in index if r['panel'] not in reviewed):
        raise ValueError('Unrecorded human review')
    for row in index:
        if 'reports/evidence/roi_alignment_review/'+row['panel'] not in evidence:
            raise ValueError('Review panel is not hash bound')
    if protocol.get('counts_by_partition') != geometry_counts(read_rows(root/'reports/evidence/roi_alignment_geometry.csv')):
        raise ValueError('Coordinate protocol inventory counts mismatch')
    return protocol


def locked_cohort(root: Path, protocol: dict, partition: str, table: list[dict]) -> dict:
    """Compare actual scoped mappings to frozen inventory; no test pixels or metrics."""
    for reference, expected in protocol['partition_source_sha256'][partition].items():
        path=local_path(root,reference,('data',))
        if bound_hash(path)!=expected:
            raise ValueError(f'Coordinate cohort source changed: {reference}')
    official='test' if partition=='test' else 'train'
    baseline=[r for r in read_rows(root/f'reports/evidence/real_roi_{official}_audit.csv') if r['final_partition']==partition]
    identity_fields=('record_id','role','canonical_image_id','abnormality_id','participant_id',
                     'reference','path','SOPInstanceUID','mapping_status')
    identity=lambda rows: sorted(tuple(r.get(k,'') for k in identity_fields) for r in rows)
    if identity(table)!=identity(baseline):
        raise ValueError('ROI mappings differ from locked coordinate inventory')
    geometry=[r for r in read_rows(root/'reports/evidence/roi_alignment_geometry.csv') if r['final_partition']==partition]
    by_record={r['record_id']:r for r in geometry}
    if len(by_record)!=len(geometry) or set(by_record)!={r['record_id'] for r in baseline if r['role']=='mask'}:
        raise ValueError('Coordinate geometry cohort mismatch')
    return by_record


def finalise_protocol(root: Path) -> dict:
    """Explicit one-time finalisation from recorded human decisions; no image inference."""
    if (root/ANCHOR).exists():
        raise ValueError('Coordinate protocol already locked; do not overwrite the lock')
    human=json.loads((root/'reports/evidence/roi_alignment_human_review.json').read_text(encoding='utf-8'))
    if human.get('approved_rules')!=RULES:
        raise ValueError('Explicit human identity/exclusion rule authorisation required')
    quantitative=json.loads((root/'reports/evidence/roi_alignment_equal_grid_validation.json').read_text(encoding='utf-8'))
    if quantitative.get('development_partitions')!=['train','validation'] or quantitative.get('test_pixels_used') is not False or quantitative.get('scores_computed') is not False:
        raise ValueError('Coordinate evidence development scope is invalid')
    index=read_rows(root/'reports/evidence/roi_alignment_review/index.csv')
    evidence=list(EVIDENCE)+['reports/evidence/roi_alignment_review/'+r['panel'] for r in index]
    geometry=read_rows(root/'reports/evidence/roi_alignment_geometry.csv')
    counts=geometry_counts(geometry)
    protocol=dict(rule_version=VERSION,status='LOCKED_FOR_VALIDATION_ALIGNMENT',locked=True,
        hash_scheme='sha256_lf_text_v1',rules=RULES,development_partitions=['train','validation'],
        gradcam_outputs_used=False,test_localisation_outcomes_used=False,test_geometry_used_to_select_rule=False,
        test_eligibility='Inventoried only. Final evaluation still requires completed validation and a separate final-localisation lock.',
        counts_by_partition=counts,
        official_training_totals={key:counts['train'][key]+counts['validation'][key] for key in ('total_masks','eligible_masks','excluded_masks')},
        human_review_cases=human['cases'],human_review_scope=human['scope'],
        human_review_counts=dict(Counter(r['review_status'] for r in human['cases'])),
        quantitative_crop_correspondence=quantitative['results'],
        diagnostic_screen_interpretation='Historical foreground/Otsu screening is not an admission gate. Human review supports equal-grid coordinate correspondence despite tissue-proxy limitations.',
        exclusions='All unequal-grid annotations; geometry only, independent of pathology and model outputs. No proportional scaling or translation is approved.',
        multiple_lesion_handling='Retain separate eligible lesions and their OR-union per eligible image. Keep exclusions in the audit; never transform excluded annotations.',
        implementation_sha256={p:bound_hash(root/p) for p in IMPLEMENTATION},
        config_sha256=bound_hash(root/'config/localisation.yaml'),
        evidence_sha256={p:bound_hash(local_path(root,p,('reports','evidence'))) for p in evidence},
        partition_source_sha256={},locked_at_utc=datetime.now(timezone.utc).isoformat())
    for partition in ('train','validation','test'):
        official='test' if partition=='test' else 'train'
        paths=(f'data/processed/manifests/{partition}_processed.csv',f'data/splits/{partition}.csv',
               f'data/raw/CBIS-DDSM/metadata/mass_case_description_{official}_set.csv')
        protocol['partition_source_sha256'][partition]={p:bound_hash(root/p) for p in paths}
    (root/PROTOCOL).write_text(json.dumps(protocol,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    anchor={'protocol':PROTOCOL,'protocol_sha256':bound_hash(root/PROTOCOL),'hash_scheme':'sha256_lf_text_v1',
            'scope':'Coordinate rule only; not a final Test localisation lock'}
    with (root/ANCHOR).open('x',encoding='utf-8') as handle:
        json.dump(anchor,handle,indent=2);handle.write('\n')
    load_coordinate_protocol(root)
    return protocol
