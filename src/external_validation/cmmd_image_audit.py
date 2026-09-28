"""Local CMMD integrity, side mapping and canonical preprocessing; never inference.

DERIVED objects remain excluded under the requested prospective rule. Diagnostic
preprocessing of side-matched objects does not grant evaluation eligibility.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pydicom
import yaml

from src.preprocessing.canonical import preprocess_dicom
from .cmmd_manifest import load_locked_ids, parse_manifest, BASELINE_HASHES
from .cmmd_provenance import (ROOT, EVIDENCE, CHECKPOINT, CHECKPOINT_SHA256,
                              THRESHOLD, RESULTS, sha256, isolated_path, write_new)

IMAGES = 'data/external/CMMD/images'
MANIFESTS = 'data/external/CMMD/manifests'
PROCESSED = 'data/external/CMMD/processed/cmmd_external_processed.csv'
COHORT = f'{MANIFESTS}/cmmd_external_image_cohort.csv'
IMAGE_LOCK = f'{EVIDENCE}/cmmd_image_cohort_lock.json'
CLINICAL = f'{EVIDENCE}/cmmd_eligible_sides.csv'
CATALOGUE = f'{MANIFESTS}/cmmd_locked_series.csv'
TCIA_MANIFEST = f'{MANIFESTS}/CMMD_locked_mass_external_validation.tcia'
PREPROCESSING = ['config/project.yaml', 'src/preprocessing/canonical.py',
                 'src/preprocessing/dicom.py', 'src/preprocessing/settings.py',
                 'src/preprocessing/model_inputs.py']
SOP_CLASSES = {'1.2.840.10008.5.1.4.1.1.1.2', '1.2.840.10008.5.1.4.1.1.1.2.1'}
VIEW_CODES = {('SCT', '399162004'): 'CC', ('SCT', '399368009'): 'MLO',
              ('SRT', 'R-10242'): 'CC', ('SRT', 'R-10226'): 'MLO'}
COHORT_FIELDS = ['path', 'sha256', 'patient_id', 'study_uid', 'series_uid', 'sop_uid',
                 'laterality', 'view', 'label', 'bytes']


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n', extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def text(ds, key):
    return str(getattr(ds, key, '')).strip()


def laterality(ds):
    values = [text(ds, key) for key in ('ImageLaterality', 'Laterality') if text(ds, key)]
    if not values or any(v not in ('L', 'R') for v in values) or len(set(values)) != 1:
        raise ValueError('missing_or_ambiguous_laterality')
    return values[0]


def view(ds):
    values = []
    if text(ds, 'ViewPosition'):
        values.append(text(ds, 'ViewPosition'))
    sequence = getattr(ds, 'ViewCodeSequence', [])
    if len(sequence) > 1:
        raise ValueError('missing_or_ambiguous_view')
    for item in sequence:
        values.append(VIEW_CODES.get((text(item, 'CodingSchemeDesignator'), text(item, 'CodeValue')), 'UNKNOWN'))
        if getattr(item, 'ViewModifierCodeSequence', []):
            raise ValueError('modified_or_partial_view')
    if getattr(ds, 'ViewModifierCodeSequence', []) or text(ds, 'PartialView') not in ('', 'NO'):
        raise ValueError('modified_or_partial_view')
    if not values or any(v not in ('CC', 'MLO') for v in values) or len(set(values)) != 1:
        raise ValueError('missing_or_ambiguous_view')
    return values[0]


def clinical_map(rows):
    result = {}
    for row in rows:
        key = (row['patient_id'], row['laterality'])
        if key in result or key[1] not in ('L', 'R') or row['label'] not in ('0', '1') or row['abnormality'] != 'mass':
            raise ValueError('Invalid or duplicate frozen clinical side')
        result[key] = int(row['label'])
    return result


def header_record(ds, sides, ids):
    record = {name: text(ds, tag) for name, tag in dict(patient_id='PatientID', study_uid='StudyInstanceUID',
              series_uid='SeriesInstanceUID', sop_uid='SOPInstanceUID', modality='Modality',
              rows='Rows', columns='Columns', photometric='PhotometricInterpretation', bits_stored='BitsStored',
              image_laterality='ImageLaterality', series_laterality='Laterality', view_position='ViewPosition',
              sop_class='SOPClassUID', samples_per_pixel='SamplesPerPixel', number_frames='NumberOfFrames',
              partial_view='PartialView', burned_in_annotation='BurnedInAnnotation').items()}
    record.update(transfer_syntax=text(getattr(ds, 'file_meta', ds), 'TransferSyntaxUID'),
                  image_type='\\'.join(str(v) for v in getattr(ds, 'ImageType', [])),
                  view_codes=';'.join(text(v, 'CodingSchemeDesignator') + ':' + text(v, 'CodeValue')
                                      for v in getattr(ds, 'ViewCodeSequence', [])),
                  laterality='', view='', label='')
    reasons = []
    if record['patient_id'] not in ids:
        reasons.append('patient_not_locked')
    if record['modality'] != 'MG':
        reasons.append('non_mammography')
    if any(not record[key] for key in ('patient_id', 'study_uid', 'series_uid', 'sop_uid')):
        reasons.append('missing_identifiers')
    if record['sop_class'] not in SOP_CLASSES or record['number_frames'] not in ('', '1'):
        reasons.append('unsupported_non_full_object')
    for field, normalizer in (('laterality', laterality), ('view', view)):
        try:
            record[field] = normalizer(ds)
        except ValueError as exc:
            reasons.append(str(exc))
    key = (record['patient_id'], record['laterality'])
    if record['laterality'] and record['patient_id'] in ids:
        if key not in sides:
            reasons.append('contralateral_side_not_in_locked_cohort')
        else:
            record['label'] = sides[key]
    image_type = list(getattr(ds, 'ImageType', []))
    if not image_type or image_type[0] != 'ORIGINAL':
        reasons.append('derived_or_unspecified_image_type')
    record['reasons'] = ';'.join(sorted(set(reasons)))
    return record


def reconcile(records, catalogue, ids, manifest_uids):
    readable = [r for r in records if r.get('sop_uid')]
    sop_counts = Counter(r['sop_uid'] for r in readable)
    duplicates = sorted(k for k, v in sop_counts.items() if v > 1)
    local = defaultdict(list)
    for row in readable:
        local[row['series_uid']].append(row)
    expected = {r['SeriesInstanceUID']: r for r in catalogue}
    discrepancies = []
    for uid in sorted(set(local) & set(expected)):
        rows, cat = local[uid], expected[uid]
        problems = []
        if len(rows) != int(cat['ImageCount']): problems.append('object_count')
        if sum(r['bytes'] for r in rows) != int(cat['FileSize']): problems.append('file_bytes')
        if {r['patient_id'] for r in rows} != {cat['PatientID']}: problems.append('patient')
        if {r['study_uid'] for r in rows} != {cat['StudyInstanceUID']}: problems.append('study')
        if {r['modality'] for r in rows} != {'MG'} or cat['Collection'] != 'CMMD' or cat['Modality'] != 'MG':
            problems.append('collection_or_modality')
        if problems: discrepancies.append(dict(series_uid=uid, reasons=problems))
    patients = {r['patient_id'] for r in readable}
    result = dict(patients=len(patients), studies=len({r['study_uid'] for r in readable}),
                  series=len(local), sop_instances=len(sop_counts), duplicate_sop_uids=duplicates,
                  missing_patients=sorted(set(ids)-patients), extra_patients=sorted(patients-set(ids)),
                  missing_series=sorted(set(expected)-set(local)), extra_series=sorted(set(local)-set(expected)),
                  series_discrepancies=discrepancies,
                  manifest_matches_catalogue=set(manifest_uids) == set(expected),
                  duplicate_catalogue_series=len(expected) != len(catalogue),
                  catalogue_objects=sum(int(r['ImageCount']) for r in catalogue),
                  catalogue_bytes=sum(int(r['FileSize']) for r in catalogue),
                  unreadable_headers=sum(r.get('header_error', '') != '' for r in records))
    result['status'] = 'PASS' if not any(result[k] for k in ('duplicate_sop_uids', 'missing_patients',
        'extra_patients', 'missing_series', 'extra_series', 'series_discrepancies', 'duplicate_catalogue_series',
        'unreadable_headers')) and result['manifest_matches_catalogue'] else 'FAIL'
    return result


def counts(rows):
    sides = {(r['patient_id'], r['laterality']) for r in rows}
    patients = {r['patient_id'] for r in rows}
    result = dict(images=len(rows), sides=len(sides), patients=len(patients),
                  views=dict(sorted(Counter(r['view'] for r in rows).items())),
                  laterality=dict(sorted(Counter(r['laterality'] for r in rows).items())))
    for label, name in ((0, 'benign'), (1, 'malignant')):
        subset = [r for r in rows if int(r['label']) == label]
        result[name] = dict(images=len(subset), sides=len({(r['patient_id'],r['laterality']) for r in subset}),
                            patients=len({r['patient_id'] for r in subset}))
    pv, sv = defaultdict(set), defaultdict(set)
    for r in rows:
        pv[r['patient_id']].add(r['view'])
        sv[(r['patient_id'], r['laterality'])].add(r['view'])
    result['patients_with_cc_and_mlo'] = sum(v == {'CC','MLO'} for v in pv.values())
    result['sides_with_cc_and_mlo'] = sum(v == {'CC','MLO'} for v in sv.values())
    result['patient_label_counts_overlap'] = True
    return result


def validate_preprocessed(result):
    image, metadata = result.image, result.metadata
    if image.shape != (224, 224) or image.dtype != np.float32 or not np.isfinite(image).all():
        raise ValueError('Invalid canonical shape/dtype/finite output')
    if image.min() < 0 or image.max() > 1:
        raise ValueError('Invalid canonical intensity range')
    top, bottom = metadata['padding_top'], metadata['padding_bottom']
    left, right = metadata['padding_left'], metadata['padding_right']
    if any(np.any(border != 0) for border in (image[:top], image[224-bottom:], image[:, :left], image[:, 224-right:])):
        raise ValueError('Nonblack canonical padding')
    return hashlib.sha256(image.tobytes(order='C')).hexdigest()


def _safe_local(root, relative, base):
    path = (root / relative).resolve()
    directory = root / base
    if directory.resolve() != directory or not path.is_relative_to(directory):
        raise ValueError('Redirected CMMD local path')
    return path


def audit(root=ROOT):
    root = Path(root).resolve()
    ids = set(load_locked_ids(root))  # Includes checkpoint, clinical and original provenance locks.
    input_hashes = {p: sha256(root / p) for p in [*BASELINE_HASHES, CATALOGUE, TCIA_MANIFEST, *PREPROCESSING]}
    sides = clinical_map(csv_read(root / CLINICAL))
    catalogue = csv_read(root / CATALOGUE)
    _, manifest_uids = parse_manifest((root / TCIA_MANIFEST).read_bytes())
    outputs = [f'{EVIDENCE}/cmmd_download_integrity.json', f'{EVIDENCE}/cmmd_download_integrity.md',
               f'{EVIDENCE}/cmmd_dicom_headers.csv', f'{EVIDENCE}/cmmd_image_exclusions.csv',
               f'{EVIDENCE}/cmmd_image_cohort_summary.json', f'{EVIDENCE}/cmmd_image_cohort_audit.md',
               IMAGE_LOCK, COHORT, PROCESSED]
    for relative in outputs:
        if (root / relative).exists(): raise FileExistsError(relative)
    image_root = _safe_local(root, IMAGES, IMAGES)
    if not image_root.is_dir(): raise FileNotFoundError('CMMD images absent')
    entries = sorted(image_root.rglob('*'))
    files = [p for p in entries if p.is_file()]
    records, other = [], []
    for path in files:
        relative = path.relative_to(root).as_posix()
        _safe_local(root, relative, IMAGES)
        # Attempt every non-sidecar, including extensionless DICOMs.
        if path.name == '.gitkeep' or (path.suffix.lower() == '.csv' and path.parent.name == 'metadata'):
            other.append(dict(path=relative, bytes=path.stat().st_size))
            continue
        base = dict(path=relative, bytes=path.stat().st_size, sha256=sha256(path), header_error='')
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            base.update(header_record(ds, sides, ids))
        except Exception as exc:
            base.update(header_error=type(exc).__name__, reasons='unreadable_dicom_header')
        records.append(base)
    records.sort(key=lambda r: (r.get('patient_id',''), r.get('sop_uid',''), r['path']))
    integrity = reconcile(records, catalogue, ids, manifest_uids)
    duplicate_sops = set(integrity['duplicate_sop_uids'])
    views = Counter((r.get('patient_id'),r.get('laterality'),r.get('view')) for r in records)
    additional_views = [{'patient_id':k[0], 'laterality':k[1], 'view':k[2], 'count':v}
                        for k,v in sorted(views.items()) if v > 1 and all(k)]
    for row in records:
        if row.get('sop_uid') in duplicate_sops:
            row['reasons'] += ';duplicate_sop_instance'
        if views[(row.get('patient_id'), row.get('laterality'), row.get('view'))] > 1:
            row['reasons'] += ';unresolved_additional_same_side_view'
        row['reasons'] = ';'.join(sorted(set(filter(None, row['reasons'].split(';')))))
    # Diagnostic population: all other requirements except the explicit DERIVED exclusion.
    candidates = [r for r in records if set(filter(None,r['reasons'].split(';'))) <= {'derived_or_unspecified_image_type'}]
    config = yaml.safe_load((root / 'config/project.yaml').read_text())
    processed, sample = [], []
    sample_rows = candidates[::max(1, len(candidates)//12)][:12]
    for row in sample_rows:
        try:
            result = preprocess_dicom(root / row['path'], config)
            sample.append(dict(sop_uid=row['sop_uid'], status='PASS', array_sha256=validate_preprocessed(result)))
        except Exception as exc:
            sample.append(dict(sop_uid=row['sop_uid'], status='FAIL', error=type(exc).__name__))
    print(f'Deterministic diagnostic sample: {len(sample)}; validating {len(candidates)} mapped objects', flush=True)
    sample_hashes = {r['sop_uid']: r.get('array_sha256') for r in sample}
    for i, row in enumerate(candidates):
        item = {k: row.get(k,'') for k in COHORT_FIELDS}
        try:
            result = preprocess_dicom(root / row['path'], config)
            digest = validate_preprocessed(result)
            if row['sop_uid'] in sample_hashes and digest != sample_hashes[row['sop_uid']]:
                raise ValueError('Sample/full preprocessing mismatch')
            item.update(preprocessing_status='PASS', array_sha256=digest, minimum=float(result.image.min()),
                        maximum=float(result.image.max()), shape='224x224', dtype='float32',
                        crop_reason=result.metadata['crop_reason'], error='')
        except Exception as exc:
            item.update(preprocessing_status='FAIL', error=type(exc).__name__)
            row['reasons'] = ';'.join(filter(None,[row['reasons'],'decode_or_preprocessing_failed']))
        item['evaluation_eligible'] = not bool(row['reasons'])
        processed.append(item)
        if (i+1) % 200 == 0: print(f'Preprocessing validation {i+1}/{len(candidates)}', flush=True)
    eligible = [r for r in records if not r['reasons']]
    excluded = [r for r in records if r['reasons']]
    distribution_keys = ['modality','rows','columns','photometric','bits_stored','samples_per_pixel',
                         'image_laterality','series_laterality','view_position','view_codes','image_type',
                         'sop_class','transfer_syntax','partial_view','burned_in_annotation']
    integrity.update(total_files=len(files), dcm_extension_files=sum(p.suffix.lower()=='.dcm' for p in files),
                     directories=sum(p.is_dir() for p in entries), total_file_bytes=sum(p.stat().st_size for p in files),
                     dicom_file_bytes=sum(r['bytes'] for r in records if not r['header_error']),
                     other_files=other, size_measurement='Logical file lengths; filesystem allocated clusters not measured.',
                     distributions={k: dict(sorted(Counter(r.get(k,'MISSING') or 'MISSING' for r in records).items()))
                                    for k in distribution_keys}, additional_side_views=additional_views,
                     manual_retriever_report=dict(completed_series=1090, failed=0, skipped=0, cancelled=0,
                         queued=0, in_progress=0, displayed_gb=4.46, elapsed='6:09:15', provenance='User-supplied report'),
                     transfer_size_conclusion='Local uncompressed DICOM byte total equals the locked official catalogue. '
                         'No count/size deficit. The 4.46 GB GUI figure is not local logical size. Compressed transfer '
                         'or GUI accounting is possible, but its precise basis is unverified without version/log evidence.',
                     official_metadata=f'{CATALOGUE} (NBIA catalogue snapshot; no new downloads)',
                     input_hashes=input_hashes)
    exclusions = dict(sorted(Counter(reason for r in excluded for reason in r['reasons'].split(';')).items()))
    summary = dict(status='BLOCKED', reasons=[], clinical_counts=dict(sides=len(sides), patients=len(ids)),
                   eligible=counts(eligible), diagnostic_side_matched_candidates=counts(candidates),
                   exclusions_by_reason=exclusions, excluded_images=len(excluded),
                   exclusion_counts_overlap=True, derived_policy='STRICT_EXCLUSION',
                   preprocessing=dict(scope='Diagnostic side-matched candidates; not an eligibility override',
                       sample=sample, tested=len(processed), passed=sum(r['preprocessing_status']=='PASS' for r in processed),
                       failed=sum(r['preprocessing_status']=='FAIL' for r in processed),
                       canonical_version='canonical_v1', configuration='config/project.yaml', configuration_unchanged=True,
                       data_provenance='CMMD; CBIS-DDSM config reused solely as immutable algorithm parameters',
                       adapter_for_future='src.preprocessing.model_inputs.transfer_input(..., family="resnet50")',
                       decoding_changes='None', caches_written=False), predictions_generated=0,
                   unresolved_additional_views=additional_views)
    if integrity['status'] != 'PASS': summary['reasons'].append('Download reconciliation failed')
    if not eligible: summary['reasons'].append('No eligible images under strict DERIVED exclusion; all downloaded objects are DERIVED/PRIMARY')
    if summary['preprocessing']['failed']: summary['reasons'].append('Decoding/preprocessing failures require review')
    if additional_views: summary['reasons'].append('Additional same-side views require resolution')
    if not summary['reasons']: summary['status'] = 'READY_FOR_CMMD_EXTERNAL_INFERENCE'
    # Freeze the empty or eligible cohort honestly; never substitute diagnostic candidates.
    blobs = {COHORT: csv_bytes(eligible, COHORT_FIELDS),
             f'{EVIDENCE}/cmmd_dicom_headers.csv': csv_bytes(records, list(dict.fromkeys(k for r in records for k in r))),
             f'{EVIDENCE}/cmmd_image_exclusions.csv': csv_bytes(excluded, COHORT_FIELDS+['reasons']),
             PROCESSED: csv_bytes(processed, COHORT_FIELDS+['preprocessing_status','evaluation_eligible','array_sha256',
                 'minimum','maximum','shape','dtype','crop_reason','error'])}
    for relative, obj in ((f'{EVIDENCE}/cmmd_download_integrity.json', integrity),
                          (f'{EVIDENCE}/cmmd_image_cohort_summary.json', summary)):
        blobs[relative] = (json.dumps(obj, indent=2, sort_keys=True)+'\n').encode()
    sources = ('View normalization: https://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_4014.html\n\n'
               'Collection provenance (TIFF-to-DICOM conversion): https://www.cancerimagingarchive.net/collection/cmmd/\n\n'
               'Retriever transfer/decompression workflow: https://github.com/TCIA/data-retriever\n')
    blobs[f'{EVIDENCE}/cmmd_download_integrity.md'] = (f'# CMMD local download integrity: {integrity["status"]}\n\n'
        f'{len(records):,} objects; {integrity["patients"]:,} patients; {integrity["series"]:,} series. '
        f'DICOM logical size: {integrity["dicom_file_bytes"]:,} bytes ({integrity["dicom_file_bytes"]/1e9:.9f} GB).\n\n'
        f'{integrity["transfer_size_conclusion"]}\n\n'
        'Every series is reconciled by patient, study, object count and file-size sum. Exact official SOP membership '
        'and server-side checksums were not supplied by the catalogue; this is not a server checksum attestation.\n\n'
        'Header evidence contains pseudonymous identifiers and only mapping/encoding fields; no names, birth dates or accession numbers.\n\n'
        + sources).encode()
    blobs[f'{EVIDENCE}/cmmd_image_cohort_audit.md'] = (f'# CMMD image cohort: {summary["status"]}\n\n'
        + '\n'.join('- '+reason for reason in summary['reasons']) + '\n\n'
        f'Frozen clinical cohort: {len(sides):,} sides, {len(ids):,} patients. No eligibility changes.\n\n'
        f'Actual eligible cohort: {len(eligible):,} images. Diagnostic side-matched candidates: {len(candidates):,}. '
        'The latter are not approved for evaluation. A header-only DERIVED exclusion is retained pending explicit clarification '
        'of whether documented TIFF-to-DICOM full-mammogram conversions may qualify. No exception is inferred from sample size.\n\n'
        'ImageLaterality determines side; standard coded ViewCodeSequence determines CC/MLO when ViewPosition is absent. '
        'Conflicting tags, modifiers, partial views, duplicate SOPs and unresolved same-side extra views are rejected. '
        'Contralateral images never inherit a diagnosis. Patient label categories may overlap for bilateral different diagnoses.\n\n'
        f'Canonical preprocessing: {summary["preprocessing"]["passed"]:,}/{len(processed):,} diagnostic objects passed '
        'after a deterministic 12-object sample. Outputs are finite float32 224x224 within [0,1], with checked black padding. '
        'The original grayscale decode, MONOCHROME handling, crop and aspect-preserving resize are called unchanged. '
        'No adapter/model was loaded, no predictions or model caches were written.\n\n'
        'The processed CSV records diagnostic validation, not inference readiness. The new image lock supplements, '
        'and does not rewrite, the frozen clinical/download locks. A BLOCKED lock forbids inference.\n\n'+sources).encode()
    bound = dict(input_hashes)
    bound.update({name: hashlib.sha256(content).hexdigest() for name,content in blobs.items()})
    bound['src/external_validation/cmmd_image_audit.py'] = sha256(root/'src/external_validation/cmmd_image_audit.py')
    lock = dict(status=summary['status'], dataset='CMMD', purpose='external_validation', reasons=summary['reasons'],
                clinical_cohort_sha256=input_hashes[CLINICAL], image_cohort_sha256=bound[COHORT],
                checkpoint=CHECKPOINT, checkpoint_sha256=CHECKPOINT_SHA256, classifier_threshold=THRESHOLD,
                preprocessing_hashes={p: input_hashes[p] for p in PREPROCESSING}, counts=summary['eligible'],
                exclusions_by_reason=exclusions, training_allowed=False, fine_tuning_allowed=False,
                model_selection_allowed=False, calibration_allowed=False, threshold_tuning_allowed=False,
                predictions_generated=0, inference_enabled=False, output_root=RESULTS, bound_files=bound)
    blobs[IMAGE_LOCK] = (json.dumps(lock,indent=2,sort_keys=True)+'\n').encode()
    load_locked_ids(root)
    for name, digest in input_hashes.items():
        if sha256(root/name) != digest: raise ValueError('Audit input changed during run: '+name)
    for relative, content in blobs.items():
        if relative == PROCESSED:
            target = _safe_local(root, relative, 'data/external/CMMD/processed')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream: stream.write(content)
        else:
            write_new(relative, content.decode('utf-8'), root,
                      evidence=relative.startswith(EVIDENCE+'/'), manifests=relative.startswith(MANIFESTS+'/'))
    return summary


def verify_image_lock(root=ROOT, *, require_ready=True):
    root = Path(root).resolve()
    load_locked_ids(root)
    lock = json.loads((root/IMAGE_LOCK).read_text())
    expected = dict(dataset='CMMD', purpose='external_validation', checkpoint=CHECKPOINT,
                    checkpoint_sha256=CHECKPOINT_SHA256, classifier_threshold=THRESHOLD, output_root=RESULTS,
                    training_allowed=False, fine_tuning_allowed=False, model_selection_allowed=False,
                    calibration_allowed=False, threshold_tuning_allowed=False, inference_enabled=False, predictions_generated=0)
    for key, value in expected.items():
        if type(lock.get(key)) is not type(value) or lock[key] != value: raise ValueError('Image lock changed: '+key)
    required = {*BASELINE_HASHES, CATALOGUE, TCIA_MANIFEST, *PREPROCESSING, COHORT, PROCESSED,
                f'{EVIDENCE}/cmmd_dicom_headers.csv', 'src/external_validation/cmmd_image_audit.py'}
    if not required <= set(lock['bound_files']): raise ValueError('Image lock omits required bindings')
    for relative, digest in lock['bound_files'].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest: raise ValueError('Image binding changed: '+relative)
    if lock['clinical_cohort_sha256'] != sha256(root/CLINICAL) or lock['image_cohort_sha256'] != sha256(root/COHORT):
        raise ValueError('Cohort digest mismatch')
    rows = csv_read(root/COHORT)
    seen, side_views = set(), set()
    sides = clinical_map(csv_read(root/CLINICAL))
    for row in rows:
        source = _safe_local(root,row['path'],IMAGES)
        if sha256(source) != row['sha256']: raise ValueError('Source DICOM changed')
        mapped = header_record(pydicom.dcmread(source,stop_before_pixels=True), sides, set(load_ids_from_sides(sides)))
        if mapped['reasons'] or any(str(mapped[k]) != str(row[k]) for k in ('patient_id','laterality','view','label','study_uid','series_uid','sop_uid')):
            raise ValueError('Ambiguous or changed image mapping')
        key = (row['patient_id'],row['laterality'],row['view'])
        if row['sop_uid'] in seen or key in side_views: raise ValueError('Duplicate image mapping')
        seen.add(row['sop_uid']); side_views.add(key)
    if counts(rows) != lock['counts']: raise ValueError('Image counts changed')
    isolated_path(RESULTS+'/predictions/future.csv', root)
    if require_ready and (lock['status'] != 'READY_FOR_CMMD_EXTERNAL_INFERENCE' or not rows or lock['reasons']):
        raise ValueError('BLOCKED: '+ '; '.join(lock['reasons']))
    return lock


def load_ids_from_sides(sides):
    return {key[0] for key in sides}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Read-only readiness guard; exits nonzero when blocked')
    args = parser.parse_args()
    result = verify_image_lock() if args.verify else audit()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
