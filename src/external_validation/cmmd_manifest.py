"""Read-only verification of locked patients, catalogue metadata and NBIA manifests.

No network client, Data Retriever, model, inference or cohort-rebuilding code.
The caller supplies an official catalogue response and official manifest bytes.
"""
from collections import Counter, defaultdict
import hashlib
import re
from pathlib import Path

from .cmmd_protocol import verify_protocol, PROTOCOL
from .cmmd_provenance import ROOT, EVIDENCE, sha256

BASELINE = 'eb9fdd7'
IDS = 'data/external/CMMD/manifests/cmmd_eligible_patient_ids.txt'
BASELINE_HASHES = {
    PROTOCOL: 'bdd59230bd7974a72e93bdd66a03978f1cbac55ae7ad30e4841d3e202311964a',
    IDS: 'a91ee92208604bf4e2efe41b707dd0e077159cad04f0474a7808da44456eaed1',
    f'{EVIDENCE}/cmmd_eligible_sides.csv': '736ae87c64b3b0dd4ed284d67a56e0c368b327f9089453e657b1bc9d6c90d506',
}
FIELDS = ('PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'Collection', 'Modality', 'ImageCount', 'FileSize')


def load_locked_ids(root=ROOT):
    root = Path(root)
    for name, expected in BASELINE_HASHES.items():
        if sha256(root / name) != expected:
            raise ValueError(f'Patient/cohort lock differs from {BASELINE}: {name}')
    protocol = verify_protocol(root)
    ids = (root / IDS).read_text().splitlines()
    if len(ids) != 1090 or len(set(ids)) != 1090 or ids != protocol['eligible_patient_ids']:
        raise ValueError('Patient lock mismatch')
    return ids


def valid_uid(value):
    return isinstance(value, str) and len(value) <= 64 and re.fullmatch(r'(0|[1-9]\d*)(\.(0|[1-9]\d*))+', value) is not None


def reconcile(ids, selected):
    """Reject extra patients and incompatible series; do not silently discard them."""
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicated locked PatientIDs')
    found = {r['PatientID'] for r in selected}
    counts = Counter(r['SeriesInstanceUID'] for r in selected)
    studies, series = defaultdict(set), defaultdict(set)
    invalid = []
    for row in selected:
        patient = row['PatientID']
        studies[patient].add(row['StudyInstanceUID'])
        series[patient].add(row['SeriesInstanceUID'])
        if (row['Collection'] != 'CMMD' or row['Modality'] != 'MG'
                or not valid_uid(row['StudyInstanceUID']) or not valid_uid(row['SeriesInstanceUID'])
                or type(row['ImageCount']) is not int or row['ImageCount'] <= 0
                or (row['FileSize'] is not None and (type(row['FileSize']) is not int or row['FileSize'] <= 0))):
            invalid.append(row['SeriesInstanceUID'])
    missing, extra = sorted(set(ids) - found), sorted(found - set(ids))
    duplicates = sorted(uid for uid, count in counts.items() if count > 1)
    sizes_known = all(r['FileSize'] is not None for r in selected)
    total_bytes = sum(r['FileSize'] for r in selected) if sizes_known else None
    return dict(status='FAIL' if missing or extra or duplicates or invalid else 'PASS',
                requested_patients=len(ids), found_patients=len(found), missing_patients=missing, extra_patients=extra,
                duplicate_series=duplicates, invalid_series=invalid,
                multiple_studies={p: sorted(v) for p, v in sorted(studies.items()) if len(v) > 1},
                multiple_series={p: sorted(v) for p, v in sorted(series.items()) if len(v) > 1},
                series_count=len(counts), dicom_objects=sum(r['ImageCount'] for r in selected),
                estimated_bytes=total_bytes, estimated_gb=total_bytes / 1e9 if total_bytes is not None else None)


def select_catalogue(ids, catalogue):
    """Select by locked PatientID only, then verify collection and modality."""
    wanted = set(ids)
    return sorted(({key: r.get(key) for key in FIELDS} for r in catalogue if r.get('PatientID') in wanted),
                  key=lambda r: (r['PatientID'], r['SeriesInstanceUID']))


def parse_manifest(content):
    """Strict manifestVersion=3.0 parser following official NBIA API output."""
    lines = content.decode('utf-8').splitlines()
    marker = 'ListOfSeriesToDownload='
    if lines.count(marker) != 1:
        raise ValueError('Missing/duplicated manifest series marker')
    index = lines.index(marker)
    headers = {}
    for line in lines[:index]:
        key, separator, value = line.partition('=')
        if not separator or key in headers:
            raise ValueError('Invalid/duplicated manifest header')
        headers[key] = value
    expected = {'downloadServerUrl', 'includeAnnotation', 'noOfrRetry', 'databasketId', 'manifestVersion'}
    if set(headers) != expected:
        raise ValueError('Unexpected manifest headers')
    allowed_servers = {f'https://{host}.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet'
                       for host in ('nbia', 'services')}
    if (headers['downloadServerUrl'] not in allowed_servers or headers['includeAnnotation'] != 'false'
            or headers['manifestVersion'] != '3.0' or not headers['noOfrRetry'].isdigit()
            or not re.fullmatch(r'manifest-\d+\.tcia', headers['databasketId'])):
        raise ValueError('Unsupported or unsafe NBIA manifest header')
    uids = lines[index + 1:]
    if not uids or any(not valid_uid(uid) for uid in uids) or len(set(uids)) != len(uids):
        raise ValueError('Invalid/duplicate manifest SeriesUID')
    return headers, uids


def verify_manifest(ids, selected, content):
    result = reconcile(ids, selected)
    headers, uids = parse_manifest(content)
    expected = {r['SeriesInstanceUID'] for r in selected}
    result.update(manifest_missing_series=sorted(expected - set(uids)),
                  manifest_extra_series=sorted(set(uids) - expected),
                  manifest_sha256=hashlib.sha256(content).hexdigest(),
                  series_uid_list_sha256=hashlib.sha256(('\n'.join(sorted(expected)) + '\n').encode()).hexdigest(),
                  series_uid_hash_encoding='UTF-8, unique sorted UIDs, LF separator and final LF',
                  manifest_headers=headers, manifest_series_count=len(uids))
    if result['manifest_missing_series'] or result['manifest_extra_series']:
        result['status'] = 'FAIL'
    return result
