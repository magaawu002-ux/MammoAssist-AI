"""Inspect/audit clinical XLSX only. No downloads, pixels, model or evaluation."""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import re
from .cmmd_metadata import inspect_workbook, text, workbook_structure
from .cmmd_cohort import build_cohort
from .cmmd_provenance import EVIDENCE, ROOT, sha256, write_new, isolated_path


def csv_text(rows):
    stream = io.StringIO(newline='')
    fields = ['worksheet', 'source_row', 'id1', 'id2', 'patient_id', 'laterality', 'label', 'abnormality', 'reasons']
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row, reasons='; '.join(row['reasons'])))
    return stream.getvalue()


def audit(path, mapping_document=None, root=ROOT):
    # Missing metadata causes no directories, evidence or numerical claims.
    sheets = inspect_workbook(path)
    document = mapping_document or {}
    eligible, excluded = build_cohort(sheets, document.get('columns', {}), document.get('side_specific_evidence', {}))
    records = eligible + excluded
    mapped = [r for r in records if r['patient_id']]
    schema = {}
    structure = workbook_structure(path)
    for name, frame in sheets.items():
        schema[name] = {**structure[name], 'columns': list(frame.columns), 'source_rows': len(frame),
                        'missing_values': {str(c): int(frame[c].map(text).eq('').sum()) for c in frame.columns},
                        'value_counts': {str(c): dict(Counter(frame[c].map(text))) for c in frame.columns}}
    summary = dict(status='COHORT_REQUIRES_DICOM_VERIFICATION' if eligible else 'STOP_NO_DEFENSIBLE_MASS_COHORT',
                   spreadsheet=Path(path).name, spreadsheet_sha256=sha256(path), worksheets=schema,
                   total_source_rows=sum(len(f) for f in sheets.values()), mapping=document,
                   semantic_findings=document.get('semantic_findings', {}),
                   unique_mapped_patients=len({r['patient_id'] for r in mapped}) if mapped else None,
                   identifier_patterns=dict(Counter(re.sub(r'\d', '#', r['patient_id']) for r in mapped)),
                   d1_d2_source_row_counts=dict(Counter(r['patient_id'][:2] for r in mapped)),
                   classification_source_row_counts=dict(Counter(str(r['label']) for r in records)),
                   abnormality_source_row_counts=dict(Counter(r['abnormality'] for r in records)),
                   laterality_source_row_counts=dict(Counter(r['laterality'] for r in records)),
                   eligible_breast_sides=len(eligible), eligible_benign_sides=sum(r['label'] == 0 for r in eligible),
                   eligible_malignant_sides=sum(r['label'] == 1 for r in eligible), excluded_source_rows=len(excluded),
                   unique_eligible_patients=len({r['patient_id'] for r in eligible}),
                   eligible_d1_d2_side_counts=dict(Counter(r['patient_id'][:2] for r in eligible)),
                   unique_patient_d1_d2_counts=dict(Counter(p[:2] for p in {r['patient_id'] for r in mapped})),
                   repeated_patient_ids={p: n for p, n in Counter(r['patient_id'] for r in mapped).items() if n > 1},
                   duplicate_patient_side_rows=sum(n for n in Counter((r['patient_id'], r['laterality']) for r in mapped).values() if n > 1),
                   primary_exclusion_counts=dict(Counter(r['reasons'][0] for r in excluded)),
                   exclusion_reasons=dict(Counter(reason for r in excluded for reason in r['reasons'])),
                   manual_clarification=[{'worksheet': r['worksheet'], 'source_row': r['source_row'], 'patient_id': r['patient_id'], 'reasons': r['reasons']}
                                         for r in excluded if any(reason not in ('calcification_only', 'mixed_mass_and_calcification_excluded') for reason in r['reasons'])],
                   note='Counts describe clinical records only; all image matches await DICOM verification. No contralateral labels inferred.')
    report = '# CMMD metadata audit\n\n' + summary['status'] + '\n\n'
    report += 'No external inference has been performed. No threshold adjustment is permitted.\n\n'
    report += f"Workbook: `{summary['spreadsheet']}`\n\nSHA-256: `{summary['spreadsheet_sha256']}`\n\n"
    for key in ('total_source_rows', 'unique_mapped_patients', 'classification_source_row_counts',
                'abnormality_source_row_counts', 'laterality_source_row_counts', 'd1_d2_source_row_counts',
                'unique_patient_d1_d2_counts', 'eligible_breast_sides', 'eligible_benign_sides',
                'eligible_malignant_sides', 'unique_eligible_patients', 'excluded_source_rows',
                'exclusion_reasons', 'primary_exclusion_counts', 'duplicate_patient_side_rows'):
        report += f'- **{key}**: {summary[key]}\n'
    report += '\nCounts of classifications and abnormalities are source breast-side rows, not patient-level diagnoses.\n'
    report += 'Exclusion reason counts overlap; primary reasons are mutually exclusive. No image count is inferred from the spreadsheet number column.\n\n'
    for name, details in schema.items():
        report += f"## Worksheet {name}\n\nColumns (exact order): `{details['columns']}`\n\n"
        for key in ('source_rows', 'native_cell_types', 'missing_values', 'exact_duplicate_rows_beyond_first'):
            report += f'- {key}: {details[key]}\n'
    report += '\n## Label and identifier semantics\n\n' + json.dumps(document.get('semantic_findings', {}), indent=2) + '\n'
    report += '\n## Cases needing clarification\n\n```json\n' + json.dumps(summary['manual_clarification'], indent=2) + '\n```\n'
    report += '\nAll raw column distributions and repeated patient IDs are retained in `cmmd_metadata_summary.json`. Row-level exclusions are in `cmmd_exclusions.csv`.\n'
    outputs = {'cmmd_metadata_summary.json': json.dumps(summary, indent=2) + '\n',
               'cmmd_metadata_audit.md': report, 'cmmd_eligible_sides.csv': csv_text(eligible),
               'cmmd_exclusions.csv': csv_text(excluded)}
    for name in outputs:
        if isolated_path(f'{EVIDENCE}/{name}', root, evidence=True).exists():
            raise FileExistsError('Audit evidence already exists; preserve and review it before a new audit')
    for name, content in outputs.items():
        write_new(f'{EVIDENCE}/{name}', content, root, evidence=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xlsx', type=Path)
    parser.add_argument('--mapping', type=Path, help='Reviewed JSON column mapping; omit for schema-only STOP audit')
    args = parser.parse_args()
    result = audit(args.xlsx, json.loads(args.mapping.read_text()) if args.mapping else None)
    print(result['status'])
    return 0 if result['status'] == 'COHORT_REQUIRES_DICOM_VERIFICATION' else 2


if __name__ == '__main__':
    raise SystemExit(main())
