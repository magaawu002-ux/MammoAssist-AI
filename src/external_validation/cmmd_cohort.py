"""Prospective conservative MASS-only eligibility; never infer contralateral labels."""
from collections import Counter, defaultdict
from .cmmd_metadata import imaging_id, normalize_label, normalize_side, normalize_type, text, validate_mapping

# TCIA reports unresolved identical pixels across these participants. Exclude
# entire affected participants until independent identity review resolves them.
KNOWN_DUPLICATES = {'D1-0202', 'D2-0284', 'D1-0808', 'D1-1292'}


def build_cohort(sheets, mappings, side_specific_evidence):
    records = []
    for sheet, frame in sheets.items():
        mapping = mappings.get(sheet)
        error = None
        try:
            validate_mapping(frame, mapping or {})
            if not text(side_specific_evidence.get(sheet)):
                raise ValueError('Manual clarification required: label-to-side semantics unconfirmed')
        except ValueError as exc:
            error = str(exc)
        for index, source in frame.iterrows():
            row = dict(worksheet=sheet, source_row=int(index) + 2,
                       patient_id='', laterality='', label='', abnormality='', reasons=[])
            if error:
                row['reasons'].append(error)
            else:
                get = lambda key: text(source[mapping[key]]) if mapping[key] is not None else ''
                row.update(id1=get('id1'), id2=get('id2'),
                           patient_id=imaging_id(get('id1'), get('id2')) or '',
                           laterality=normalize_side(get('laterality')) or '',
                           label=normalize_label(get('classification')),
                           abnormality=normalize_type(get('abnormality')) or '')
                for invalid, reason in (
                    (not row['patient_id'], 'unsafe_imaging_patient_id'),
                    (not row['laterality'], 'missing_or_ambiguous_laterality'),
                    (row['label'] is None, 'missing_or_ambiguous_classification'),
                    (row['abnormality'] == 'calcification', 'calcification_only'),
                    (row['abnormality'] == 'mixed_mass_and_calcification', 'mixed_mass_and_calcification_excluded'),
                    (row['abnormality'] not in ('mass', 'calcification', 'mixed_mass_and_calcification'), 'missing_or_ambiguous_abnormality'),
                    (row['patient_id'] in KNOWN_DUPLICATES, 'TCIA_unresolved_duplicate_pixels')):
                    if invalid:
                        row['reasons'].append(reason)
            records.append(row)
    counts = Counter((r['patient_id'], r['laterality']) for r in records if r['patient_id'] and r['laterality'])
    aliases = defaultdict(set)
    d2_malignant_sides = defaultdict(set)
    for row in records:
        if row['patient_id'] and row.get('id1'):
            aliases[row['id1']].add(row['patient_id'])
        if row['patient_id'].startswith('D2-') and row['label'] == 1 and row['laterality']:
            d2_malignant_sides[row['patient_id']].add(row['laterality'])
    for row in records:
        if counts[(row['patient_id'], row['laterality'])] > 1:
            row['reasons'].append('unresolved_duplicate_breast_side')
        if len(aliases[row.get('id1', '')]) > 1:
            row['reasons'].append('ambiguous_ID1_to_imaging_ID_mapping')
        if len(d2_malignant_sides[row['patient_id']]) > 1:
            row['reasons'].append('D2_bilateral_malignancy_requires_clarification')
    records.sort(key=lambda r: (r['patient_id'], r['laterality'], r['worksheet'], r['source_row']))
    return [r for r in records if not r['reasons']], [r for r in records if r['reasons']]
