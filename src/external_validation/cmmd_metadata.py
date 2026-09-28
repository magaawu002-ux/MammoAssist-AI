"""Read actual worksheets first; clinical semantics always require explicit mapping."""
from pathlib import Path
import re
import pandas as pd


def workbook_structure(path):
    """Inspect native XLSX cell types without coercion or filling merged cells."""
    from collections import Counter
    import openpyxl
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    result = {}
    try:
        for sheet in book:
            rows = list(sheet.values)
            headers = list(rows[0]) if rows else []
            if any(h is None for h in headers) or len(set(headers)) != len(headers):
                raise ValueError('Ambiguous or missing XLSX column headings')
            result[sheet.title] = {
                'native_cell_types': {str(h): dict(Counter(type(r[i]).__name__ for r in rows[1:]))
                                      for i, h in enumerate(headers)},
                'exact_duplicate_rows_beyond_first': len(rows[1:]) - len(set(rows[1:]))}
    finally:
        book.close()
    return result


def text(value):
    return '' if value is None or pd.isna(value) else str(value).strip()


def inspect_workbook(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('WAITING_FOR_CMMD_CLINICAL_METADATA')
    if path.suffix.lower() != '.xlsx':
        raise ValueError('Expected the original clinical XLSX')
    # Do not forward-fill identifiers, sides, labels or merged cells.
    workbook_structure(path)  # Reject ambiguous headers before pandas can rename them.
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False, engine='openpyxl')
    return {name: frame.fillna('') for name, frame in sheets.items()}


def imaging_id(id1, id2):
    """Prefer ID2 whenever supplied, never fall back from an invalid ID2."""
    value = text(id2) or text(id1)
    if not re.fullmatch(r'D[12]-\d{4}', value):
        return None
    return value


def normalize_label(value):
    return {'benign': 0, 'malignant': 1}.get(text(value).casefold())


def normalize_side(value):
    return {'l': 'L', 'left': 'L', 'r': 'R', 'right': 'R'}.get(text(value).casefold())


def normalize_type(value):
    # Mixed mass/calcification is deliberately excluded from the primary cohort.
    return {'mass': 'mass', 'masses': 'mass', 'both': 'mixed_mass_and_calcification', 'calcification': 'calcification',
            'calcifications': 'calcification', 'calcification only': 'calcification'}.get(text(value).casefold())


def validate_mapping(frame, mapping):
    required = {'id1', 'id2', 'laterality', 'classification', 'abnormality'}
    if set(mapping) != required or not any(mapping[k] for k in ('id1', 'id2')):
        raise ValueError('Explicit ID and side-specific clinical column mapping required')
    columns = [value for value in mapping.values() if value is not None]
    if len(set(columns)) != len(columns) or any(value not in frame.columns for value in columns):
        raise ValueError('Mapping must reference distinct actual worksheet columns')
    if any(mapping[k] is None for k in ('laterality', 'classification', 'abnormality')):
        raise ValueError('STOP: metadata cannot support a defensible MASS-only side-specific cohort')
