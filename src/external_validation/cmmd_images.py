"""Future full-mammogram mapping interface; never guess from filenames."""
from pathlib import Path
from .cmmd_metadata import normalize_side


def verify_header(header, eligible_sides, *, full_mammogram_evidence):
    """A header match is necessary, not sufficient, for inclusion.

    Full-image provenance must be independently reviewed: MG tags alone cannot
    distinguish a crop. Pixel duplicate/identity review is a later release gate.
    """
    if not full_mammogram_evidence or not str(full_mammogram_evidence).strip():
        raise ValueError('Full mammogram provenance required')
    patient = str(getattr(header, 'PatientID', '')).strip()
    side = normalize_side(getattr(header, 'ImageLaterality', ''))
    series_side = normalize_side(getattr(header, 'Laterality', ''))
    if side is None or (series_side and series_side != side):
        raise ValueError('Missing/conflicting DICOM laterality')
    if getattr(header, 'Modality', '') != 'MG' or getattr(header, 'ViewPosition', '') not in ('CC', 'MLO'):
        raise ValueError('Require MG modality and verified CC/MLO view')
    if int(getattr(header, 'NumberOfFrames', 1)) != 1:
        raise ValueError('Multiframe input requires separate harmonisation review')
    matches = [r for r in eligible_sides if r['patient_id'] == patient and r['laterality'] == side and not r['reasons']]
    if len(matches) != 1:
        raise ValueError('Unlabelled contralateral side or unresolved clinical match')
    result = dict(patient_id=patient, laterality=side, view=str(header.ViewPosition), label=matches[0]['label'],
                  worksheet=matches[0]['worksheet'], source_row=matches[0]['source_row'],
                  full_mammogram_evidence=full_mammogram_evidence)
    for tag in ('StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'):
        value = str(getattr(header, tag, '')).strip()
        if not value:
            raise ValueError(f'Missing DICOM {tag}')
        result[tag] = value
    return result


def read_header(path, eligible_sides, *, image_root, full_mammogram_evidence):
    import pydicom
    from .cmmd_provenance import sha256
    path = Path(path).resolve()
    if not path.is_relative_to(Path(image_root).resolve()) or not path.is_file():
        raise ValueError('DICOM must exist beneath the CMMD image root')
    result = verify_header(pydicom.dcmread(path, stop_before_pixels=True), eligible_sides,
                           full_mammogram_evidence=full_mammogram_evidence)
    return dict(result, path=str(path), dicom_sha256=sha256(path))
