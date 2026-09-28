"""Versioned, pre-inference CMMD format-conversion eligibility amendment.

No model or prediction imports. The original strict audit remains immutable.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pydicom

from . import cmmd_image_audit as old
from .cmmd_provenance import ROOT, EVIDENCE, CHECKPOINT, CHECKPOINT_SHA256, THRESHOLD, RESULTS, sha256, write_new

VERSION = 'cmmd_full_mammogram_conversion_v2'
RULE = 'ALLOW_DOCUMENTED_CMMD_FORMAT_CONVERSION'
READY = 'READY_FOR_CMMD_EXTERNAL_INFERENCE'
PROTOCOL = f'{EVIDENCE}/cmmd_external_protocol_v2.json'
COHORT = f'{old.MANIFESTS}/cmmd_external_image_cohort_v2.csv'
SUMMARY = f'{EVIDENCE}/cmmd_image_cohort_summary_v2.json'
AUDIT = f'{EVIDENCE}/cmmd_image_cohort_audit_v2.md'
GEOMETRY = f'{EVIDENCE}/cmmd_format_conversion_objects.csv'
REVIEW = f'{EVIDENCE}/cmmd_format_conversion_review.md'
SOURCES = f'{EVIDENCE}/cmmd_format_conversion_sources.json'
SOURCE = 'src/external_validation/cmmd_conversion.py'
PREDICT_SOURCE = 'src/external_validation/cmmd_predict.py'
OLD_LOCK_SHA256 = 'a3a631b6f10da447cf4fd794b8678ea42eb4e09b183086e8f4b00b832ef457fb'
TAGS = ['SOPClassUID','Rows','Columns','BitsAllocated','BitsStored','HighBit','PixelRepresentation',
        'PhotometricInterpretation','SamplesPerPixel','PlanarConfiguration','ViewPosition','ImageLaterality',
        'Laterality','SeriesDescription','StudyDescription','ImageType','PresentationIntentType','PresentationLUTShape',
        'PixelIntensityRelationship','PixelIntensityRelationshipSign','BurnedInAnnotation','PartialView',
        'DerivationDescription','LossyImageCompression','ImagerPixelSpacing','PatientOrientation']
PROHIBITED_TAGS = ['DerivationCodeSequence','SourceImageSequence','GraphicAnnotationSequence',
                   'DisplayedAreaSelectionSequence','ShutterShape','PresentationLUTSequence','VOILUTSequence',
                   'SegmentationType','ROIContourSequence','StructureSetROISequence','IconImageSequence']


def baseline(root):
    if sha256(root / old.IMAGE_LOCK) != OLD_LOCK_SHA256:
        raise ValueError('Original strict image lock changed')
    return old.verify_image_lock(root, require_ready=False)


def conversion_header(ds, sides, catalogue, original, source_sha256, *, dataset='CMMD'):
    """Known official source identity + full mammography metadata; no pixel/model judgement."""
    mapped = old.header_record(ds, sides, {p for p,s in sides})
    reasons = set(filter(None, mapped['reasons'].split(';')))
    reasons.discard('derived_or_unspecified_image_type')
    cat = catalogue.get(mapped['series_uid'])
    if dataset != 'CMMD' or not cat or cat['Collection'] != 'CMMD' or cat['Modality'] != 'MG':
        reasons.add('not_verified_cmmd_series')
    elif cat['PatientID'] != mapped['patient_id'] or cat['StudyInstanceUID'] != mapped['study_uid']:
        reasons.add('catalogue_identity_mismatch')
    if not original or source_sha256 != original.get('sha256'):
        reasons.add('not_original_downloaded_object')
    elif any(str(mapped[k]) != str(original.get(k,'')) for k in ('patient_id','study_uid','series_uid','sop_uid','laterality','view')):
        reasons.add('original_header_identity_mismatch')
    if list(getattr(ds,'ImageType',[])) != ['DERIVED','PRIMARY']:
        reasons.add('undocumented_image_type')
    if old.text(ds,'SOPClassUID') != '1.2.840.10008.5.1.4.1.1.1.2':
        reasons.add('not_full_mammography_presentation_sop')
    if (getattr(ds,'Rows',None),getattr(ds,'Columns',None)) != (2294,1914):
        reasons.add('non_collection_dimensions')
    bits = getattr(ds,'BitsAllocated',0)
    if bits not in (8,16) or getattr(ds,'BitsStored',0) != bits or getattr(ds,'HighBit',-1) != bits-1:
        reasons.add('unsupported_bit_encoding')
    expected = dict(PhotometricInterpretation='MONOCHROME2',SamplesPerPixel='1',PixelRepresentation='0',
                    PresentationIntentType='FOR PRESENTATION',PresentationLUTShape='IDENTITY',
                    BurnedInAnnotation='NO',LossyImageCompression='00')
    if any(old.text(ds,k) != v for k,v in expected.items()):
        reasons.add('incompatible_presentation_encoding')
    if old.text(ds,'PlanarConfiguration'):
        reasons.add('unexpected_planar_configuration')
    if old.text(getattr(ds,'file_meta',ds),'TransferSyntaxUID') != '1.2.840.10008.1.2.1':
        reasons.add('unreviewed_transfer_syntax')
    spacing = getattr(ds,'ImagerPixelSpacing',[])
    if len(spacing)!=2 or any(abs(float(v)-0.094090909)>1e-8 for v in spacing):
        reasons.add('non_collection_detector_geometry')
    if any(getattr(ds,k,None) for k in PROHIBITED_TAGS) or any(0x6000<=e.tag.group<=0x60ff for e in ds):
        reasons.add('derived_processing_or_overlay_evidence')
    descriptions = ' '.join(old.text(ds,k) for k in ['SeriesDescription','StudyDescription','DerivationDescription',
                                                   'AcquisitionDeviceProcessingDescription'])
    if re.search(r'crop|roi|partial|screenshot|secondary|annotat|segment|composite|magnif|spot', descriptions, re.I):
        reasons.add('suspicious_description')
    elif descriptions.strip():
        reasons.add('unreviewed_description')
    mapped['reasons'] = ';'.join(sorted(reasons))
    return mapped


def pixel_geometry(ds):
    """Label-blind field/encoding checks; not segmentation or pathology localization."""
    if 'PixelData' not in ds:
        raise ValueError('missing_pixel_data')
    pixels = ds.pixel_array
    if pixels.shape != (2294,1914) or pixels.dtype.kind != 'u':
        raise ValueError('unexpected_pixel_geometry')
    expected = 2294*1914*int(ds.BitsAllocated)//8
    if len(ds.PixelData) != expected:
        raise ValueError('unexpected_pixel_payload_length')
    low, high = int(pixels.min()), int(pixels.max())
    if low == high: raise ValueError('constant_image')
    mask = pixels > low
    ys, xs = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    height, width = int(ys[-1]-ys[0]+1), int(xs[-1]-xs[0]+1)
    fraction = float(mask.mean())
    # Same conservative field-span guards as canonical cropping; no label-based tuning.
    if fraction < .01 or height/2294 < .1 or width/1914 < .1:
        raise ValueError('sparse_or_small_field_requires_review')
    if len(np.unique(pixels[::8,::8])) < 16:
        raise ValueError('posterized_or_annotation_like_pixels')
    return dict(pixel_data_present=True,pixel_bytes=len(ds.PixelData),pixel_dtype=str(pixels.dtype),
                pixel_min=low,pixel_max=high,foreground_fraction=fraction,
                field_height=height,field_width=width,field_y0=int(ys[0]),field_x0=int(xs[0]),
                pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())


def validate_pairs(rows, sides):
    groups=defaultdict(list)
    for row in rows: groups[(row['patient_id'],row['laterality'])].append(row['view'])
    missing=sorted(set(sides)-set(groups))
    unusual=[{'patient_id':k[0],'laterality':k[1],'views':v} for k,v in sorted(groups.items())
             if sorted(v)!=['CC','MLO']]
    return dict(missing_sides=[list(k) for k in missing],unexpected_or_missing_views=unusual,
                complete_pairs=sum(sorted(v)==['CC','MLO'] for v in groups.values()))


def no_external_predictions(root):
    meaningful=[p for p in (root/RESULTS).rglob('*') if p.is_file() and p.name!='.gitkeep']
    if meaningful: raise ValueError('External result files already exist; pre-inference assertion cannot be made')


def inspect(root=ROOT):
    root=Path(root).resolve();baseline(root);no_external_predictions(root)
    for p in (GEOMETRY,SUMMARY):
        if (root/p).exists(): raise FileExistsError(p)
    sources=json.loads((root/SOURCES).read_text())
    if sources['rule']!=RULE or not sources['sources']: raise ValueError('Missing authoritative source evidence')
    sides=old.clinical_map(old.csv_read(root/old.CLINICAL))
    catalogue={r['SeriesInstanceUID']:r for r in old.csv_read(root/old.CATALOGUE)}
    originals={r['sop_uid']:r for r in old.csv_read(root/f'{EVIDENCE}/cmmd_dicom_headers.csv')}
    processed=old.csv_read(root/old.PROCESSED)
    inspected=[]; distributions={k:Counter() for k in TAGS}
    for i,row in enumerate(processed):
        path=old._safe_local(root,row['path'],old.IMAGES)
        digest=sha256(path)
        ds=pydicom.dcmread(path)
        mapped=conversion_header(ds,sides,catalogue,originals.get(str(ds.SOPInstanceUID)),digest)
        item=dict(row,**{k:v for k,v in mapped.items() if k not in row})
        item['prior_strict_evaluation_eligible']=item.pop('evaluation_eligible')
        item.update(reasons=mapped['reasons'],sha256=digest,bytes=path.stat().st_size)
        for key in TAGS:
            value=str(getattr(ds,key,'MISSING'));distributions[key][value]+=1
            item[key]=value
        if row['preprocessing_status']!='PASS': item['reasons']+=';prior_preprocessing_not_passed'
        try: item.update(pixel_geometry(ds))
        except Exception as exc: item['reasons']+=';'+(str(exc) if isinstance(exc,ValueError) else type(exc).__name__)
        item['reasons']=';'.join(sorted(set(filter(None,item['reasons'].split(';')))))
        inspected.append(item)
        if (i+1)%300==0:print(f'Full-mammogram checks {i+1}/{len(processed)}',flush=True)
    pixel_counts=Counter(r.get('pixel_sha256','') for r in inspected)
    sop_counts=Counter(r['sop_uid'] for r in inspected)
    for row in inspected:
        if row.get('pixel_sha256') and pixel_counts[row['pixel_sha256']]>1: row['reasons']+=';duplicate_pixel_array'
        if sop_counts[row['sop_uid']]>1: row['reasons']+=';duplicate_sop_instance'
    eligible=[r for r in inspected if not r['reasons']]
    sizes=np.array([r['bytes'] for r in inspected],dtype=np.int64)
    original_rows=list(originals.values())
    contralateral=sum('contralateral_side_not_in_locked_cohort' in r['reasons'] for r in original_rows)
    summary=dict(version=VERSION,rule=RULE,status='REVIEW_PENDING',candidates=len(inspected),eligible=old.counts(eligible),
        contralateral_exclusions=contralateral,other_exclusions=dict(Counter(reason for r in inspected for reason in r['reasons'].split(';') if reason)),
        suspicious_objects=[{k:r.get(k,'') for k in ['patient_id','sop_uid','path','reasons']} for r in inspected if r['reasons']],
        distributions={k:dict(sorted(v.items())) for k,v in distributions.items()},pair_check=validate_pairs(eligible,sides),
        file_sizes=dict(min=int(sizes.min()),median=float(np.median(sizes)),max=int(sizes.max()),total=int(sizes.sum()),
                        distribution=dict(sorted(Counter(int(n) for n in sizes).items()))),
        preprocessing=dict(unchanged=True,passed=sum(r['preprocessing_status']=='PASS' for r in inspected),
            source_manifest=old.PROCESSED,source_sha256=sha256(root/old.PROCESSED),
            rationale='Reuse prior full-cohort canonical validation only for identical source SHA256 and frozen implementation/configuration.',
            version='canonical_v1'),predictions_generated=0,amended_before_inference=True,
        inference_not_executed=True,patient_label_counts_overlap=True,
        field_geometry_ranges={key:[min(r[key] for r in inspected if key in r),max(r[key] for r in inspected if key in r)]
                               for key in ['foreground_fraction','field_height','field_width']})
    fields=list(dict.fromkeys(k for r in inspected for k in r))
    write_new(GEOMETRY,old.csv_bytes(inspected,fields).decode(),root,evidence=True)
    write_new(SUMMARY,json.dumps(summary,indent=2,sort_keys=True)+'\n',root,evidence=True)
    return summary


def freeze(root=ROOT):
    root=Path(root).resolve();baseline(root);no_external_predictions(root)
    rows=old.csv_read(root/GEOMETRY);summary=json.loads((root/SUMMARY).read_text())
    review=json.loads((root/f'{EVIDENCE}/cmmd_conversion_visual_review.json').read_text())
    if review['status']!='PASS' or review['geometry_csv_sha256']!=sha256(root/GEOMETRY):
        raise ValueError('Independent visual field review missing or mismatched')
    eligible=[r for r in rows if not r['reasons']]
    if not eligible or summary['suspicious_objects'] or summary['pair_check']['missing_sides'] or summary['pair_check']['unexpected_or_missing_views']:
        raise ValueError('Suspicious/missing candidate mappings require review before freezing')
    if old.counts(eligible)!=summary['eligible']: raise ValueError('Counts differ from reviewed cohort')
    csv_content=old.csv_bytes(eligible,old.COHORT_FIELDS+['pixel_sha256','array_sha256'])
    for path in (COHORT,PROTOCOL,AUDIT):
        if (root/path).exists():raise FileExistsError(path)
    summary['status']=READY
    summary['visual_review']=f'{EVIDENCE}/cmmd_conversion_visual_review.json'
    with (root/SUMMARY).open('w',encoding='utf-8',newline='') as stream:
        stream.write(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    report=(f'# CMMD image cohort v2: {READY}\n\n'
        f'Eligibility rule: `{VERSION}` / `{RULE}`. This pre-inference amendment supplements '
        'the original clinical and strict image protocols; their bytes remain unchanged.\n\n'
        f'Eligible image/side/patient counts: {summary["eligible"]["images"]} / {summary["eligible"]["sides"]} / '
        f'{summary["eligible"]["patients"]}. Contralateral exclusions: {summary["contralateral_exclusions"]}. '
        f'Other exclusions: {summary["other_exclusions"]}.\n\n'
        'Every object passes official-source identity, full mammography SOP/header, geometry, '
        'pixel-content and original canonical preprocessing checks. Each eligible side has exactly CC and MLO. '
        'No duplicate SOP or pixel array, partial-view marker, overlay, suspicious description or incompatible derived '
        'output was found. Missing ViewPosition is resolved only by explicit standard ViewCodeSequence.\n\n'
        'The original diagnostic preprocessing validation is reused only because each raw source SHA256, '
        'preprocessing source/configuration and output array hash remains bound. No normalization or adapter changed.\n\n'
        'See cmmd_format_conversion_review.md for source citations, pixel-depth findings and evidential limits, '
        'cmmd_format_conversion_objects.csv for per-object checks and cmmd_external_protocol_v2.json for the final lock. '
        'No predictions or external metrics exist at amendment time.\n')
    write_new(AUDIT,report,root,evidence=True)
    paths=[old.IMAGE_LOCK,SOURCES,GEOMETRY,SUMMARY,AUDIT,REVIEW,f'{EVIDENCE}/cmmd_conversion_visual_review.json',SOURCE,PREDICT_SOURCE,
           old.CLINICAL,old.PROCESSED,*old.PREPROCESSING]
    bound={p:sha256(root/p) for p in paths};bound[COHORT]=hashlib.sha256(csv_content).hexdigest()
    protocol=dict(version=VERSION,rule=RULE,status=READY,dataset='CMMD',purpose='external_validation',
        supersedes_image_stage=old.IMAGE_LOCK,clinical_protocol_unchanged=old.BASELINE_HASHES,
        clinical_cohort_sha256=sha256(root/old.CLINICAL),image_cohort_sha256=bound[COHORT],
        sources=json.loads((root/SOURCES).read_text())['sources'],documented_conversion_exception=True,
        selected_model='resnet50',checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_SHA256,classifier_threshold=THRESHOLD,
        preprocessing_version='canonical_v1',preprocessing_hashes={p:sha256(root/p) for p in old.PREPROCESSING},
        training_allowed=False,fine_tuning_allowed=False,calibration_allowed=False,model_selection_allowed=False,
        threshold_tuning_allowed=False,predictions_generated=0,amended_before_inference=True,performance_influenced_eligibility=False,
        counts=summary['eligible'],contralateral_exclusions=summary['contralateral_exclusions'],other_exclusions=summary['other_exclusions'],
        output_root=RESULTS,execution_requires_explicit_flag=True,bound_files=bound)
    write_new(COHORT,csv_content.decode(),root,manifests=True)
    write_new(PROTOCOL,json.dumps(protocol,indent=2,sort_keys=True)+'\n',root,evidence=True)
    return protocol


def verify(root=ROOT, *, check_pixels=True):
    root=Path(root).resolve();baseline(root)
    lock=json.loads((root/PROTOCOL).read_text())
    expected=dict(version=VERSION,rule=RULE,status=READY,dataset='CMMD',purpose='external_validation',
        selected_model='resnet50',checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_SHA256,classifier_threshold=THRESHOLD,
        documented_conversion_exception=True,training_allowed=False,fine_tuning_allowed=False,calibration_allowed=False,
        model_selection_allowed=False,threshold_tuning_allowed=False,predictions_generated=0,amended_before_inference=True,
        performance_influenced_eligibility=False,output_root=RESULTS,execution_requires_explicit_flag=True)
    for k,v in expected.items():
        if type(lock.get(k)) is not type(v) or lock[k]!=v: raise ValueError('Amended protocol changed: '+k)
    required={old.IMAGE_LOCK,old.CLINICAL,old.PROCESSED,COHORT,GEOMETRY,SUMMARY,AUDIT,REVIEW,SOURCES,SOURCE,PREDICT_SOURCE,
              f'{EVIDENCE}/cmmd_conversion_visual_review.json',*old.PREPROCESSING}
    if set(lock['bound_files'])!=required:raise ValueError('Required amendment bindings changed')
    for p,digest in lock['bound_files'].items():
        path=(root/p).resolve()
        if not path.is_relative_to(root) or sha256(path)!=digest:raise ValueError('Amendment binding changed: '+p)
    if lock['clinical_cohort_sha256']!=sha256(root/old.CLINICAL) or lock['image_cohort_sha256']!=sha256(root/COHORT):
        raise ValueError('Cohort digest changed')
    if lock['preprocessing_hashes']!={p:sha256(root/p) for p in old.PREPROCESSING}:
        raise ValueError('Preprocessing changed')
    rows=old.csv_read(root/COHORT)
    geometry={r['sop_uid']:r for r in old.csv_read(root/GEOMETRY) if not r['reasons']}
    if len(rows)!=len(geometry) or {r['sop_uid'] for r in rows}!=set(geometry):raise ValueError('Cohort membership changed')
    sides=old.clinical_map(old.csv_read(root/old.CLINICAL))
    cat={r['SeriesInstanceUID']:r for r in old.csv_read(root/old.CATALOGUE)}
    original={r['sop_uid']:r for r in old.csv_read(root/f'{EVIDENCE}/cmmd_dicom_headers.csv')}
    for r in rows:
        if any(r[k]!=geometry[r['sop_uid']][k] for k in r):raise ValueError('Cohort record changed')
        path=old._safe_local(root,r['path'],old.IMAGES)
        if check_pixels and sha256(path)!=r['sha256']:raise ValueError('DICOM changed')
        ds=pydicom.dcmread(path,stop_before_pixels=True)
        mapped=conversion_header(ds,sides,cat,original.get(r['sop_uid']),r['sha256'])
        if mapped['reasons'] or str(mapped['label'])!=r['label']:raise ValueError('Unsafe amended mapping')
    pairs=validate_pairs(rows,sides)
    if pairs['missing_sides'] or pairs['unexpected_or_missing_views'] or old.counts(rows)!=lock['counts']:
        raise ValueError('Pair/count integrity failed')
    return lock


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inspect','freeze','verify'])
    args=parser.parse_args()
    print(json.dumps({'inspect':inspect,'freeze':freeze,'verify':verify}[args.action](),indent=2))
