"""Reproducible ROI coordinate evidence; no model, scoring, or test development.

Run --geometry-all for descriptive metadata/mask geometry (test permitted only here).
Run --development for train/validation image correspondence and validation panels.
Neither command grants alignment approval or changes the localisation runner.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pydicom

from src.data.cbis_ddsm_metadata import write_csv
from src.data.provenance import sha256_file
from src.explainability.coordinate_checks import (candidate_mask, crop_consistency,
    evidence_screen, foreground_checks, grid_relation, mask_statistics, require_development)
from src.explainability.roi_audit import selected_images
from src.explainability.roi_mapping import annotation_role
from src.explainability.roi_masks import decode_mask
from src.preprocessing.dicom import decode_dicom
from src.preprocessing.manifests import local_path, read_rows, verify_full_reference
from src.utils.paths import PROJECT_ROOT

TAGS = ('PixelSpacing','ImagerPixelSpacing','NominalScannedPixelSpacing',
        'ImagePositionPatient','ImageOrientationPatient','FrameOfReferenceUID',
        'PixelAspectRatio','PatientOrientation','DetectorElementSpacing',
        'SharedFunctionalGroupsSequence','PerFrameFunctionalGroupsSequence')


def save_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def annotation_path(root, row):
    return local_path(root, 'data/raw/CBIS-DDSM/roi_cropped/'+row['path'], ('data','raw','CBIS-DDSM','roi_cropped'))


def mapped_rows(root, official):
    path=root/f'reports/evidence/real_roi_{official}_audit.csv'
    binding=json.loads((root/'reports/evidence/real_roi_collection_audit.json').read_text())
    if sha256_file(path)!=binding['evidence_sha256'][path.name]:
        raise ValueError('Previous mapping evidence hash changed')
    rows=read_rows(path)
    if any(r['mapping_status']!='unique' for r in rows):
        raise ValueError('Coordinate evidence requires unique mapping')
    return rows


def checked_mask(root, row):
    path=annotation_path(root,row)
    if sha256_file(path)!=row['mask_sha256']:
        raise ValueError('Mask changed since mapping audit')
    return decode_mask(path,row['SOPInstanceUID'])


def headers_and_image(root, image, *, pixels=False):
    path=local_path(root,image['dicom_relative_path'],('data','raw','CBIS-DDSM','images'))
    header=pydicom.dcmread(path,stop_before_pixels=True)
    verify_full_reference(image,header)
    geometry=json.loads(image['preprocessing_metadata_json'])
    if str(header.SOPInstanceUID)!=geometry['SOPInstanceUID']:
        raise ValueError('Full image differs from original preprocessing')
    if pixels:
        require_development(image['final_partition'])
        if image['official_partition']!='train':
            raise ValueError('Test full pixels forbidden for coordinate development')
        return header,decode_dicom(path),sha256_file(path)
    return header


def geometry_inventory(root):
    require_unlocked_evidence(root)
    output=root/'reports/evidence'
    images={r['canonical_image_id']:r for part in ('train','validation','test') for r in selected_images(root,part)[0]}
    result=[]
    for official in ('train','test'):
        for r in mapped_rows(root,official):
            if r['role']!='mask': continue
            image=images[r['canonical_image_id']]
            full=headers_and_image(root,image)  # No full-image/test image pixels.
            decoded=checked_mask(root,r)
            relation=grid_relation((int(full.Rows),int(full.Columns)),decoded.mask.shape)
            row={k:r[k] for k in ('canonical_image_id','record_id','abnormality_id','participant_id','final_partition','mask_sha256')}
            row.update(full_rows=int(full.Rows),full_columns=int(full.Columns),mask_rows=decoded.mask.shape[0],
                       mask_columns=decoded.mask.shape[1],**relation,**mask_statistics(decoded.mask),
                       full_series_description=str(getattr(full,'SeriesDescription','')),
                       mask_series_description=str(getattr(decoded.header,'SeriesDescription','')))
            for prefix,ds in (('full',full),('mask',decoded.header)):
                for tag in TAGS:
                    row[f'{prefix}_{tag}_present']=hasattr(ds,tag)
                    row[f'{prefix}_{tag}']=str(getattr(ds,tag,''))
            result.append(row)
            if len(result)%100==0: print('Geometry',len(result),flush=True)
    write_csv(output/'roi_alignment_geometry.csv',result)
    summary={'status':'DESCRIPTIVE_ONLY','test_usage':'Mask support and headers only; no test full/crop pixels, registration or outcomes',
             'proportional_definition':'abs(full_h/mask_h-full_w/mask_w) <= 1/mask_h+1/mask_w; integer-dimension rounding allowance, not proof',
             'counts':{p:dict(Counter(r['grid_relation'] for r in result if r['final_partition']==p)) for p in ('train','validation','test')},
             'dimension_patterns':[], 'anomalies':[],
             'geometry_csv_sha256':sha256_file(output/'roi_alignment_geometry.csv')}
    for part in ('train','validation','test'):
        patterns=Counter((r['full_rows'],r['full_columns'],r['mask_rows'],r['mask_columns'],r['grid_relation']) for r in result if r['final_partition']==part)
        summary['dimension_patterns'] += [dict(partition=part,full_rows=k[0],full_columns=k[1],mask_rows=k[2],mask_columns=k[3],grid_relation=k[4],count=v) for k,v in sorted(patterns.items())]
    summary['boundary_touching_mask_counts']={part:sum(r['bbox_y0']==0 or r['bbox_x0']==0 or r['bbox_y1']==r['mask_rows'] or r['bbox_x1']==r['mask_columns'] for r in result if r['final_partition']==part) for part in ('train','validation','test')}
    summary['boundary_touching_note']='Descriptive flag only; boundary contact alone neither invalidates a mask nor proves alignment. No case was excluded from geometry inventory.'
    summary['unequal_dimension_pattern_count']={part:len({(r['full_rows'],r['full_columns'],r['mask_rows'],r['mask_columns']) for r in result if r['final_partition']==part and not r['dimensions_equal']}) for part in ('train','validation','test')}
    summary['source_spatial_tag_presence_counts']={part:{k:sum(r[k] for r in result if r['final_partition']==part) for k in result[0] if k.endswith('_present')} for part in ('train','validation','test')}
    save_json(output/'roi_alignment_geometry_summary.json',summary)
    print('Geometry complete',summary['counts'],flush=True)


def summarise(rows):
    result={'n':len(rows),'screen_supported':sum(r['screen_supported'] for r in rows),
            'screen_failures':dict(Counter(r['screen_failure_reason'] for r in rows if not r['screen_supported']))}
    for key in ('nonbackground_fraction','otsu_largest_component_fraction','crop_ncc_at_proposed_origin','crop_ncc_local_best','negative_control_max_ncc'):
        values=[r[key] for r in rows if r.get(key) is not None]
        result[key]={'n':len(values),'min':float(min(values)),'median':float(np.median(values)),'max':float(max(values))} if values else {'n':0}
    return result


def review_selection(rows):
    """Deterministic validation-only strata coverage, plus all unequal-grid cases and failures."""
    validation=sorted((r for r in rows if r['final_partition']=='validation'),key=lambda r:r['record_id'])
    selected={r['record_id']:r for r in validation if r['grid_relation']!='equal'
              or not r.get('screen_supported',True) or r.get('otsu_largest_component_fraction',1)<.5}
    for key in ('pathology','view','laterality','multiple_lesions'):
        for value in sorted({r[key] for r in validation}):
            row=next(r for r in validation if r[key]==value and r['grid_relation']=='equal')
            selected[row['record_id']]=row
    # Include the worst three equal-grid matches rather than hiding failures.
    for row in sorted((r for r in validation if r['grid_relation']=='equal'),key=lambda r:r['crop_ncc_at_proposed_origin'] if r['crop_ncc_at_proposed_origin'] is not None else -2)[:3]:
        selected[row['record_id']]=row
    return list(selected.values())


def panels(root, rows, images, masks, crops):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    out=root/'reports/evidence/roi_alignment_review';out.mkdir(exist_ok=True)
    index=[];thumbnails=[]
    for number,row in enumerate(review_selection(rows),1):
        require_development(row['final_partition'])
        assert row['final_partition']=='validation'
        image=images[row['canonical_image_id']]
        _,decoded,_=headers_and_image(root,image,pixels=True)
        full=decoded.pixels
        mask=candidate_mask(checked_mask(root,masks[row['record_id']]).mask,full.shape)
        crop=decode_dicom(annotation_path(root,crops[row['record_id']])).pixels
        s=mask_statistics(mask);h,w=full.shape
        factor=min(1,1000/max(h,w));display=cv2.resize(full,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        small_mask=cv2.resize(mask.astype(np.uint8),(display.shape[1],display.shape[0]),interpolation=cv2.INTER_NEAREST)
        fig,axes=plt.subplots(1,4,figsize=(14,5))
        for ax in axes[:2]:ax.imshow(display,cmap='gray',extent=(0,w,h,0))
        axes[0].set_title('Original full mammogram')
        axes[1].contour(small_mask,levels=[.5],colors=['lime'],extent=(0,w,h,0),origin='upper',linewidths=.7)
        axes[1].add_patch(Rectangle((s['bbox_x0'],s['bbox_y0']),s['bbox_x1']-s['bbox_x0'],s['bbox_y1']-s['bbox_y0'],fill=False,color='yellow',linewidth=.8))
        axes[1].plot(s['centroid_x'],s['centroid_y'],'r+',markersize=5)
        axes[1].set_title('Candidate ROI; yellow bbox; red centroid')
        axes[2].imshow(crop,cmap='gray');axes[2].set_title('Supplied lesion crop')
        y,x=row['proposed_crop_y'],row['proposed_crop_x'];ch,cw=crop.shape
        patch=full[y:y+ch,x:x+cw]
        axes[3].imshow(patch,cmap='gray');axes[3].contour(mask[y:y+ch,x:x+cw],levels=[.5],colors=['lime'],linewidths=.7)
        score=row['crop_ncc_at_proposed_origin'];axes[3].set_title(f'Proposed full-image patch\nNCC={score:.5f}' if score is not None else 'Proposed patch: NCC undefined')
        for ax in axes:ax.axis('off')
        title=f"{number:02d} | {row['participant_id']} {row['laterality']} {row['view']} | lesion {row['abnormality_id']} | {row['pathology']} | {row['grid_relation']} | UNAPPROVED"
        fig.suptitle(title,fontsize=9);fig.tight_layout()
        name=f"{number:02d}_{row['canonical_image_id']}_lesion_{row['abnormality_id']}.png"
        fig.savefig(out/name,dpi=120);plt.close(fig)
        entry={k:row[k] for k in ('canonical_image_id','participant_id','record_id','abnormality_id','grid_relation','proposed_transform','full_rows','full_columns','mask_rows','mask_columns','source_mask_pixels','screen_supported','screen_failure_reason')}
        entry.update(panel=name,review_status='PENDING_HUMAN_REVIEW',reviewer='',reviewed_at='',review_notes='')
        index.append(entry)
        with Image.open(out/name) as im:
            im.thumbnail((1400,500));thumbnails.append(im.convert('RGB').copy())
    write_csv(out/'index.csv',index)
    canvas=Image.new('RGB',(max(i.width for i in thumbnails),sum(i.height for i in thumbnails)),'white');y=0
    for im in thumbnails:canvas.paste(im,(0,y));y+=im.height
    canvas.save(root/'reports/evidence/roi_alignment_validation_contact_sheet.png')


def development(root):
    require_unlocked_evidence(root)
    # Deliberately do not consume Phase A or any test-derived geometry summaries.
    images={r['canonical_image_id']:r for part in ('train','validation') for r in selected_images(root,part)[0]}
    mapping=mapped_rows(root,'train')
    if any(r['final_partition'] not in ('train','validation') or r['collection']!='Mass-Training' for r in mapping):
        raise ValueError('Test annotations prohibited in development')
    masks={r['record_id']:r for r in mapping if r['role']=='mask'}
    crops={r['record_id']:r for r in mapping if r['role']=='cropped'}
    if len(masks)+len(crops)!=len(mapping) or set(masks)!=set(crops):raise ValueError('Duplicate/missing lesion records')
    rows=[];last_id=None;source_hashes={}
    for r in sorted(masks.values(),key=lambda r:(r['canonical_image_id'],r['record_id'])):
        image=images[r['canonical_image_id']];require_development(image['final_partition'])
        if last_id!=r['canonical_image_id']:
            header,decoded,full_hash=headers_and_image(root,image,pixels=True);last_id=r['canonical_image_id']
            source_hashes[image['dicom_relative_path']]=full_hash
        md=checked_mask(root,r);full=decoded.pixels
        relation=grid_relation(full.shape,md.mask.shape)
        row={k:r[k] for k in ('canonical_image_id','record_id','abnormality_id','participant_id','final_partition','mask_sha256')}
        row.update(pathology=r['lesion_pathology'],view=image['view'],laterality=image['laterality'],
                   multiple_lesions=int(image['lesion_row_count'])>1,full_rows=full.shape[0],full_columns=full.shape[1],
                   mask_rows=md.mask.shape[0],mask_columns=md.mask.shape[1],source_mask_pixels=int(md.mask.sum()),**relation,
                   proposed_transform='identity_candidate' if relation['dimensions_equal'] else 'proportional_nearest_neighbour_candidate')
        path=annotation_path(root,crops[r['record_id']]);ch=pydicom.dcmread(path,stop_before_pixels=True)
        if annotation_role(ch)!='cropped' or str(ch.SOPInstanceUID)!=crops[r['record_id']]['SOPInstanceUID']:
            raise ValueError('Supplied crop identity changed')
        row['crop_sha256']=sha256_file(path);row['full_sha256']=full_hash
        crop=decode_dicom(path).pixels
        support=candidate_mask(md.mask,full.shape)
        row.update(foreground_checks(full,decoded.valid_mask,support))
        row.update(crop_consistency(full,crop,support))
        row['screen_supported'],row['screen_failure_reason']=evidence_screen(row)
        row['alignment_approval']='UNPROVEN_PENDING_REVIEW'
        rows.append(row)
        if len(rows)%50==0:print('Development',len(rows),flush=True)
    out=root/'reports/evidence'
    write_csv(out/'roi_alignment_development_checks.csv',rows)
    summaries={part:{grid:summarise([r for r in rows if r['final_partition']==part and r['grid_relation']==grid]) for grid in sorted({r['grid_relation'] for r in rows})} for part in ('train','validation')}
    summary={'status':'NOT_READY','development_partitions':['train','validation'],'test_pixels_used':False,'scores_computed':False,
             'screen':'nonbackground>=0.99; bbox-centred native crop NCC>=0.995; crop covers bbox; >=3 valid displaced controls with max NCC<0.98',
             'screen_is_not_alignment_approval':True,'otsu_note':'Largest-component support is descriptive tissue proxy, not tissue ground truth or an acceptance gate',
             'results':summaries,'development_csv_sha256':sha256_file(out/'roi_alignment_development_checks.csv'),
             'full_image_source_hashes':source_hashes}
    save_json(out/'roi_alignment_equal_grid_validation.json',summary)
    panels(root,rows,images,masks,crops)
    protocol={'rule_version':'coordinate_correspondence_candidate_v1','status':'NOT_READY','locked':False,
              'development_partitions':['train','validation'],'test_localisation_outcomes_used':False,
              'test_full_or_crop_pixels_used':False,'test_geometry_used_to_select_rule':False,
              'proposed_rules':['identity_candidate','proportional_nearest_neighbour_candidate'],
              'approved_rules':[], 'counts_per_candidate_rule':dict(Counter(r['proposed_transform'] for r in rows)),
              'development_screen_supported':sum(r['screen_supported'] for r in rows),
              'excluded_development_count':len(rows),'exclusion_reasons':{'coordinate_correspondence_not_yet_approved':len(rows)},
              'test_eligibility':'not assessed; geometry inventory only',
              'reason':'Diagnostic support does not establish universal pixel correspondence. Review exceptions and candidate overlays before any admission/lock. Existing strict audit unchanged.',
              'implementation_sha256':{p.name:sha256_file(p) for p in (Path(__file__),Path(__file__).with_name('coordinate_checks.py'))},
              'config_sha256':sha256_file(root/'config/localisation.yaml'),
              'evidence_sha256':{p.name:sha256_file(p) for p in (out/'roi_alignment_development_checks.csv',out/'roi_alignment_equal_grid_validation.json',out/'roi_alignment_validation_contact_sheet.png')},
              'generated_at_utc':datetime.now(timezone.utc).isoformat()}
    save_json(out/'roi_alignment_protocol.json',protocol)
    print(json.dumps(summaries,indent=2),flush=True)
    print('NOT_READY: diagnostic evidence complete; no approval or scoring',flush=True)


def require_unlocked_evidence(root):
    """Historical investigation must never overwrite a reviewed/locked evidence set."""
    path=root/'reports/evidence/roi_alignment_protocol.json'
    if (root/'config/roi_alignment_lock.json').exists() or (path.exists() and json.loads(path.read_text()).get('locked')):
        raise ValueError('Coordinate evidence is locked; regeneration would overwrite human review')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--geometry-all',action='store_true');group.add_argument('--development',action='store_true')
    args=parser.parse_args()
    if args.geometry_all:geometry_inventory(PROJECT_ROOT)
    else:development(PROJECT_ROOT)


if __name__=='__main__':main()
