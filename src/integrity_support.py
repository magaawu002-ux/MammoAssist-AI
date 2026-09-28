"""Additional header, clinical, chronology and reachable-Git-history audit; no inference."""
import importlib.metadata
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import openpyxl
import pydicom

from src.integrity_audit import ROOT, OUT, RUN, EXT, EVIDENCE, THRESHOLD, rows, digest, disagreement


def run(root=ROOT):
    root=Path(root)
    read=lambda p:json.loads((root/p).read_text(encoding='utf-8-sig'))
    computed=read(OUT+'/computed_checks.json')
    result={}
    result['calibration_mismatches']={}
    for name,path in [('cbis_test',RUN+'/test/calibration.json'),('cmmd',EXT+'/metrics/cmmd_calibration.json')]:
        expected=read(path)
        errors=[]
        for i,(got,saved) in enumerate(zip(computed[name]['calibration_bins'],expected['bins'])):
            converted=dict(lower=got['lower'],upper=got['upper'],count=got['n'],mean_probability=got['mean_score'],positive_fraction=got['malignant_fraction'])
            errors+=disagreement(converted,saved,str(i)+'.')
        result['calibration_mismatches'][name]=errors
    sources=read('reports/evidence/data_provenance.json')
    result['cbis_source_hash_checks']={s['repository_relative_path']:digest(root/s['repository_relative_path'])==s['sha256'] for s in sources['sources']}
    from src.evaluation.thresholds import select_threshold
    model_checks=[]
    for path in sorted((root/'results/experiments').glob('*/*/run_metadata.json')):
        meta=json.loads(path.read_text());directory=path.parent
        predictions=rows(directory/'validation/predictions.csv')
        threshold=json.loads((directory/'threshold.json').read_text())
        selected=select_threshold(np.array([int(r['true_label']) for r in predictions]),np.array([float(r['predicted_probability']) for r in predictions]),partition='validation',method=threshold['criterion'])
        checks={f:digest(directory/f)==meta[k] for f,k in [(meta['checkpoint'],'checkpoint_sha256'),('threshold.json','threshold_sha256'),('config_used.yaml','config_sha256'),('validation/predictions.csv','validation_predictions_sha256'),('validation/metrics.json','validation_metrics_sha256')]}
        model_checks.append(dict(model=meta['model'],selection_only=meta['selection_only'],checks=checks,
                                 validation_threshold_matches=selected['threshold']==threshold['threshold'],threshold=threshold['threshold'],
                                 training_class_counts=meta['training_class_counts']))
    result['model_bindings']=model_checks
    clinical=rows(root/EVIDENCE/'cmmd_eligible_sides.csv')
    summary=read(EVIDENCE+'/cmmd_metadata_summary.json')
    xlsx=root/'data/external/CMMD/metadata/CMMD_clinicaldata_revision.xlsx'
    assert digest(xlsx)==summary['spreadsheet_sha256']
    workbook=openpyxl.load_workbook(xlsx,read_only=True,data_only=True)
    sheet=workbook['Sheet1'];source=list(sheet.values);columns=list(source[0]);records=[dict(zip(columns,r)) for r in source[1:]]
    indexed={(r['ID1'],r['LeftRight']):r for r in records}
    assert len(indexed)==len(records)
    for r in clinical:
        original=indexed[(r['patient_id'],r['laterality'])]
        assert r['patient_id']==r['id1'] and not r['id2'] and not r['reasons']
        assert original['abnormality']=='mass' and int(r['label'])=={'Benign':0,'Malignant':1}[original['classification']]
    excluded=rows(root/EVIDENCE/'cmmd_exclusions.csv')
    result['clinical']=dict(source_rows=len(records),patients=len({r['ID1'] for r in records}),columns=columns,
        source_abnormality_counts=dict(Counter(r['abnormality'] for r in records)),eligible_sides=len(clinical),
        benign_sides=sum(r['label']=='0' for r in clinical),malignant_sides=sum(r['label']=='1' for r in clinical),
        eligible_patients=len({r['patient_id'] for r in clinical}),excluded_rows=len(excluded),
        primary_exclusion_counts=summary['primary_exclusion_counts'],all_exclusion_reasons=summary['exclusion_reasons'],
        xlsx_sha256=digest(xlsx),side_label_matches_source=True)
    workbook.close()
    headers=rows(root/EVIDENCE/'cmmd_dicom_headers.csv')
    fields=dict(patient_id='PatientID',study_uid='StudyInstanceUID',series_uid='SeriesInstanceUID',sop_uid='SOPInstanceUID',modality='Modality',rows='Rows',columns='Columns',photometric='PhotometricInterpretation',bits_stored='BitsStored')
    # Record only stable pseudonymous linkage and header agreement, never PatientName or dates.
    mismatches=[]
    for r in headers:
        ds=pydicom.dcmread(root/r['path'],stop_before_pixels=True)
        for k,tag in fields.items():
            if str(getattr(ds,tag,''))!=r[k]:mismatches.append(dict(sop_uid=r['sop_uid'],field=k))
    result['live_cmmd_header_recheck']=dict(objects=len(headers),mismatches=mismatches,pixel_decode=False)
    # Hash-bound clinical/image protocol and all downloaded file bytes, no preprocessing or model calls.
    from src.external_validation.cmmd_conversion import verify
    verify(root)
    result['cmmd_protocol_verification']='PASS'
    print('Clinical XLSX and CMMD source/header locks verified',flush=True)
    git=lambda *args:subprocess.check_output(['git',*args],cwd=root)
    paths=['results/experiments/final_selection.json','results/experiments/final_test_attempt.json','reports/evidence/final_test_recovery.md',
           'results/explainability/localisation_lock.json','results/explainability/final_localisation_attempt.json',
           EVIDENCE+'/cmmd_external_protocol_v2.json',EVIDENCE+'/cmmd_analysis_plan.json',EXT+'/predictions/cmmd_image_predictions_v2.csv']
    result['chronology']={p:git('log','--format=%H %aI %s','--all','--',p).decode().splitlines() for p in paths}
    tracked=git('ls-files').decode().splitlines()
    all_objects={}
    for line in git('rev-list','--objects','--all').decode().splitlines():
        oid,_,path=line.partition(' ');all_objects[oid]=path
    lines=subprocess.check_output(['git','cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],input=('\n'.join(all_objects)+'\n').encode(),cwd=root).decode().splitlines()
    blobs=[(oid,int(size),all_objects[oid]) for oid,kind,size in (line.split() for line in lines) if kind=='blob']
    signatures={
        'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
        'github_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
        'aws_access_key':re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
        'credential_assignment':re.compile(r'''(?i)\b(?:password|api_key|secret_key|access_token|client_secret)\s*[:=]\s*["'][^"'\r\n]{8,}["']'''),
        'personal_windows_path':re.compile(r'(?i)[A-Z]:[/\\]+Users[/\\]+[^/\\\s"\']+'),
        'patient_identifying_field':re.compile(r'\b(?:PatientName|PatientBirthDate|AccessionNumber)\b'),
    }
    hits=[];scanned=0
    for oid,size,path in blobs:
        data=git('cat-file','blob',oid)
        if b'\0' in data[:8192]:continue
        try:text=data.decode('utf-8-sig')
        except UnicodeDecodeError:continue
        scanned+=1
        for category,pattern in signatures.items():
            line_numbers=[i for i,line in enumerate(text.splitlines(),1) if pattern.search(line)]
            if line_numbers:hits.append(dict(oid=oid,path=path,category=category,lines=line_numbers))
    disallowed=lambda p:Path(p).suffix.lower() in {'.dcm','.dicom','.npy','.keras','.h5','.zip','.xlsx'} or any(x in p.split('/') for x in ('.venv','__pycache__','.pytest_cache'))
    result['repository']=dict(tracked_files=len(tracked),reachable_blobs=len(blobs),history_text_blobs_scanned=scanned,
        history_bytes=sum(b[1] for b in blobs),largest_history_blobs=[dict(oid=o,bytes=s,path=p) for o,s,p in sorted(blobs,key=lambda b:b[1],reverse=True)[:10]],
        prohibited_tracked=[p for p in tracked if disallowed(p)],prohibited_history=[dict(oid=o,path=p,bytes=s) for o,s,p in blobs if disallowed(p)],
        signature_findings=hits,checkpoint_bytes=(root/RUN/'best_model.keras').stat().st_size,
        tracked_pixel_figures=[p for p in tracked if Path(p).suffix.lower() in {'.png','.jpg','.jpeg','.pdf'}],
        limitations='Reachable Git objects only; not unreachable/deleted filesystem data, OCR, encrypted archives or comprehensive entropy detection. Signature hits require context review; no values copied.')
    result['environment']={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    result['pip_check']=subprocess.run([str(root/'.venv/Scripts/python.exe'),'-B','-m','pip','check'],capture_output=True,text=True).stdout.strip()
    result['documentation_hits']=[]
    pattern=re.compile(r'planned|future|calibrated|radiologist|intensity.marker|external|Brier|faithfulness',re.I)
    for p in tracked:
        if Path(p).suffix.lower() not in {'.md','.txt','.py'}:continue
        try:text=(root/p).read_text(encoding='utf-8-sig')
        except (OSError,UnicodeDecodeError):continue
        matches=[i for i,line in enumerate(text.splitlines(),1) if pattern.search(line)]
        if matches:result['documentation_hits'].append(dict(path=p,lines=matches))
    (root/OUT/'supporting_checks.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('Saved supporting audit checks',flush=True)


if __name__=='__main__':
    run()
