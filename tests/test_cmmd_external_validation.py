"""Synthetic CMMD safety tests: no real metadata, pixels, model or evaluation."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from src.external_validation import cmmd_provenance as p
from src.external_validation.cmmd_metadata import imaging_id, normalize_label, inspect_workbook
from src.external_validation.cmmd_cohort import build_cohort
from src.external_validation.cmmd_audit import audit, csv_text
from src.external_validation import cmmd_protocol as protocol
from src.external_validation.cmmd_images import verify_header
from src.external_validation.cmmd_metrics import report_metrics, participant_bootstrap

MAPPING = {'Clinical': dict(id1='first', id2='second', laterality='affected', classification='diagnosis', abnormality='lesion')}
EVIDENCE = {'Clinical': 'Synthetic fixture explicitly labels the affected side.'}


def row(**changes):
    return dict(first='D1-0001', second='D2-0001', affected='Left', diagnosis='Malignant', lesion='Mass', **changes)


def cohort(rows):
    return build_cohort({'Clinical': pd.DataFrame(rows)}, MAPPING, EVIDENCE)


@pytest.mark.parametrize('id1,id2,expected', [('D1-0001','D2-0001','D2-0001'), ('D1-0001','','D1-0001'),
                                          ('','D2-0001','D2-0001'), ('D1-0001','unknown',None), ('','','')])
def test_identifier_mapping(id1,id2,expected):
    assert imaging_id(id1,id2) == (expected or None)


@pytest.mark.parametrize('value,expected', [(' BENIGN ',0),('malignant',1),('Malignant/Benign',None),('',None),('1',None)])
def test_label_normalisation(value,expected):
    assert normalize_label(value) == expected


@pytest.mark.parametrize('field,value,reason', [('lesion','Calcification','calcification_only'),
    ('lesion','Mass and calcification','missing_or_ambiguous_abnormality'), ('lesion','','missing_or_ambiguous_abnormality'),
    ('affected','','missing_or_ambiguous_laterality'), ('affected','bilateral','missing_or_ambiguous_laterality'),
    ('diagnosis','unclear','missing_or_ambiguous_classification'),('second','invalid','unsafe_imaging_patient_id')])
def test_exclusions(field,value,reason):
    record=row();record[field]=value
    eligible,excluded=cohort([record])
    assert not eligible and reason in excluded[0]['reasons']


def test_duplicate_sides_exclude_all_and_allow_explicit_opposite_diagnosis():
    eligible,excluded=cohort([row(),row()])
    assert not eligible and all('unresolved_duplicate_breast_side' in r['reasons'] for r in excluded)
    other=row();other.update(affected='Right',diagnosis='Benign')
    eligible,excluded=cohort([row(),other])
    assert not excluded and {(r['laterality'],r['label']) for r in eligible} == {('L',1),('R',0)}


def test_alias_ambiguity_and_known_duplicate():
    other=row();other['second']='D2-0002'
    eligible,excluded=cohort([row(),other])
    assert not eligible and all('ambiguous_ID1_to_imaging_ID_mapping' in r['reasons'] for r in excluded)
    record=row();record['second']='D2-0284'
    assert 'TCIA_unresolved_duplicate_pixels' in cohort([record])[1][0]['reasons']


def header(**changes):
    values=dict(PatientID='D2-0001', ImageLaterality='L', Modality='MG', ViewPosition='CC',
                StudyInstanceUID='1.2',SeriesInstanceUID='1.2.3',SOPInstanceUID='1.2.3.4')
    values.update(changes)
    return SimpleNamespace(**values)


def test_side_specific_header_and_contralateral_exclusion():
    eligible,_=cohort([row()])
    result=verify_header(header(),eligible,full_mammogram_evidence='synthetic full image')
    assert result['label']==1 and result['laterality']=='L'
    for h in (header(ImageLaterality='R'),header(PatientID='D1-0001'),header(ImageLaterality=''),
              header(Laterality='R'),header(ViewPosition='unknown'),header(NumberOfFrames=2)):
        with pytest.raises(ValueError):
            verify_header(h,eligible,full_mammogram_evidence='synthetic')


def test_mapping_semantics_required():
    sheets={'Clinical':pd.DataFrame([row()])}
    assert not build_cohort(sheets,MAPPING,{})[0]
    bad=copy.deepcopy(MAPPING);bad['Clinical']['abnormality']=None
    assert not build_cohort(sheets,bad,EVIDENCE)[0]


def test_real_xlsx_parser_and_audit_outputs(tmp_path):
    file=tmp_path/'synthetic.xlsx'
    with pd.ExcelWriter(file,engine='openpyxl') as writer:
        pd.DataFrame([row()]).to_excel(writer,sheet_name='Clinical',index=False)
        pd.DataFrame([{'unexpected':'unmapped'}]).to_excel(writer,sheet_name='Other',index=False)
    sheets=inspect_workbook(file)
    assert list(sheets['Clinical'].columns)==list(row())
    result=audit(file,{'columns':MAPPING,'side_specific_evidence':EVIDENCE},root=tmp_path)
    assert result['total_source_rows']==2 and result['eligible_malignant_sides']==1
    assert result['excluded_source_rows']==1 and result['spreadsheet_sha256']==p.sha256(file)
    destination=tmp_path/p.EVIDENCE
    assert {f.name for f in destination.iterdir()} == {'cmmd_metadata_summary.json','cmmd_metadata_audit.md','cmmd_eligible_sides.csv','cmmd_exclusions.csv'}
    with pytest.raises(FileExistsError):
        audit(file,root=tmp_path)
    assert not (tmp_path/p.RESULTS).exists()


def test_schema_only_stops(tmp_path):
    file=tmp_path/'synthetic.xlsx'
    pd.DataFrame([row()]).to_excel(file,index=False)
    assert audit(file,root=tmp_path)['status']=='STOP_NO_DEFENSIBLE_MASS_COHORT'


def test_no_fabrication_without_metadata_or_images(tmp_path):
    with pytest.raises(FileNotFoundError,match=p.WAITING):
        audit(tmp_path/'absent.xlsx',root=tmp_path)
    with pytest.raises(RuntimeError,match='disabled'):
        p.run_inference()
    with pytest.raises(ValueError,match='No predictions'):
        report_metrics([])
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('key,value', [('training_allowed',True),('fine_tuning_allowed',True),
    ('threshold_tuning_allowed',True),('checkpoint','other.keras'),('checkpoint_sha256','wrong'),
    ('classifier_threshold',0.3753),('selected_model','vgg16'),('input_size',256),
    ('output_root','results/experiments'),('inference_enabled',True),('official_test_evaluation_allowed',True)])
def test_locked_config(key,value):
    config=yaml.safe_load((p.ROOT/'config/external_validation/cmmd.yaml').read_text())
    p.validate_config(config)
    config[key]=value
    with pytest.raises(ValueError): p.validate_config(config)


@pytest.mark.parametrize('path', ['results/experiments/overwrite.json','results/external_validation/CMMD/../bad.json',
    'reports/evidence/model_evaluation/test.json','C:/outside.json','/outside.json'])
def test_output_isolation(tmp_path,path):
    with pytest.raises(ValueError): p.write_new(path,'bad',tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_output_creation_never_overwrites(tmp_path):
    path=p.RESULTS+'/metrics/synthetic.json'
    p.write_new(path,'{}',tmp_path)
    with pytest.raises(FileExistsError): p.write_new(path,'bad',tmp_path)
    assert (tmp_path/path).read_text()=='{}'


def test_external_import_graph_has_no_training_or_official_evaluation():
    allowed={'argparse','ast','collections','csv','hashlib','io','json','numpy','pandas','pathlib','pydicom','re','yaml','openpyxl',
             'src.evaluation.metrics', 'src.preprocessing.canonical'}
    for file in (p.ROOT/'src/external_validation').glob('*.py'):
        file_allowed = allowed | ({'tensorflow.keras.models', 'src.preprocessing.model_inputs'}
                                  if file.name == 'cmmd_predict.py' else set())
        if file.name == 'cmmd_results.py':
            file_allowed |= {'time', 'matplotlib', 'matplotlib.pyplot', 'sklearn.metrics',
                             'src.evaluation.bootstrap', 'src.evaluation.calibration'}
        tree=ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                assert all(alias.name in file_allowed for alias in node.names)
            if isinstance(node,ast.ImportFrom) and not node.level:
                assert node.module in file_allowed
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                assert node.func.attr not in {'fit','train_on_batch','save_weights','set_weights','compile'}


def predictions():
    return [dict(patient_id='D1-0001' if i<2 else 'D2-0001',laterality='L',SOPInstanceUID=str(i),
                 label=int(i>=2),probability=prob) for i,prob in enumerate([.1,.4,.2,.8])]


def test_metrics_and_participant_bootstrap():
    result=report_metrics(predictions())
    assert [result[k] for k in ('tn','fp','fn','tp')]==[1,1,1,1]
    assert result['threshold']==p.THRESHOLD and result['false_negative_rate']==.5
    assert result['number_images']==4 and result['number_breast_sides']==2 and result['number_patients']==2
    ci=participant_bootstrap(predictions(),iterations=30)
    assert ci==participant_bootstrap(predictions(),iterations=30)
    assert ci['intervals']['roc_auc']['undefined_replicates']>0
    with pytest.raises(ValueError): report_metrics(predictions()+predictions())


def test_checkpoint_hash_tamper_detected_before_model_loading(tmp_path):
    config=tmp_path/'config/external_validation';config.mkdir(parents=True)
    (config/'cmmd.yaml').write_text((p.ROOT/'config/external_validation/cmmd.yaml').read_text())
    (config/'cmmd_lock.json').write_text(json.dumps(dict(checkpoint_sha256=p.CHECKPOINT_SHA256,
        classifier_threshold=p.THRESHOLD,frozen_files={})))
    checkpoint=tmp_path/p.CHECKPOINT;checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'synthetic tamper')
    with pytest.raises(ValueError,match='checkpoint changed'): p.verify_frozen(tmp_path)


def test_explicit_mixed_abnormality_is_excluded():
    record=row();record['lesion']='both'
    eligible,excluded=cohort([record])
    assert not eligible and excluded[0]['reasons']==['mixed_mass_and_calcification_excluded']


def test_d2_bilateral_malignant_sides_require_clarification():
    other=row();other['affected']='Right'
    eligible,excluded=cohort([row(),other])
    assert not eligible and len(excluded)==2
    assert all('D2_bilateral_malignancy_requires_clarification' in r['reasons'] for r in excluded)


def test_d1_explicit_bilateral_malignancy_not_conflated_with_d2_caution():
    first=row();first['second']=''
    other=dict(first,affected='Right')
    eligible,excluded=cohort([first,other])
    assert len(eligible)==2 and not excluded


def test_deterministic_cohort_serialisation():
    first=cohort([row()])[0]
    second=cohort([row()])[0]
    assert csv_text(first).encode()==csv_text(second).encode()
    assert '\r' not in csv_text(first)


def test_duplicate_actual_column_headers_rejected(tmp_path):
    import openpyxl
    book=openpyxl.Workbook();book.active.append(['ID1','ID1']);book.active.append(['D1-0001','D2-0001'])
    file=tmp_path/'duplicate.xlsx';book.save(file)
    with pytest.raises(ValueError,match='headings'): inspect_workbook(file)


def synthetic_catalogue(patient,uid):
    return dict(PatientID=patient,Collection='CMMD',Modality='MG',StudyInstanceUID='study'+uid,
                SeriesInstanceUID=uid,ImageCount=4,FileSize=1000)


def two_class_cohort():
    benign=row();benign.update(first='D1-0002',second='',diagnosis='Benign')
    return cohort([row(),benign])[0]


def test_selective_download_plan_does_not_invent_eligible_image_count():
    plan=protocol.download_plan(two_class_cohort(),[synthetic_catalogue('D2-0001','1'),
        synthetic_catalogue('D1-0002','2'),synthetic_catalogue('D1-0003','3')])
    assert len(plan)==2 and sum(r['catalogue_dicom_count'] for r in plan)==8
    assert all(r['right_label']=='' and r['download_status']=='NOT_DOWNLOADED' for r in plan)


def test_missing_or_duplicate_catalogue_mapping_stops():
    with pytest.raises(ValueError,match='missing'): protocol.download_plan(two_class_cohort(),[])
    series=[synthetic_catalogue('D2-0001','1'),synthetic_catalogue('D2-0001','1')]
    with pytest.raises(ValueError,match='duplicate'): protocol.download_plan(two_class_cohort(),series)


def protocol_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(protocol,'verify_frozen',lambda root: {})
    workbook=tmp_path/'synthetic.xlsx'
    benign=row();benign.update(first='D1-0002',second='',diagnosis='Benign')
    pd.DataFrame([row(),benign]).to_excel(workbook,sheet_name='Clinical',index=False)
    mapping={'columns':MAPPING,'side_specific_evidence':EVIDENCE}
    mapping_path=tmp_path/'mapping.json';mapping_path.write_text(json.dumps(mapping))
    audit(workbook,mapping,root=tmp_path)
    catalogue=tmp_path/'catalogue.csv'
    catalogue.write_text(protocol.plan_csv([synthetic_catalogue('D2-0001','1'),synthetic_catalogue('D1-0002','2')]))
    source=tmp_path/'source.json';source.write_text(json.dumps({'catalogue_csv_sha256':p.sha256(catalogue)}))
    return workbook,mapping_path,catalogue,source


def test_freeze_protocol_and_tamper_detection(tmp_path,monkeypatch):
    paths=protocol_fixture(tmp_path,monkeypatch)
    locked=protocol.freeze_protocol(*paths,root=tmp_path)
    assert locked['status']==protocol.READY and locked['download']['eligible_image_count'] is None
    assert locked['classifier_threshold']==p.THRESHOLD and locked['inference_enabled'] is False
    assert protocol.verify_protocol(tmp_path)==locked
    with pytest.raises(FileExistsError): protocol.freeze_protocol(*paths,root=tmp_path)
    target=tmp_path/p.EVIDENCE/'cmmd_eligible_sides.csv'
    target.write_text(target.read_text()+'tamper')
    with pytest.raises(ValueError,match='source changed'): protocol.verify_protocol(tmp_path)


def test_freeze_refuses_audit_drift(tmp_path,monkeypatch):
    paths=protocol_fixture(tmp_path,monkeypatch)
    target=tmp_path/p.EVIDENCE/'cmmd_eligible_sides.csv';target.write_text('changed')
    with pytest.raises(ValueError,match='cohort mismatch'): protocol.freeze_protocol(*paths,root=tmp_path)
    assert not (tmp_path/protocol.PROTOCOL).exists()


@pytest.mark.parametrize('key', ['model_selection_allowed','calibration_allowed'])
def test_additional_prohibitions(key):
    config=yaml.safe_load((p.ROOT/'config/external_validation/cmmd.yaml').read_text())
    config[key]=True
    with pytest.raises(ValueError): p.validate_config(config)
