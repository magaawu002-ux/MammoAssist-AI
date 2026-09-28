"""Synthetic catalogue/manifest checks, without network, pixels or models."""
import ast
from pathlib import Path
import pytest
from src.external_validation import cmmd_manifest as m


def series(patient='D1-0001', uid='1.2.3', **changes):
    return dict(PatientID=patient, StudyInstanceUID='1.2', SeriesInstanceUID=uid,
                Collection='CMMD', Modality='MG', ImageCount=2, FileSize=1000) | changes


def manifest(uids=('1.2.3',)):
    return ('downloadServerUrl=https://nbia.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet\n'
            'includeAnnotation=false\nnoOfrRetry=4\ndatabasketId=manifest-123.tcia\nmanifestVersion=3.0\n'
            'ListOfSeriesToDownload=\n' + '\n'.join(uids) + '\n').encode()


def test_baseline_patient_lock_tamper_stops_before_other_work(tmp_path, monkeypatch):
    for name in m.BASELINE_HASHES:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'tampered')
    monkeypatch.setattr(m,'verify_protocol',lambda root: pytest.fail('Must reject baseline first'))
    with pytest.raises(ValueError,match='differs'): m.load_locked_ids(tmp_path)


def test_select_only_locked_patients_then_validate():
    selected=m.select_catalogue(['D1-0001'],[series(),series('D1-0002')])
    assert len(selected)==1 and m.reconcile(['D1-0001'],selected)['status']=='PASS'


@pytest.mark.parametrize('change',[dict(Modality='CT'),dict(Collection='OTHER'),dict(ImageCount=0),
                                   dict(SeriesInstanceUID='invented'),dict(FileSize=-1)])
def test_invalid_series_rejected(change):
    assert m.reconcile(['D1-0001'],[series(**change)])['status']=='FAIL'


def test_extra_missing_and_duplicate_series_reported():
    report=m.reconcile(['D1-0001','D1-0003'],[series(),series(),series('D1-0002','1.2.4')])
    assert report['extra_patients']==['D1-0002'] and report['missing_patients']==['D1-0003']
    assert report['duplicate_series']==['1.2.3'] and report['status']=='FAIL'


def test_multiple_studies_and_series_reported_without_forcing_previous_counts():
    report=m.reconcile(['D1-0001'],[series(),series(uid='1.2.4',StudyInstanceUID='1.3')])
    assert report['multiple_studies']=={'D1-0001':['1.2','1.3']}
    assert report['multiple_series']=={'D1-0001':['1.2.3','1.2.4']}
    assert report['dicom_objects']==4 and report['estimated_bytes']==2000


def test_unknown_size_is_not_invented():
    assert m.reconcile(['D1-0001'],[series(FileSize=None)])['estimated_bytes'] is None


def test_manifest_parsing_and_determinism():
    first=m.verify_manifest(['D1-0001'],[series()],manifest())
    assert first==m.verify_manifest(['D1-0001'],[series()],manifest())
    assert first['status']=='PASS' and first['manifest_series_count']==1
    assert m.parse_manifest(manifest().replace(b'\n',b'\r\n'))[1]==['1.2.3']


@pytest.mark.parametrize('content',[manifest(('1.2.3','1.2.3')),manifest(('not-a-uid',)),
    manifest().replace(b'includeAnnotation=false',b'includeAnnotation=true'),
    manifest().replace(b'https://nbia.cancerimagingarchive.net',b'https://untrusted.example'),
    manifest()+b'ListOfSeriesToDownload=\n',manifest().replace(b'manifestVersion=3.0',b'manifestVersion=9.0')])
def test_invalid_manifest_rejected(content):
    with pytest.raises(ValueError): m.parse_manifest(content)


def test_wrong_manifest_set_fails():
    result=m.verify_manifest(['D1-0001'],[series()],manifest(('1.2.4',)))
    assert result['status']=='FAIL' and result['manifest_extra_series']==['1.2.4']
    assert result['manifest_missing_series']==['1.2.3']


def test_no_network_inference_or_download_execution():
    tree=ast.parse(Path(m.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            assert all(a.name in {'hashlib','re'} for a in node.names)
        if isinstance(node,ast.ImportFrom):
            assert node.module in {'collections','pathlib','cmmd_protocol','cmmd_provenance'}
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in {'predict','fit','urlopen','run','Popen','write_bytes','write_text'}
