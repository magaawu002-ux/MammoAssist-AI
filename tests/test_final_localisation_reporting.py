"""Final reporting uses saved outputs, with synthetic Test fixtures only."""

import json

import pytest

from test_localisation_reporting import cohort
from src.data.provenance import sha256_file
from src.reporting.localisation_final import completed_run, enrich_test
from src.reporting.localisation_validation import breakdowns


@pytest.fixture
def test_cohort(cohort):
    metrics,predictions,manifest,mapping=cohort
    for row in metrics: row['partition']='test'
    for row in predictions+manifest: row.update(final_partition='test',official_partition='test')
    for row in mapping: row['reference']='Mass-Test_P_1'
    return cohort


def test_final_multiple_lesions_and_union(test_cohort):
    rows,excluded=enrich_test(*test_cohort); groups=breakdowns(rows)
    assert excluded==0
    assert groups['lesion/overall']['n']==2
    assert groups['image_union/overall']['n']==1
    assert groups['image_union/malignant/false_negative']['n']==1


@pytest.mark.parametrize('position',[0,1,2])
def test_other_partitions_rejected(test_cohort,position):
    test_cohort[position][0]['partition' if position==0 else 'final_partition']='validation'
    with pytest.raises(ValueError,match='Only official Test'): enrich_test(*test_cohort)


def test_exclusion_is_retained_but_not_scored(test_cohort):
    test_cohort[3].append(dict(canonical_image_id='a',record_id='excluded',role='mask',reference='Mass-Test_P_1',
        alignment_status='excluded',alignment_method='EXCLUDE_UNPROVEN',reason='exact_spatial_transform_unproven',canonical_mask_pixels=''))
    rows,n=enrich_test(*test_cohort)
    assert n==1 and len(rows)==3
    test_cohort[0][0]['lesion_id']='excluded'
    with pytest.raises(ValueError,match='Excluded'): enrich_test(*test_cohort)


@pytest.mark.parametrize('field,value',[('threshold','0.4'),('predicted_probability','nan'),('error_type','true_positive')])
def test_saved_decisions_must_match_fixed_threshold(test_cohort,field,value):
    test_cohort[1][0][field]=value
    with pytest.raises(ValueError,match='Frozen classifier'): enrich_test(*test_cohort)


def test_test_heatmap_threshold_unchanged(test_cohort):
    test_cohort[0][0]['threshold']='0.4'
    with pytest.raises(ValueError,match='heatmap threshold'): enrich_test(*test_cohort)


@pytest.fixture
def completed(tmp_path,monkeypatch):
    root=tmp_path; directory=root/'results/explainability/test/synthetic'; directory.mkdir(parents=True)
    (directory/'metrics.csv').write_text('synthetic')
    identity={'synthetic':True}
    summary=dict(status='COMPLETE',partition='test',identity=identity,artifact_sha256={'metrics.csv':sha256_file(directory/'metrics.csv')})
    (directory/'summary.json').write_text(json.dumps(summary))
    receipt=dict(status='COMPLETE',run='results/explainability/test/synthetic',summary_sha256=sha256_file(directory/'summary.json'))
    base=root/'results/explainability'
    (base/'final_localisation_attempt.json').write_text(json.dumps(receipt))
    (base/'localisation_lock.json').write_text(json.dumps({'identity':identity}))
    monkeypatch.setattr('src.reporting.localisation_final.fixed_protocol',lambda root:(identity,{}))
    def guard(root): raise ValueError('Final localisation already attempted; no repeated test-driven tuning')
    monkeypatch.setattr('src.reporting.localisation_final.require_final_lock',guard)
    return root,directory


def test_completed_receipt_guard_check_does_not_run_evaluation(completed,monkeypatch):
    def forbidden(*args,**kwargs): pytest.fail('Reporting must not run evaluation')
    monkeypatch.setattr('src.explainability.localisation.evaluate_localisation',forbidden)
    root,directory=completed
    assert completed_run(root)[0]==directory


def test_receipt_hash_tampering_blocks_reporting(completed):
    root,directory=completed
    with (directory/'summary.json').open('a') as f: f.write(' ')
    with pytest.raises(ValueError,match='differs from receipt'): completed_run(root)


def test_final_artifact_tampering_blocks_reporting(completed):
    root,directory=completed
    (directory/'metrics.csv').write_text('modified')
    with pytest.raises(ValueError,match='artifact changed'): completed_run(root)


def test_no_report_if_repeat_guard_does_not_block(completed,monkeypatch):
    root,_=completed
    monkeypatch.setattr('src.reporting.localisation_final.require_final_lock',lambda root:None)
    with pytest.raises(ValueError,match='protection failed'): completed_run(root)
