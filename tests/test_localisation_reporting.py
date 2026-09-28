"""Validation reporting integrity using small synthetic rows only."""

import copy

import pytest

from src.reporting.localisation_validation import (THRESHOLD, aggregate, breakdowns,
    enrich, representatives, validation_rows)


@pytest.fixture
def cohort():
    image=dict(canonical_image_id='a',participant_id='p',final_partition='validation',
               official_partition='train',binary_label='1',pathology='MALIGNANT',view='CC',laterality='LEFT')
    prediction=dict(image,true_label='1',predicted_probability='0.2',threshold=str(THRESHOLD),
                    predicted_class='0',correct='False',error_type='false_negative')
    mapping=[]; metrics=[]
    for lesion in ('one','two'):
        mapping.append(dict(canonical_image_id='a',record_id=lesion,role='mask',reference='Mass-Training_P_1',
                            alignment_status='verified',alignment_method='locked_equal_grid_identity',mask_sha256=lesion))
        metrics.append(dict(canonical_image_id='a',participant_id='p',partition='validation',scope='lesion',
                            lesion_id=lesion,pathology='MALIGNANT',mask_sha256=lesion,threshold='0.5',
                            pointing_game='False',lesion_hit='True',zero_heatmap='False',iou='0.1',dice=str(2/11),activation_mass_inside='0.25'))
    union=dict(metrics[0],scope='image_union',lesion_id='union',mask_sha256='',iou='0.2',dice=str(1/3),activation_mass_inside='0.5')
    metrics.append(union)
    return metrics,[prediction],[image],mapping


def test_multiple_lesions_remain_distinct_from_image_union(cohort):
    rows=enrich(*cohort); groups=breakdowns(rows)
    assert groups['lesion/overall']['n']==2
    assert groups['image_union/overall']['n']==1
    assert groups['lesion/overall']['participants']==1
    assert groups['lesion/malignant/false_negative']['n']==2
    assert groups['image_union/malignant/false_negative']['n']==1
    assert groups['lesion/overall']['lesion_hit_rate']==1
    assert groups['lesion/overall']['mean_iou']==.1
    assert dict(representatives(rows))['multiple_lesions']['canonical_image_id']=='a'


@pytest.mark.parametrize('field,value', [('threshold','0.6'),('predicted_class','1'),('predicted_probability','nan'),('correct','True')])
def test_existing_prediction_decisions_cannot_change(cohort,field,value):
    cohort[1][0][field]=value
    with pytest.raises(ValueError): enrich(*cohort)


@pytest.mark.parametrize('position',[0,1,2])
def test_test_rows_rejected_before_join(cohort,position):
    row=cohort[position][0]
    row['partition' if position==0 else 'final_partition']='test'
    with pytest.raises(ValueError,match='Only Validation'): enrich(*cohort)


def test_test_mapping_forbidden(cohort):
    cohort[3][0]['reference']='Mass-Test_P_1'
    with pytest.raises(ValueError,match='Test mapping'): enrich(*cohort)


def test_unequal_mask_entering_metrics_fails(cohort):
    cohort[3][0].update(alignment_status='excluded',alignment_method='EXCLUDE_UNPROVEN')
    with pytest.raises(ValueError,match='Excluded'): enrich(*cohort)


def test_proportional_mapping_never_scored(cohort):
    cohort[3][0]['alignment_method']='PROPORTIONAL_NEAREST_NEIGHBOUR'
    with pytest.raises(ValueError,match='Unproven'): enrich(*cohort)


@pytest.mark.parametrize('field,value',[('threshold','0.4'),('iou','nan'),('dice','inf'),('activation_mass_inside','')])
def test_invalid_metric_or_threshold_fails(cohort,field,value):
    cohort[0][0][field]=value
    with pytest.raises(ValueError): enrich(*cohort)


def test_incomplete_lesion_cohort_fails(cohort):
    cohort[0].pop(0)
    with pytest.raises(ValueError,match='Incomplete'): enrich(*cohort)


def test_zero_map_mass_is_undefined_not_zero(cohort):
    rows=enrich(*cohort)
    row=rows[0]; row.update(zero_heatmap=True,activation_mass_inside=None)
    stats=aggregate([row])
    assert stats['mean_activation_mass_inside'] is None
    assert stats['activation_mass_inside_defined_n']==0
    assert stats['zero_heatmaps']==1


def test_figures_select_both_extremes_deterministically(cohort):
    row=enrich(*cohort)[-1]
    high=dict(row,canonical_image_id='z',iou=.8)
    low=dict(row,canonical_image_id='b',iou=0)
    tied=dict(low,canonical_image_id='a')
    first=representatives([high,low,tied]); second=representatives([tied,high,low])
    assert first==second
    assert first[0][1]['canonical_image_id']=='z'
    assert first[1][1]['canonical_image_id']=='a'
