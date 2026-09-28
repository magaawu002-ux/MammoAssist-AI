"""Synthetic coordinate diagnostics; no raw medical data or model required."""

import ast
from pathlib import Path

import numpy as np
import pytest

from src.explainability.coordinate_checks import (candidate_mask, crop_consistency,
    evidence_screen, foreground_checks, grid_relation, mask_statistics, ncc, require_development)
from src.explainability.coordinate_evidence import development, review_selection


def mask(shape=(100,80)):
    result=np.zeros(shape,bool);result[30:50,20:35]=True
    return result


def test_equal_grid_identity_is_exact_and_independent_copy():
    source=mask(); result=candidate_mask(source,source.shape)
    np.testing.assert_array_equal(result,source)
    result[0,0]=True
    assert not source[0,0]


def test_proportional_candidate_binary_and_deterministic_not_approval():
    source=mask()
    a=candidate_mask(source,(200,160));b=candidate_mask(source,(200,160))
    assert a.dtype==bool and set(np.unique(a))=={False,True}
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a,np.repeat(np.repeat(source,2,axis=0),2,axis=1))


def test_nonproportional_candidate_rejected():
    assert grid_relation((200,90),(100,80))['grid_relation']=='non_proportional'
    with pytest.raises(ValueError,match='Aspect-ratio'):
        candidate_mask(mask(),(200,90))


@pytest.mark.parametrize('shape',[(0,80),(-1,80),(100.0,80)])
def test_invalid_geometry_rejected(shape):
    with pytest.raises(ValueError,match='dimensions'):
        grid_relation(shape,(100,80))


def test_support_statistics_use_exclusive_bbox_and_actual_centroid():
    s=mask_statistics(mask())
    assert (s['bbox_y0'],s['bbox_y1'],s['bbox_x0'],s['bbox_x1'])==(30,50,20,35)
    assert (s['centroid_y'],s['centroid_x'])==(39.5,27)
    assert s['mask_pixels']==300 and s['mask_area_fraction']==300/8000


@pytest.mark.parametrize('pixels',[np.zeros((4,4),bool),np.ones((4,4),bool),np.eye(4,dtype=float)])
def test_invalid_mask_excluded(pixels):
    with pytest.raises(ValueError): mask_statistics(pixels)


def test_crop_correspondence_survives_intensity_rescaling_and_rejects_wrong_site():
    rng=np.random.default_rng(5);full=rng.uniform(10,100,(100,80))
    support=mask()
    crop=3*full[25:55,15:40]+120
    result=crop_consistency(full,crop,support)
    assert result['crop_ncc_at_proposed_origin']==pytest.approx(1)
    assert result['local_search_dy']==result['local_search_dx']==0
    assert result['negative_control_max_ncc'] < .2
    check=dict(result,nonbackground_fraction=1)
    assert evidence_screen(check)==(True,'')
    wrong=crop_consistency(full,full[60:90,45:70],support)
    assert not evidence_screen(dict(wrong,nonbackground_fraction=1))[0]


def test_constant_correlations_are_undefined_not_perfect_matches():
    assert ncc(np.ones((8,8)),np.ones((8,8))) is None
    assert ncc(np.ones((8,8)),np.arange(64).reshape(8,8)) is None


def test_background_support_does_not_pass_evidence_screen():
    full=np.zeros((100,80));full[20:80,:50]=100
    support=np.zeros_like(full,bool);support[20:30,65:75]=True
    checks=foreground_checks(full,np.ones_like(full,bool),support)
    assert checks['nonbackground_fraction']==0
    result=dict(checks,crop_ncc_at_proposed_origin=1,crop_contains_mask_bbox=True,
                negative_control_count=8,negative_control_max_ncc=.1)
    assert evidence_screen(result)==(False,'background_support')


@pytest.mark.parametrize('part',['test','unknown',''])
def test_development_partitions_fail_closed(part):
    with pytest.raises(ValueError,match='test is forbidden'):require_development(part)


def test_development_refuses_test_injected_into_mapping_before_pixel_access(tmp_path,monkeypatch):
    monkeypatch.setattr('src.explainability.coordinate_evidence.selected_images',lambda root,part: ([],{}))
    monkeypatch.setattr('src.explainability.coordinate_evidence.mapped_rows',lambda root,part:[dict(final_partition='test',collection='Mass-Test')])
    def forbidden(*args,**kwargs):pytest.fail('Test pixels accessed')
    monkeypatch.setattr('src.explainability.coordinate_evidence.headers_and_image',forbidden)
    with pytest.raises(ValueError,match='Test annotations prohibited'):development(tmp_path)


def test_review_selection_preserves_multiple_lesions_and_excludes_test():
    base=dict(grid_relation='equal',pathology='BENIGN',view='CC',laterality='LEFT',
              multiple_lesions=True,crop_ncc_at_proposed_origin=.99)
    rows=[dict(base,record_id=str(i),final_partition='validation') for i in (1,2)]
    rows += [dict(base,record_id='test',final_partition='test')]
    assert {r['record_id'] for r in review_selection(rows)}=={'1','2'}


def test_evidence_modules_do_not_import_models_or_performance_runners():
    package=Path(__file__).resolve().parents[1]/'src/explainability'
    forbidden=('tensorflow','gradcam','localisation_metrics','evaluation','training')
    for name in ('coordinate_checks.py','coordinate_evidence.py'):
        tree=ast.parse((package/name).read_text())
        for node in ast.walk(tree):
            imports=([node.module or ''] if isinstance(node,ast.ImportFrom) else
                     [n.name for n in node.names] if isinstance(node,ast.Import) else [])
            assert not any(part in module.split('.') for module in imports for part in forbidden)


def test_positive_diagnostic_screen_is_not_an_alignment_review():
    from src.explainability.alignment import prove_full_grid
    import pydicom
    a=pydicom.Dataset();a.Rows=100;a.Columns=80;a.SOPInstanceUID='1.1'
    b=pydicom.Dataset();b.Rows=100;b.Columns=80;b.SOPInstanceUID='1.2'
    check=dict(nonbackground_fraction=1,crop_ncc_at_proposed_origin=1,crop_contains_mask_bbox=True,
               negative_control_count=8,negative_control_max_ncc=.1)
    assert evidence_screen(check)[0]
    with pytest.raises(ValueError,match='UNVERIFIED_ALIGNMENT'):prove_full_grid(a,b,check)


def test_review_includes_foreground_exceptions_even_with_perfect_crop_match():
    base=dict(grid_relation='equal',pathology='BENIGN',view='CC',laterality='LEFT',
              multiple_lesions=False,crop_ncc_at_proposed_origin=.99,final_partition='validation')
    rows=[dict(base,record_id=str(i),screen_supported=True,otsu_largest_component_fraction=1) for i in range(4)]
    rows.append(dict(base,record_id='background',screen_supported=False,crop_ncc_at_proposed_origin=1,otsu_largest_component_fraction=1))
    rows.append(dict(base,record_id='proxy_failure',screen_supported=True,crop_ncc_at_proposed_origin=1,otsu_largest_component_fraction=0))
    chosen={r['record_id'] for r in review_selection(rows)}
    assert {'background','proxy_failure'} <= chosen
