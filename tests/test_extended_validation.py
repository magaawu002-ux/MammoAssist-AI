"""Synthetic checks; never read real Test inputs or invoke cohort evaluation."""
import numpy as np
import pytest

from src.extended_validation import analysis as a


@pytest.mark.parametrize('name', a.STRESSES)
def test_perturbations_deterministic_bounded_and_immutable(name):
    image=np.random.default_rng(4).uniform(.1,.9,(224,224)).astype(np.float32)
    image[:20]=0; before=image.copy()
    first=a.perturb(image,name,'synthetic')
    np.testing.assert_array_equal(first,a.perturb(image,name,'synthetic'))
    np.testing.assert_array_equal(image,before)
    assert first.dtype==np.float32 and np.isfinite(first).all()
    assert first.min()>=0 and first.max()<=1 and not first[:20].any()
    if name=='original': np.testing.assert_array_equal(first,image)
    else: assert not np.array_equal(first,image)


@pytest.mark.parametrize('partition,official',[('test','test'),('train','train'),('validation','test')])
def test_reject_test_or_nonvalidation_before_pixels(partition,official):
    with pytest.raises(ValueError,match='Only'):
        a.validate_rows([dict(final_partition=partition,official_partition=official,canonical_image_id='x')])


def test_loader_rejects_test_rows_before_loading_pixels(monkeypatch,tmp_path):
    monkeypatch.setattr(a,'contract',lambda *args,**kw: {})
    def rows(path):
        assert path.name=='validation_processed.csv'
        return [dict(final_partition='test',official_partition='test',canonical_image_id='x')]
    monkeypatch.setattr(a,'read_rows',rows)
    monkeypatch.setattr(a.np,'load',lambda *args,**kw: pytest.fail('Pixels must not be loaded'))
    with pytest.raises(ValueError,match='Only'): a.validation_images(tmp_path)


def test_fixed_threshold_metrics():
    assert a.THRESHOLD==0.3753475248813629
    baseline=np.array([.1,.37532,.8,.9]); altered=np.array([.1,.4,.2,.9])
    metrics=a.robustness_metrics([0,0,1,1],altered,baseline)
    assert metrics['flip_count']==2 and metrics['flip_rate']==.5
    assert metrics['sensitivity']==.5 and metrics['specificity']==.5
    assert metrics['balanced_accuracy']==.5
    assert metrics['roc_auc']==.75
    assert metrics['brier_score']==pytest.approx(np.mean((altered-[0,0,1,1])**2))
    assert metrics['threshold']==a.THRESHOLD


def test_occlusion_is_matched_disjoint_deterministic_and_not_label_driven():
    image=np.full((224,224),.5,np.float32)
    heat=np.zeros_like(image);heat[:32,:32]=1
    pair,status=a.occlusion_pair(image,heat)
    assert status=='eligible' and pair==a.occlusion_pair(image,heat)[0]
    high,low=pair
    assert high['y']==0 and high['x']==0 and low['x']==32
    assert high['foreground']==low['foreground'] and high['brightness']==low['brightness']
    image[:32,:32]=np.linspace(.2,.8,1024,dtype=np.float32).reshape(32,32)
    before=image.copy(); result=a.occlude(image,high)
    np.testing.assert_array_equal(before,image)
    assert result[:32,:32].std()<1e-6
    np.testing.assert_array_equal(result[32:],image[32:])


@pytest.mark.parametrize('kind,reason',[('zero','zero_heatmap'),('uniform','no_disjoint_matched_lower_attention_patch'),('background','no_foreground_patch')])
def test_sanity_exclusions(kind,reason):
    image=np.full((224,224),.5,np.float32); heat=np.ones_like(image)
    if kind=='zero':heat[:]=0
    if kind=='background':image[:]=0
    assert a.occlusion_pair(image,heat)==(None,reason)


def test_cluster_bootstrap_preserves_negative_findings():
    rows=[dict(participant_id=str(i),drop_high=-.2,drop_low=.1) for i in range(3)]
    ci=a.paired_cluster_ci(rows)
    assert ci==a.paired_cluster_ci(rows)
    assert ci['mean_paired_difference']['upper']==pytest.approx(-.3)
    assert ci['high_larger_drop_fraction']['upper']==0


def test_no_overwrite_existing_experiment(tmp_path):
    (tmp_path/'reports/evidence/robustness').mkdir(parents=True)
    with pytest.raises(ValueError,match='already exists'): a.run(tmp_path)


def test_hash_verifier_detects_missing_modified_and_unsafe_files(tmp_path):
    from src.extended_validation.verify import digest,check_hashes
    path=tmp_path/'evidence.json';path.write_bytes(b'original\r\n')
    bindings={'evidence.json':dict(mode='bytes',sha256=digest(path))}
    before=path.read_bytes()
    assert check_hashes(tmp_path,bindings)==[] and path.read_bytes()==before
    path.write_bytes(b'changed')
    assert 'Hash mismatch' in check_hashes(tmp_path,bindings)[0]
    path.unlink()
    assert 'Missing' in check_hashes(tmp_path,bindings)[0]
    assert 'unsafe' in check_hashes(tmp_path,{'../outside':bindings['evidence.json']})[0]


def test_verifier_missing_manifest_fails_without_inference(tmp_path):
    from src.extended_validation.verify import verify
    result=verify(tmp_path)
    assert result['status']=='FAIL'
    assert result['inference_executed'] is False and result['test_pixels_opened'] is False
    assert list(tmp_path.iterdir())==[]


def test_verifier_import_is_standard_library_only():
    import subprocess,sys
    code="import sys; import src.extended_validation.verify; assert 'tensorflow' not in sys.modules; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True)


@pytest.mark.parametrize('change',['threshold','heatmap','layer','rule','receipt'])
def test_lock_semantics_reject_changes_without_inference(monkeypatch,tmp_path,change):
    from src.extended_validation import verify as v
    selection=dict(status='LOCKED_FOR_FINAL_TEST',run=str(v.Path(v.CHECKPOINT).parent).replace('\\','/'),
        threshold=v.THRESHOLD,checkpoint_sha256=v.CHECKPOINT_HASH,split_hashes={},threshold_sha256='x',config_sha256='x',run_metadata_sha256='x')
    lock=dict(status='LOCKED',classifier_threshold=v.THRESHOLD,heatmap_threshold=.5,target_layer='conv5_block3_out',
        identity=dict(checkpoint=v.CHECKPOINT,checkpoint_sha256=v.CHECKPOINT_HASH,classifier_selection_sha256='x',
                      coordinate_protocol_sha256='x',implementation_sha256={}),
        coordinate_rules=dict(equal_grid='IDENTITY',unequal_grid='EXCLUDE_UNPROVEN'),threshold_prespecified=True,
        tuning_after_validation=False,validation_run='validation',validation_summary_sha256='x',validation_artifact_sha256={},reporting_source_sha256='x')
    classifier=dict(status='COMPLETE',run='classifier',metrics_sha256='x');localisation=dict(status='COMPLETE',run='localisation',summary_sha256='x')
    if change=='threshold':selection['threshold']=.3753
    if change=='heatmap':lock['heatmap_threshold']=.6
    if change=='layer':lock['target_layer']='different'
    if change=='rule':lock['coordinate_rules']['unequal_grid']='RESIZE'
    if change=='receipt':classifier['status']='STARTED'
    fixtures={'final_selection.json':selection,'localisation_lock.json':lock,'final_test_attempt.json':classifier,'final_localisation_attempt.json':localisation}
    monkeypatch.setattr(v,'read',lambda root,p:fixtures[v.Path(p).name])
    monkeypatch.setattr(v,'check_hashes',lambda *args:[])
    assert v.check_locks(tmp_path)


def test_model_card_uses_evidence_values(monkeypatch,tmp_path):
    from src.extended_validation import report as r
    config=dict(project=dict(dataset='Synthetic research cohort',subset='mass',positive_class='MALIGNANT',input_size=224,preprocessing=dict(version='synthetic')),
        experiment=dict(model='resnet50',transfer=dict(weights='imagenet',fine_tuning_epochs=10,unfreeze_from='block',fine_tuning_learning_rate=1e-5),
                        epochs=30,learning_rate=.0001,optimizer='adam',loss='binary_crossentropy',batch_size=16,augmentation='none',class_weight=False,early_stopping_patience=7,threshold_method='youden_j'))
    selection=dict(rationale='Synthetic evidence rationale.',threshold=a.THRESHOLD,run='synthetic/run',checkpoint_sha256='abc',cache_fingerprint='cache')
    lock=dict(target_layer=a.TARGET,heatmap_threshold=.5,identity=dict(coordinate_protocol_sha256='def'))
    metrics=dict(partition='OFFICIAL HOLDOUT TEST',sensitivity='.123456',specificity='.6',balanced_accuracy='.5',roc_auc='.7',pr_auc='.6',brier_score='.2',false_negatives='9',false_negative_rate='.25')
    loc=dict(n=10,images=8,participants=5,pointing_game_rate=.1,lesion_hit_rate=.2,mean_iou=.01,mean_dice=.02,mean_activation_mass_inside=.03)
    fixtures={'extended_validation_sources.json':dict(config=config,selection=selection,localisation_lock=lock),
        'cohort_counts.csv':[dict(partition='train',images=24,participants=12)],'resnet50_validation_vs_test.csv':[metrics],
        'test_confidence_intervals.csv':[],'calibration_summary.csv':[],
        'validation_localisation/summary.json':dict(groups={'lesion/overall':loc}),
        'final_localisation/summary.json':dict(groups={'lesion/overall':loc}),
        'robustness_summary.csv':[], 'gradcam_sanity/summary.json':dict(n_eligible=8,n_total=10,participants=5,mean_drop_high=-.2,mean_drop_low=-.1,mean_paired_difference=-.1,high_larger_drop_fraction=.4),
        'stability_summary.json':dict(any_flip_count=3,groups={}),
        'baseline_consistency.json':dict(max_absolute_difference=.000001,classification_difference_count=1)}
    def load(root,p):
        assert 'predictions' not in p and 'data/' not in p
        return next(value for key,value in fixtures.items() if p.endswith(key))
    monkeypatch.setattr(r,'load',load)
    card=r.model_card(tmp_path)
    assert '0.123456' in card and 'Synthetic evidence rationale' in card
    assert 'not a calibrated clinical probability' in card and 'not clinically validated' in card
    assert '9** malignant' in card and str(a.THRESHOLD) in card


def test_diagrams_and_unfilled_review_templates(tmp_path):
    from src.extended_validation.report import diagrams,forms,load
    diagrams(tmp_path);forms(tmp_path)
    figure=tmp_path/'reports/evidence/figures'
    assert len(list(figure.glob('*.png')))==3 and len(list(figure.glob('*.svg')))==3
    for path in figure.glob('*.png'):assert path.read_bytes().startswith(b'\x89PNG')
    questions=load(tmp_path,'reports/evidence/radiologist_feedback/feedback_questions.csv')
    assert len(questions)==11 and all(not r['response'] for r in questions)
    checklist=(tmp_path/'reports/evidence/app/manual_acceptance_checklist.md').read_text(encoding='utf-8')
    assert checklist.count('| PENDING |')==17
    form=tmp_path/'reports/evidence/radiologist_feedback/feedback_form.md'
    form.write_text('Synthetic human response',encoding='utf-8')
    forms(tmp_path)
    assert form.read_text(encoding='utf-8')=='Synthetic human response'
