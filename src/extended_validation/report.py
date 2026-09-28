"""Generate small reports from saved evidence; no inference or cohort evaluation."""
import csv
import json
from pathlib import Path
import platform
import subprocess

import yaml

from .verify import ROOT,CHECKPOINT,digest,verify


def load(root,relative):
    with (root/relative).open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle)) if relative.endswith('.csv') else json.load(handle)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def text(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value.rstrip()+'\n',encoding='utf-8')


def source_snapshot(root):
    """Curate existing hash-bound training metadata for the committed model card."""
    selection=load(root,'results/experiments/final_selection.json');run=selection['run']
    paths=[run+'/config_used.yaml',run+'/run_metadata.json','results/experiments/final_selection.json',
           'results/explainability/localisation_lock.json']
    hashes={p:digest(root/p) for p in paths}
    if hashes[run+'/config_used.yaml']!=selection['config_sha256'] or hashes[run+'/run_metadata.json']!=selection['run_metadata_sha256']:
        raise ValueError('Frozen training metadata changed')
    config=yaml.safe_load((root/paths[0]).read_text());metadata=load(root,paths[1])
    snapshot=dict(config=config,training_environment=metadata['environment'],reproducibility=metadata['reproducibility'],
        selection={k:v for k,v in selection.items() if k!='selection_participants'},
        localisation_lock=load(root,paths[3]),source_sha256=hashes)
    save(root/'reports/evidence/extended_validation_sources.json',snapshot)


def model_card(root):
    evidence='reports/evidence/'
    sources=load(root,evidence+'extended_validation_sources.json')
    config=sources['config'];selection=sources['selection'];lock=sources['localisation_lock'];experiment=config['experiment']
    cohorts=load(root,evidence+'model_evaluation/cohort_counts.csv')
    metrics=load(root,evidence+'model_evaluation/resnet50_validation_vs_test.csv')
    intervals=load(root,evidence+'model_evaluation/test_confidence_intervals.csv')
    calibration=load(root,evidence+'model_evaluation/calibration_summary.csv')
    val=load(root,evidence+'validation_localisation/summary.json')['groups']['lesion/overall']
    final=load(root,evidence+'final_localisation/summary.json')['groups']['lesion/overall']
    robust=load(root,evidence+'robustness/robustness_summary.csv')
    sanity=load(root,evidence+'gradcam_sanity/summary.json')
    stability=load(root,evidence+'robustness/stability_summary.json')
    consistency=load(root,evidence+'robustness/baseline_consistency.json')
    official=next(r for r in metrics if r['partition']=='OFFICIAL HOLDOUT TEST')
    lines=['# MammoAssist AI model card','',
        '## Purpose, users and permitted scope','',
        'Research decision-support prototype for qualified clinicians reviewing an academic demonstration and researchers assessing reproducibility. '
        'It classifies full mammograms from the CBIS-DDSM mass subset; it does not detect, segment or diagnose lesions. '
        'Not a medical device, not clinically validated, and not for autonomous diagnosis, screening deployment, patient management or ruling disease in/out. '
        'Do not input ROI masks or lesion crops. The malignant-pattern score is not a calibrated clinical probability of cancer.','',
        '## Dataset and split','',
        f"Dataset: {config['project']['dataset']}; scope: {config['project']['subset']}; positive class: {config['project']['positive_class']}. "
        'BENIGN and BENIGN_WITHOUT_CALLBACK form the negative class. Digitised film mammograms differ from contemporary digital equipment. '
        'This mass-only, selected retrospective cohort cannot represent general screening prevalence, calcification-only disease or other populations. '
        'Official Training was partitioned by participant into development Train/Validation. The official Test holdout was kept separate; participant overlap checks underpin the split.','',
        '| Partition | Full mammograms | Participants |','| --- | ---: | ---: |']
    lines += [f"| {r['partition']} | {r['images']} | {r['participants']} |" for r in cohorts]
    lines += ['', 'Counts are the processed classifier cohorts, not raw download or lesion counts. Source: `model_evaluation/cohort_counts.csv`.','',
        '## Preprocessing and model development','',
        f"Preprocessing version `{config['project']['preprocessing']['version']}`: deterministic grayscale decoding including MONOCHROME inversion, "
        'safe empty-border crop, aspect-preserving resize and black padding; non-CLAHE inference. '
        f"Input {config['project']['input_size']} x {config['project']['input_size']}; grayscale replicated into RGB, scaled and adapted using the existing ResNet50 preprocessing adapter. "
        f"Selected architecture: {experiment['model']}, {experiment['transfer']['weights']} transfer learning. "
        f"Frozen-backbone head training up to {experiment['epochs']} epochs, followed by up to {experiment['transfer']['fine_tuning_epochs']} fine-tuning epochs "
        f"from `{experiment['transfer']['unfreeze_from']}`; head learning rate {experiment['learning_rate']}, fine-tuning rate {experiment['transfer']['fine_tuning_learning_rate']}. "
        f"Optimizer {experiment['optimizer']}, loss {experiment['loss']}, batch size {experiment['batch_size']}, augmentation {experiment['augmentation']}, "
        f"class weighting {experiment['class_weight']}; early-stopping patience {experiment['early_stopping_patience']}. These are configured maxima, not claimed completed epoch counts.", '',
        'Compared majority reference, MammoCNN baseline, regularised MammoCNN, VGG16, ResNet50 and EfficientNetB0 using Validation. '+selection['rationale'], '',
        f"Threshold selection used `{experiment['threshold_method']}` on Validation. Fixed threshold: **{selection['threshold']}**. "
        'Model/threshold selection was locked before the one-time official classifier Test evaluation. No subsequent recalibration, retraining or threshold/layer selection occurred.','',
        '## Frozen official classifier Test results','',
        '| Measure | Frozen value |','| --- | ---: |']
    for key in ('sensitivity','specificity','balanced_accuracy','roc_auc','pr_auc','brier_score'):
        lines.append(f"| {key} | {float(official[key]):.6f} |")
    lines += ['', 'PR-AUC uses average precision. These are research holdout results, not clinical certification.','',
              '| Measure | Participant-cluster bootstrap 95% CI | Replicates |','| --- | --- | ---: |']
    lines += [f"| {r['metric']} | {float(r['lower_95']):.6f} to {float(r['upper_95']):.6f} | {r['valid_replicates']} |" for r in intervals]
    lines += ['', f"Frozen false negatives: **{official['false_negatives']}** malignant full mammograms; false-negative rate **{float(official['false_negative_rate']):.2%}**. "
        'Low scores cannot exclude disease. Intervals reflect participant resampling within this cohort, not uncertainty across hospitals or populations.','',
        'Calibration was assessed descriptively, not corrected using Test. '
        +'; '.join(f"{r['partition']}: mean score {float(r['mean_probability']):.4f}, observed malignant fraction {float(r['observed_malignant_fraction']):.4f}" for r in calibration)+'. '
        'Brier score and calibration plots do not establish clinical probability calibration. Sources: `model_evaluation/resnet50_validation_vs_test.csv`, `test_confidence_intervals.csv`, `calibration_summary.csv`.','',
        '## Explainability and frozen localisation','',
        f"Genuine gradient-derived Grad-CAM at `{lock['target_layer']}`. Prespecified localisation heatmap threshold: {lock['heatmap_threshold']}. "
        'Full mammogram is the model input; ROI masks are references only. Exact equal-grid masks use validated identity mapping; unequal-grid masks are excluded because registration is unproven. '
        'Exclusion is geometry-based, independent of pathology/model outputs. Human SUPPORTED review concerns coordinate correspondence, not clinical contour correctness.','',
        '| Frozen cohort | Lesions | Images | Participants | Pointing rate | Hit rate | Mean IoU | Mean Dice | Mean activation mass |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,r in [('VALIDATION',val),('OFFICIAL TEST LOCALISATION',final)]:
        lines.append(f"| {name} | {r['n']} | {r['images']} | {r['participants']} | {r['pointing_game_rate']:.2%} | {r['lesion_hit_rate']:.2%} | {r['mean_iou']:.6f} | {r['mean_dice']:.6f} | {r['mean_activation_mass_inside']:.6f} |")
    lines += ['', 'These are lesion-level results; individual and image-union results remain in the original summaries. Different classification/localisation cohort sizes must not be conflated. '
        'Overlap was low. Grad-CAM is a model-attention visualisation, not a validated detector or lesion boundary. '
        'Sources: `validation_localisation/summary.json`, `final_localisation/summary.json`.','',
        '## Extended Validation-only analysis','',
        'This is a post-hoc exploratory extension undertaken after the core frozen evaluations, not a retrospectively preregistered experiment or independent/external validation. '
        'Parameters were fixed before new Validation inference. No Test pixels/predictions were used; frozen Test summaries above are read-only. No tuning followed these findings.','',
        '| Stress | Flip rate | Mean absolute score change | ROC-AUC |','| --- | ---: | ---: | ---: |']
    lines += [f"| {r['perturbation']} | {float(r['flip_rate']):.2%} | {float(r['mean_absolute_probability_change']):.6f} | {float(r['roc_auc']):.6f} |" for r in robust]
    lines += ['', f"Images flipping under at least one stress: {stability['any_flip_count']}. "
        +'; '.join(f"{name}: {g['any_flip_count']}/{g['n']} ({g['any_flip_rate']:.2%})" for name,g in stability['groups'].items())+'. '
        'These baseline-conditioned descriptive rates also depend on threshold margin; they do not establish causal vulnerability.','',
        f"The fresh unperturbed Validation baseline differs from saved probabilities by at most {consistency['max_absolute_difference']:.9g}; "
        f"{consistency['classification_difference_count']} case(s) cross the exact fixed threshold due to numerical differences near the boundary. "
        'Extended stress comparisons use the fresh baseline consistently. The original frozen Validation metrics are preserved; no threshold tolerance or rounding is applied to decisions.','',
        f"Matched Grad-CAM occlusion: {sanity['n_eligible']}/{sanity['n_total']} images eligible, across {sanity['participants']} participants. "
        f"Mean score drop high-attention {sanity['mean_drop_high']:.6f}, low-attention {sanity['mean_drop_low']:.6f}; "
        f"paired difference {sanity['mean_paired_difference']:.6f}; high-attention larger-drop fraction {sanity['high_larger_drop_fraction']:.2%}. "
        'See `gradcam_sanity/summary.md` for exclusion counts and paired participant-bootstrap intervals. '
        'Occlusion changes the input distribution, approximate matching does not ensure equivalent anatomy, and intervention sensitivity cannot prove faithfulness or clinical localisation.','',
        '## Limitations, ethics and reproducibility','',
        'Small retrospective dataset; repeated use of Validation for development; domain and prevalence shift; limited spatial resolution; no prospective, multi-centre or external validation; '
        'demographic subgroup fairness is not established. Digital perturbations are approximations, not measured acquisition changes. Multiple stresses and correlated views are descriptive; '
        'no significance search or claimed clinical benefit. Protect image privacy, check burned-in text, and avoid automation bias. '
        'The app is a session-only research prototype, not production authentication or a clinical record system. Expert feedback forms are pending and must not be mistaken for collected evidence.','',
        f"Checkpoint: `{selection['run']}/best_model.keras`\n\nSHA256: `{selection['checkpoint_sha256']}`\n\n"
        f"Cache fingerprint: `{selection['cache_fingerprint']}`\n\nCoordinate protocol SHA256 (LF-normalised): `{lock['identity']['coordinate_protocol_sha256']}`.", '',
        'All numerical performance values above are loaded from the cited saved evidence; training settings and identifiers are curated in `extended_validation_sources.json`, committed with this card. '
        'The full hash index, environments and reproduction commands are in `reproducibility_manifest.json` and `reproducibility_check.md`. '
        'Frozen final Test evaluations must not be rerun.']
    return '\n'.join(lines)+'\n'


def interpretation(root):
    """Read-only analysis of saved Validation outputs; do not regenerate inference."""
    folder=root/'reports/evidence'; plan=load(root,'reports/evidence/robustness/robustness_config.json')
    threshold=plan['classifier_threshold']
    saved=load(root,str(Path(CHECKPOINT).parent.as_posix())+'/validation/predictions.csv')
    if any(r['final_partition']!='validation' or r['official_partition']!='train' for r in saved):
        raise ValueError('Validation predictions only')
    saved={r['canonical_image_id']:r for r in saved}
    baseline=[r for r in load(root,'reports/evidence/robustness/robustness_per_image.csv') if r['perturbation']=='original']
    if any(r['partition']!='validation' for r in baseline):raise ValueError('Validation outputs only')
    changed=[];maxdiff=0
    for row in baseline:
        original=float(saved[row['canonical_image_id']]['predicted_probability']);fresh=float(row['probability'])
        maxdiff=max(maxdiff,abs(original-fresh))
        if (original>=threshold)!=(fresh>=threshold):
            changed.append(dict(canonical_image_id=row['canonical_image_id'],saved_probability=original,
                fresh_probability=fresh,threshold=threshold,saved_class=int(original>=threshold),fresh_class=int(fresh>=threshold)))
    save(folder/'robustness/baseline_consistency.json',dict(max_absolute_difference=maxdiff,
        classification_difference_count=len(changed),cases=changed,explanation='Floating-point differences at an exact threshold boundary. Fresh baseline used for all extension comparisons; frozen Validation evidence unchanged.'))
    summary_path=folder/'robustness/robustness_summary.md'
    summary_text=summary_path.read_text(encoding='utf-8').split('\n## Fresh baseline precision note')[0]
    text(summary_path,summary_text.rstrip()+'\n\n## Fresh baseline precision note\n\n'
        f"Maximum difference from saved Validation scores: {maxdiff:.9g}; {len(changed)} threshold decision(s) changed at the numerical boundary. "
        'The extension consistently uses fresh unperturbed scores as its baseline, with the exact locked threshold. '
        'Original frozen Validation sensitivity and balanced accuracy therefore differ slightly; they were not overwritten. '
        'See `baseline_consistency.json` for the affected case and exact values.')
    sanity=load(root,'reports/evidence/gradcam_sanity/summary.json');ci=sanity['paired_participant_cluster_95ci']
    text(folder/'gradcam_sanity/summary.md','# Validation-only Grad-CAM sanity analysis\n\n'
        f"Matched comparisons: {sanity['n_eligible']}/{sanity['n_total']} images, {sanity['participants']} participants. "
        f"Status counts: {sanity['status_counts']}.\n\n"
        '| Measure | Estimate | Participant-cluster 95% percentile CI |\n| --- | ---: | --- |\n'
        +'\n'.join(f"| {key} | {sanity[key]:.6f} | {ci[key]['lower']:.6f} to {ci[key]['upper']:.6f} |" for key in ci)+'\n\n'
        'Negative drop means the malignant-pattern score increased after occlusion. '
        f"High-attention occlusion increased scores in {sanity['high_score_increase_count']} matched images, "
        f"and low-attention occlusion increased scores in {sanity['low_score_increase_count']}.\n\n"
        +('This check does **not support** the hypothesised greater score reduction from high-attention occlusion. '
          if sanity['mean_paired_difference']<=0 else 'The mean paired difference favours high-attention occlusion under this specific intervention. ')
        +'It cannot establish general faithfulness: replacing texture by local mean is distribution-shifting and approximate foreground/brightness matching does not guarantee equivalent tissue. '
        'No conclusion about clinical lesion detection follows. All signs and exclusions are retained; no parameters were retuned.\n\n'
        f"Bootstrap: {sanity['bootstrap_iterations']} participant-cluster replicates, fixed seed in `protocol.json`, resampling paired outcomes together. "
        'This is a descriptive exploratory interval, not a multiple-comparison-adjusted confirmatory test. '
        'All input images were Validation full mammograms; no ROI masks, Test inputs or Test predictions were used.')
    stability=load(root,'reports/evidence/robustness/stability_summary.json')
    text(folder/'robustness/prediction_stability.md','# Validation prediction stability\n\n'
        f"{stability['any_flip_count']} images flip under at least one fixed stress.\n\n"
        '| Fresh baseline group | Images | Any-stress flips | Rate |\n| --- | ---: | ---: | ---: |\n'
        +'\n'.join(f"| {key} | {g['n']} | {g['any_flip_count']} | {g['any_flip_rate']:.2%} |" for key,g in stability['groups'].items())+'\n\n'
        'False negatives show a higher observed flip fraction than correctly classified malignant cases under these stresses. '
        'This is descriptive: group sizes differ, closeness to the threshold is a confounder, and flips in baseline false negatives are toward the positive class. '
        'It is not evidence of a causal instability mechanism or improved diagnostic utility.\n\n'
        f"Fresh baseline versus saved Validation: maximum score difference {maxdiff:.9g}; {len(changed)} changed threshold decision(s). "
        'See `baseline_consistency.json`. Frozen results remain unchanged.\n\n'
        'The ten closest cases (absolute distance, then canonical ID) are saved in `stability_summary.json`. '
        'All cases and flip counts are in `prediction_stability.csv`. The figure shows margins and signed distributions without selecting successful cases.')


def diagrams(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']='mammoassist_extended_v1'
    from matplotlib.patches import FancyBboxPatch
    folder=root/'reports/evidence/figures';folder.mkdir(parents=True,exist_ok=True)
    def draw(name,title,nodes,edges,foot):
        fig,ax=plt.subplots(figsize=(13,6));ax.set(xlim=(-.2,10.2),ylim=(-.6,4.9));ax.axis('off')
        for key,(x,y,label) in nodes.items():
            ax.add_patch(FancyBboxPatch((x-1.05,y-.35),2.1,.7,boxstyle='round,pad=0.05',facecolor='#e9f1f5',edgecolor='#245469',lw=1.2))
            ax.text(x,y,label,ha='center',va='center',fontsize=10,color='#173a49')
        for start,end in edges:
            x,y,_=nodes[start];u,v,_=nodes[end]
            dx,dy=u-x,v-y
            if abs(dx)>abs(dy): a=(x+1.12*(1 if dx>0 else -1),y);b=(u-1.12*(1 if dx>0 else -1),v)
            else:a=(x,y+.42*(1 if dy>0 else -1));b=(u,v-.42*(1 if dy>0 else -1))
            ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#245469',lw=1.4))
        ax.set_title(title,fontsize=16,pad=12);ax.text(5,-.35,foot,ha='center',va='center',fontsize=9,color='#444444')
        fig.tight_layout()
        for extension in ('png','svg'):
            fig.savefig(folder/(name+'.'+extension),dpi=200,bbox_inches='tight',metadata={'Date':None} if extension=='svg' else None)
        plt.close(fig)
    draw('system_architecture','MammoAssist AI: research prototype',
        {'u':(1,4,'Full-mammogram\nupload'),'v':(3.65,4,'Format / input\nquality checks'),
         'p':(6.3,4,'Locked canonical\npreprocessing'),'m':(9,4,'Locked ResNet50\nfull-image input'),
         's':(9,2,'Malignant-pattern\nscore'),'g':(6.3,2,'Gradient-derived\nGrad-CAM'),
         'r':(3.65,2,'Research result\nand limitations'),'e':(1,2,'PNG / PDF\nresearch export')},
        [('u','v'),('v','p'),('p','m'),('m','s'),('s','g'),('g','r'),('r','e')],
        'Grad-CAM uses the same model/input and output gradient. Attention is not a lesion detector; score is not clinical probability.')
    draw('experimental_protocol','Classifier experimental protocol',
        {'o':(1,4,'Official Training'),'p':(3.65,4,'Participant-safe\nTrain / Validation'),
         'd':(6.3,4,'Development and\nValidation selection'),'l':(9,4,'Model / threshold\nlock'),
         't':(9,2,'Official Test\nevaluate once'),'f':(6.3,2,'Frozen results\nread-only reporting'),
         'e':(3.65,.7,'Extended Validation\nanalysis; no tuning')},
        [('o','p'),('p','d'),('d','l'),('l','t'),('t','f'),('p','e')],
        'No Test-to-selection feedback. The exploratory extension uses only Validation inputs after the core results were frozen.')
    draw('explainability_protocol','Grad-CAM localisation research protocol',
        {'i':(1,4,'Full mammogram'),'g':(3.65,4,'Locked model\ngenuine Grad-CAM'),
         'r':(1,2,'Reference ROI masks'),'a':(3.65,2,'Validated identity grid\nunequal grids excluded'),
         'm':(6.3,3,'Validation localisation\nfixed metric definitions'),
         'l':(9,3,'Protocol + Validation\nresult lock'),'t':(9,.7,'Official Test once\nthen frozen')},
        [('i','g'),('r','a'),('g','m'),('a','m'),('m','l'),('l','t')],
        'ROI masks never enter the classifier. New matched-occlusion analysis is separate from these completed localisation evaluations.')


def forms(root):
    folder=root/'reports/evidence/radiologist_feedback'
    def blank_text(path,value):
        # Rebuilding derived reports must not erase later human responses.
        if not path.exists(): text(path,value)
    questions=[('image','Clarity of uploaded full-mammogram presentation'),('score','Clarity of malignant-pattern score wording'),
        ('threshold','Usefulness of fixed research threshold presentation'),('attention','Clarity of Grad-CAM attention map'),
        ('warnings','Usefulness of warnings and interpretation limitations'),('workflow','Perceived workflow usability')]
    scale='1 = very poor; 2 = poor; 3 = adequate; 4 = good; 5 = very good; N/A = unable to assess'
    blank_text(folder/'feedback_form.md','# Expert usability feedback — not clinical validation\n\n'
        'Reviewer code: ____  Date: ____  Relevant experience (optional): ____\n\n'
        'Participation is voluntary. Review takes approximately 10–15 minutes. Use de-identified Validation demonstrations only. '
        'Do not assess diagnostic accuracy from this small demonstration or enter patient information. No responses have been collected in this template.\n\n'
        +scale+'\n\n| Item | Rating (blank until reviewed) | Comment |\n| --- | --- | --- |\n'
        +'\n'.join('| '+q+' | | |' for _,q in questions)+'\n\n'
        'Could any wording mislead clinicians? Yes / No / Unsure: ____ Exact wording and concern: ____\n\n'
        '1. What was most difficult to understand or use?\n2. How would you explain the score and attention map after viewing them?\n'
        '3. Which warning or wording should be changed, and why?\n4. What single improvement would most help the research demonstration?\n\n'
        'Feedback supports usability refinement only. It does not certify clinical safety, diagnostic accuracy or efficacy.')
    folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'feedback_questions.csv').exists():
        with (folder/'feedback_questions.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle);writer.writerow(['id','question','response_scale','response','comment'])
            for key,q in questions:writer.writerow([key,q,scale,'',''])
            writer.writerow(['misleading','Could any wording mislead clinicians?','Yes / No / Unsure','',''])
            for i,q in enumerate(['What was difficult to understand or use?','Explain the score and attention map in your own words.','Which warning or wording needs changing and why?','What single improvement would help most?'],1):
                writer.writerow([f'open_{i}',q,'Free text','',''])
    blank_text(folder/'evaluation_protocol.md','# Expert usability feedback protocol\n\n'
        'Purpose: formative expert usability feedback, NOT clinical validation. Recruit one radiologist voluntarily; obtain permission to quote anonymised comments. '
        'Do not send this package automatically or imply a response exists.\n\n'
        '1. Briefing (2 minutes): explain research scope, uncalibrated score, attention limitations and frozen model.\n'
        '2. Demonstration (5 minutes): use the Validation case recorded in `../app/app_smoke.json`; upload, inspect quality checks, '
        'view score and Grad-CAM, open limitations and export. This is a fixed demonstration, not a diagnostic reader study.\n'
        '3. Ratings and open comments (5–8 minutes): reviewer completes the form independently; facilitator records only clarification requests and consented comments.\n\n'
        'Keep responses separate from this blank template, with reviewer code/date/app commit. Report item ratings and comments descriptively; '
        'do not calculate diagnostic performance, imply representative clinician consensus, or tune scientific settings. '
        'Record suggested UI improvements as future work; do not alter the frozen evaluation.')
    items=['Dark mode','Light mode','Narrow browser width','Validation full-mammogram DICOM upload','PNG upload','JPEG upload',
        'Invalid/corrupt file rejection','Recognisable ROI/cropped file rejection (unmarked crops cannot be reliably detected)',
        'Replace upload clears previous result','Clear session removes uploaded image/results','Score wording is not clinical probability',
        'Threshold displays 0.3753; full precision remains in technical details','Grad-CAM heatmap and overlay display',
        'PNG research export','PDF research export','PDF pagination and readable text','Research disclaimers visible']
    blank_text(root/'reports/evidence/app/manual_acceptance_checklist.md','# Manual app acceptance checklist\n\n'
        'Tester: ____  Date: ____  App commit: ____  Browser/version: ____  Validation image reference: ____\n\n'
        'All entries are pending. Record PASS / FAIL / N/A with evidence and reason. Use Validation or synthetic fixtures only; '
        'no official Test development images. Do not include patient identifiers in screenshots.\n\n'
        '| Check | Status | Observation / evidence |\n| --- | --- | --- |\n'+'\n'.join('| '+item+' | PENDING | |' for item in items))


def manifest(root):
    source=load(root,'reports/evidence/extended_validation_sources.json');selection=source['selection']
    paths={CHECKPOINT,'results/experiments/final_selection.json','results/experiments/final_test_attempt.json',
           'results/explainability/localisation_lock.json','results/explainability/final_localisation_attempt.json',
           'reports/evidence/roi_alignment_protocol.json','reports/evidence/preprocessing_validation.json',
           'reports/evidence/extended_validation_sources.json','reports/evidence/model_card.md','config/project.yaml','config/localisation.yaml'}
    paths.update(source['source_sha256'])
    paths.update('data/splits/'+part+'.csv' for part in selection['split_hashes'])
    paths.update(load(root,'reports/evidence/model_evaluation/provenance.json')['source_sha256'])
    for folder in ('robustness','gradcam_sanity','model_evaluation','final_localisation','validation_localisation','radiologist_feedback','figures'):
        paths.update(p.relative_to(root).as_posix() for p in (root/'reports/evidence'/folder).rglob('*') if p.is_file())
    paths.add('reports/evidence/app/manual_acceptance_checklist.md')
    paths.update(p.relative_to(root).as_posix() for p in (root/'src/extended_validation').glob('*.py'))
    history=subprocess.check_output(['git','log','-12','--format=%H %s'],cwd=root,text=True).splitlines()
    result=dict(version='extended_evidence_v1',git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        important_git_history=history,local_environment=load(root,'reports/evidence/robustness/execution.json'),
        training_environment=source['training_environment'],training_environment_note='GPU device presence recorded; GPU model name and training Git commit not recorded.',
        checkpoint=CHECKPOINT,checkpoint_sha256=selection['checkpoint_sha256'],classifier_threshold=selection['threshold'],
        gradcam_target_layer=source['localisation_lock']['target_layer'],heatmap_threshold=source['localisation_lock']['heatmap_threshold'],
        split_sha256=selection['split_hashes'],preprocessing_cache_fingerprint=selection['cache_fingerprint'],
        coordinate_protocol_sha256=source['localisation_lock']['identity']['coordinate_protocol_sha256'],
        coordinate_protocol_hash_mode='lf',files={p:dict(mode='bytes',sha256=digest(root/p)) for p in sorted(paths)},
        note='Raw bytes unless explicitly LF-normalised. No raw medical pixels embedded. Historical and new evidence hashes; not a cryptographic authenticity signature. Final Test receipts must never be removed to enable reruns.',
        commands=dict(read_only_verify='.\\.venv\\Scripts\\python.exe -m src.extended_validation.verify',
                      regenerate_reports_no_inference='.\\.venv\\Scripts\\python.exe -m src.extended_validation.report',
                      validation_experiment_fresh_checkout_only='.\\.venv\\Scripts\\python.exe -m src.extended_validation.analysis',
                      synthetic_tests='.\\.venv\\Scripts\\python.exe -m pytest tests/test_extended_validation.py -q'),
        historical_non_test_commands_fresh_development_checkout_only=[
            '.\\.venv\\Scripts\\python.exe -m src.preprocessing.cache --split train',
            '.\\.venv\\Scripts\\python.exe -m src.preprocessing.cache --split validation',
            '.\\.venv\\Scripts\\python.exe -m src.training.train --experiment resnet50'],
        historical_commands_note='Documentation only; not executed in this extension. Requires audited dataset and fresh pre-lock development checkout. Never unlock/retrain this frozen project.')
    save(root/'reports/evidence/reproducibility_manifest.json',result)


def run(root=ROOT):
    source_snapshot(root)
    interpretation(root)
    text(root/'reports/evidence/model_card.md',model_card(root))
    diagrams(root);forms(root);manifest(root)
    result=verify(root)
    text(root/'reports/evidence/reproducibility_check.md','# Reproducibility check\n\n'
        +'```json\n'+json.dumps(result,indent=2)+'\n```\n\n'
        'Run from the project root in PowerShell:\n\n```powershell\n.\\.venv\\Scripts\\python.exe -m src.extended_validation.verify\n```\n\n'
        'The verifier is read-only and uses only the Python standard library. It hashes saved artifacts, checks lock identities and completion receipts, '
        'and never imports TensorFlow, deserialises the model, opens Test pixels or executes inference. Missing artifacts fail explicitly. '
        'Raw inputs, caches and weights must be obtained separately; they are not committed.\n\n'
        'Rebuild derived reports without inference:\n\n```powershell\n.\\.venv\\Scripts\\python.exe -m src.extended_validation.report\n```\n\n'
        'For an independently prepared fresh reproduction checkout with the same Validation inputs and no existing extended output directories:\n\n'
        '```powershell\n.\\.venv\\Scripts\\python.exe -m src.extended_validation.analysis\n```\n\n'
        'The experiment refuses to overwrite existing evidence. Do not delete current evidence to bypass this guard. '
        'Historical non-Test preprocessing/training commands are recorded in the manifest and described in `src/preprocessing/README.md` and `src/training/README.md`. '
        'They require a separate pre-lock development checkout; this extension does not retrain or unlock the current project. '
        'Hardware/library differences may change floating-point results; saved Validation baseline agreement is checked before stresses. '
        'Final classifier Test and final Test localisation are complete: never rerun casually, remove their receipts or use Test to select settings. '
        'This manifest does not bind itself or this check report; Git provides their version history. Human feedback/checklists remain pending.')
    if result['status']!='PASS': raise ValueError(result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
