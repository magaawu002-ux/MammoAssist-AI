"""Run with: python -m streamlit run src/app/streamlit_app.py."""

from pathlib import Path
import sys

# Streamlit executes the file as a script; resolve imports independently of cwd.
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from datetime import datetime,timezone
import hashlib
import uuid

import streamlit as st

from src.app.core import (TITLE,DISCLAIMER,SCORE_NOTE,ATTENTION_NOTE,DOMAIN_NOTE,AppError,
                         CHECKPOINT,contract,prepare_upload,analyse)
from src.app.export import png_report,pdf_report
from src.app.runtime import model_resource
from src.app.dashboard import dashboard_evidence,validation_evidence
from src.app.display import attention_view
from src.app.session import add_case,clear_history,comparison,near_threshold,COMPARISON_NOTE,SESSION_NOTE
from src.utils.paths import PROJECT_ROOT


def clear_upload():
    clear_history(st.session_state)
    for key in ('prepared','result','upload_digest','anonymous_reference','timestamp','exports','confirmed','attention_mode','attention_opacity'):
        st.session_state.pop(key,None)
    previous=st.session_state.get('upload_generation',0)
    st.session_state.pop('upload_'+str(previous),None)
    st.session_state['upload_generation']=previous+1


def render_session():
    history=st.session_state.get('case_history',[])
    with st.expander('Session case history'):
        st.caption(SESSION_NOTE)
        st.caption('Only canonical previews and attention overlays are retained in memory for comparison; original uploads are not stored in history.')
        st.button('Clear session history',on_click=lambda:clear_history(st.session_state))
        if history:
            st.dataframe([{'Case':r['case_number'],'Timestamp (UTC)':r['timestamp'],'Format':r['format'],
                'Resolution':f"{r['width']} × {r['height']}",'Score':f"{r['score']:.4f}",
                'Research threshold result':'Above or equal' if r['score']>=r['threshold'] else 'Below',
                'Near threshold':'Caution' if r.get('near_threshold',near_threshold(r['score'])) else 'No',
                'Quality':f"{r['quality_warning_count']} heuristic warnings" if r['quality_warning_count'] else 'No heuristic warnings',
                'Grad-CAM':'Available' if r['gradcam_available'] else 'Unavailable'} for r in history],hide_index=True,width='stretch')
        else: st.caption('No completed analyses in this session.')
    with st.expander('Research comparison view'):
        st.caption('Compare two completed analyses from this active session. No additional inference is performed.')
        if len(history)<2:
            st.write('Analyse at least two images in this session to enable comparison.');return
        ids=[r['case_number'] for r in history]
        a,b=st.columns(2)
        left=a.selectbox('First case',ids,key='compare_left',format_func=lambda n:f'Case {n}')
        right=b.selectbox('Second case',ids,index=1,key='compare_right',format_func=lambda n:f'Case {n}')
        if left==right:st.info('Choose two different session cases.');return
        selected=comparison(st.session_state,left,right)
        st.metric('Absolute difference in model score',f"{selected['absolute_score_difference']:.4f}",help='Arithmetic difference only; no clinical interpretation.')
        st.info(COMPARISON_NOTE)
        for col,record,view in zip(st.columns(2),selected['records'],selected['views']):
            with col:
                st.markdown(f"**Case {record['case_number']}** · {record['timestamp']}")
                st.metric('Malignant-pattern score',f"{record['score']:.4f}")
                st.caption(f"Research threshold: {record['threshold']:.4f}")
                st.write(record['decision'])
                if near_threshold(record['score']):st.warning('Near research decision threshold')
                st.image(view['canonical'],caption='Canonical full mammogram',clamp=True,width='stretch')
                st.image(view['overlay'],caption='Grad-CAM model-attention overlay',clamp=True,width='stretch')
                st.caption(f"Input-quality heuristics · {record['width']} × {record['height']} · brightness {view['brightness']:.3f} · contrast {view['contrast']:.3f}")
                for warning in view['warnings']:st.warning(warning)
                if not view['warnings']:st.caption('No heuristic warnings; not clinical image-quality certification.')
                if view['zero_heatmap']:st.warning('Zero positive attention map; this does not establish absence of an abnormality.')
        st.caption(SCORE_NOTE);st.caption(ATTENTION_NOTE)


def render_dashboard():
    with st.expander('Research evidence dashboard'):
        st.write('Performance on the frozen CBIS-DDSM official holdout used in this project.')
        st.caption('These are research evaluation results and do not constitute clinical validation.')
        try:
            evidence=dashboard_evidence()
            st.caption(f"{evidence['images']} full mammograms · {evidence['participants']} participants · saved aggregate evidence only")
            keys=('sensitivity','specificity','balanced_accuracy','roc_auc','pr_auc')
            for group in (keys[:3],keys[3:]):
                for col,key in zip(st.columns(len(group)),group):
                    label={'roc_auc':'ROC-AUC','pr_auc':'PR-AUC'}.get(key,key.replace('_',' ').capitalize())
                    col.metric(label,f"{evidence['metrics'][key]:.3f}",help='Ranking discrimination across thresholds; not performance at one operating point.' if key=='roc_auc' else None)
                    if key in evidence['confidence_intervals']:
                        lo,hi=evidence['confidence_intervals'][key]
                        col.caption(f'95% participant-cluster bootstrap CI: {lo:.3f}–{hi:.3f}')
            a,b=st.columns(2)
            a.metric('False-negative malignant mammograms',str(evidence['false_negatives']))
            b.metric('Brier score',f"{evidence['metrics']['brier_score']:.3f}")
            st.caption(f"False-negative rate among malignant cases: {evidence['false_negative_rate']:.2%}")
            st.markdown('**Explainability evaluation — frozen official Test localisation**')
            local=evidence['localisation']
            for col,key,label in zip(st.columns(2),('pointing_game_rate','lesion_hit_rate'),('Lesion-level pointing rate','Lesion-hit rate')):
                col.metric(label,f"{local[key]:.2%}")
            for col,key,label in zip(st.columns(2),('mean_iou','mean_dice'),('Mean lesion IoU','Mean lesion Dice')):
                col.metric(label,f"{local[key]:.4f}")
            st.write('Grad-CAM localisation overlap was low in this study.')
            st.caption('Lesion-level metrics use eligible equal-grid annotations; their cohort differs from the classifier holdout. Grad-CAM is not a lesion detector or validated boundary.')
        except Exception:
            st.warning('Frozen research evidence is unavailable or inconsistent; no replacement performance values are generated.')
        st.markdown('**Additional Validation analyses**')
        try:
            extra=validation_evidence()
            a,b=st.columns(2)
            a.metric('Robustness flip-rate range',f"{extra['flip_min']:.1%}–{extra['flip_max']:.1%}")
            b.metric('Validation images flipping under ≥1 stress',f"{extra['any_flip_count']} / {extra['images']}")
            st.caption('Most disruptive tested perturbation by flip rate: '+extra['most_disruptive'].replace('_',' ')+'.')
            st.write(extra['sanity_conclusion'])
            st.caption('Exploratory Validation-only digital stresses and matched occlusion; not external validation. Occlusion changes the input distribution. No tuning followed these results.')
        except Exception:
            st.warning('Additional Validation evidence is unavailable or inconsistent; no substitute results are generated.')


def render_result(result,upload,details):
    st.subheader('3 · Model result')
    score,threshold=st.columns(2)
    score.metric('Model output: malignant-pattern score',f"{result['score']:.4f}",help='Locked model output; not a calibrated clinical probability of cancer.')
    threshold.metric('Research decision threshold',f"{details['threshold']:.4f}",help='Fixed Validation-selected operating point. Decisions use full precision, not the rounded display.')
    st.info(result['decision'])
    if near_threshold(result['score']):
        st.warning('Near research decision threshold')
        st.caption('This model output lies close to the fixed research threshold and should be interpreted cautiously. The ±0.05 margin is a presentation-only caution, not a confidence interval; it does not change the decision.')
    st.caption(SCORE_NOTE)
    st.subheader('4 · Grad-CAM model-attention map')
    mode=st.radio('Attention view',('Original','Heatmap','Overlay'),index=2,horizontal=True,key='attention_mode',
        help='Grad-CAM shows model attention from output gradients, not a lesion boundary. Original means the uncoloured canonical preview.')
    opacity=st.slider('Overlay opacity',0.0,1.0,.35,.05,key='attention_opacity',disabled=mode!='Overlay',
        help='Maximum colour opacity only. No change to the heatmap values or research threshold.')
    st.caption('Display controls do not alter the model prediction or Grad-CAM computation.')
    cols=st.columns(2)
    cols[0].image(upload.canonical,caption='Canonical full mammogram',clamp=True,width='stretch')
    captions={'Original':'Original (uncoloured canonical preview)','Heatmap':'Grad-CAM heatmap · 0 (dark) to 1 (bright)','Overlay':'Model-attention overlay · display-only opacity'}
    cols[1].image(attention_view(upload.canonical,result,mode,opacity),caption=captions[mode],clamp=True,width='stretch')
    st.caption('Case exports and comparison previews retain the original overlay rendering.')
    if result['zero_heatmap']: st.warning('The positive Grad-CAM map is zero for this input. No positive attention region is displayed; this does not establish absence of an abnormality.')
    st.caption(ATTENTION_NOTE)
    st.subheader('5 · Interpretation limitations')
    st.write(DOMAIN_NOTE)
    st.write('The project measured low Grad-CAM localisation overlap. A high or low model score cannot establish or exclude disease.')
    st.write(DISCLAIMER)
    st.subheader('6 · Export')
    st.caption('Anonymous research summaries. Check the image itself for burned-in identifying text before downloading or sharing. Exports do not include filenames, patient tags or DICOM identifiers.')
    try:
        if 'exports' not in st.session_state:
            args=(result,upload.canonical,st.session_state['anonymous_reference'],st.session_state['timestamp'],upload.warnings)
            st.session_state['exports']=(png_report(*args),pdf_report(*args))
        png,pdf=st.session_state['exports']
        a,b=st.columns(2)
        a.download_button('Download PNG research summary',png,'mammoassist_research.png','image/png')
        b.download_button('Download PDF research summary',pdf,'mammoassist_research.pdf','application/pdf')
    except Exception:
        st.error('The research export could not be generated. The displayed result remains available; ask the project maintainer to check export dependencies.')


def main():
    st.set_page_config(page_title='MammoAssist AI · Research prototype',page_icon='◉',layout='wide')
    st.title(TITLE)
    st.caption('Full-mammogram research workflow · ResNet50 · Genuine gradient-derived Grad-CAM')
    st.caption('1 Upload → 2 Review quality → 3 Analyse → 4 Inspect attention → 5 Export / Compare')
    st.warning(DISCLAIMER)
    st.write(SCORE_NOTE)
    with st.sidebar:
        st.subheader('Research session')
        st.write('Upload → quality review → model score → attention → export')
        st.caption('Use de-identified full mammograms only. Uploads and results stay in active-session memory; no upload is written to disk or shared through a data cache.')
        st.caption(SESSION_NOTE)
        st.button('Clear uploaded image and results',on_click=clear_upload,width='stretch')
        st.caption('Use Streamlit Settings to choose light or dark appearance. This local prototype has no production authentication.')
    try: details=contract()
    except AppError as exc:
        st.error(str(exc));return
    with st.expander('About this research model'):
        st.write(DOMAIN_NOTE)
        st.write('Frozen classifier and attention-map study results are available in the Research evidence dashboard. No evaluation runs at app startup.')
    with st.expander('Technical details'):
        st.json({k:v for k,v in details.items() if k!='config'})
        st.caption('The 0.5 heatmap threshold belongs to the completed localisation study. The app displays the continuous attention map without lesion detection or threshold optimisation.')
    render_dashboard()
    render_session()
    st.subheader('1 · Input image')
    st.caption('DICOM (.dcm/.dicom), PNG or JPEG · one full mammogram · up to 128 MiB / 32 megapixels. Obvious ROI/cropped files are rejected; unmarked crops cannot be reliably identified automatically.')
    generation=st.session_state.get('upload_generation',0)
    file=st.file_uploader('Choose a de-identified full mammogram',type=['dcm','dicom','png','jpg','jpeg'],key='upload_'+str(generation))
    if file is None:
        for key in ('prepared','result','upload_digest','anonymous_reference','timestamp','exports','attention_mode','attention_opacity'):
            st.session_state.pop(key,None)
        st.info('Choose an image to start. No model inference runs until you confirm the input and select Analyse.')
        return
    data=file.getvalue(); digest=hashlib.sha256(data+file.name.encode()).hexdigest()
    if digest!=st.session_state.get('upload_digest'):
        for key in ('prepared','result','exports','timestamp','confirmed','attention_mode','attention_opacity'):
            st.session_state.pop(key,None)
        st.session_state['upload_digest']=digest
        st.session_state['anonymous_reference']='upload-'+uuid.uuid4().hex[:12]
        try: st.session_state['prepared']=prepare_upload(data,file.name,details['config'])
        except AppError as exc: st.error(str(exc))
    if 'prepared' not in st.session_state:
        st.error('No valid full-mammogram input is ready. Replace the upload to continue.');return
    upload=st.session_state['prepared']
    a,b=st.columns(2)
    a.image(upload.original,caption='Uploaded image · decoded grayscale, linear display normalisation',clamp=True,width='stretch')
    b.image(upload.canonical,caption='Locked canonical input · 224 × 224 · no CLAHE',clamp=True,width='stretch')
    with st.expander('Non-identifying input metadata'): st.json(upload.technical)
    st.subheader('2 · Input-quality checks')
    st.caption(upload.quality['interpretation'])
    a,b,c=st.columns(3)
    a.metric('Source resolution',f"{upload.quality['width']} × {upload.quality['height']}")
    b.metric('Normalised brightness',f"{upload.quality['normalised_mean']:.3f}")
    c.metric('Normalised contrast (SD)',f"{upload.quality['normalised_std']:.3f}")
    for warning in upload.warnings: st.warning(warning)
    if not upload.warnings: st.info('No heuristic warning. This does not certify clinical image quality or suitability.')
    confirmed=st.checkbox('I confirm this is a de-identified full mammogram, not a cropped lesion, ROI mask or screenshot, and have reviewed the input-quality warnings.',key='confirmed')
    if st.button('Analyse full mammogram',type='primary',disabled=not confirmed):
        st.session_state.pop('result',None);st.session_state.pop('exports',None)
        try:
            with st.spinner('Computing the locked model score and genuine Grad-CAM…'):
                model=model_resource(details['checkpoint_sha256'],(PROJECT_ROOT/CHECKPOINT).stat().st_mtime_ns)
                st.session_state['result']=analyse(upload.canonical,model)
                st.session_state['timestamp']=datetime.now(timezone.utc).isoformat(timespec='seconds')
                add_case(st.session_state,upload,st.session_state['result'],st.session_state['timestamp'])
        except AppError as exc: st.error(str(exc))
        except Exception: st.error('The model could not be used. Check the local checkpoint and project dependencies; no result was generated.')
        else: st.rerun()  # Refresh history with the already-computed result; never re-infer.
    if 'result' in st.session_state: render_result(st.session_state['result'],upload,details)


if __name__=='__main__': main()
