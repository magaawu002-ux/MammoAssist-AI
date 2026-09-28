"""Presentation-only session helpers. No filesystem, model or persistence calls."""
from src.app.core import THRESHOLD,score_wording

# UI caution only. Never used to classify or modify scientific configuration.
NEAR_THRESHOLD_MARGIN=0.05
COMPARISON_NOTE='Score differences between images do not indicate disease progression, treatment response or a change in clinical diagnosis.'
SESSION_NOTE='Session history is temporary and is not saved. It is for workflow demonstration only, not an electronic medical record.'


def near_threshold(score):
    score_wording(score)  # Same finite/range validation; no alternative decision rule.
    return abs(score-THRESHOLD)<=NEAR_THRESHOLD_MARGIN


def add_case(state,upload,result,timestamp):
    """Keep allowlisted metadata plus the minimal previews needed for comparison."""
    if result['threshold']!=THRESHOLD: raise ValueError('Locked threshold mismatch')
    score=float(result['score']);decision=score_wording(score)
    number=state.get('case_sequence',0)+1
    record=dict(case_number=number,timestamp=timestamp,format=upload.technical['format'],
        width=int(upload.quality['width']),height=int(upload.quality['height']),score=score,
        threshold=THRESHOLD,decision=decision,near_threshold=near_threshold(score),quality_warning_count=len(upload.warnings),gradcam_available=True)
    # Full-resolution upload, raw DICOM, identifiers, filename and heatmap are not retained here.
    view=dict(canonical=upload.canonical.copy(),overlay=result['overlay'].copy(),
        warnings=list(upload.warnings),brightness=float(upload.quality['normalised_mean']),
        contrast=float(upload.quality['normalised_std']),zero_heatmap=bool(result['zero_heatmap']))
    state['case_history']=[*state.get('case_history',[]),record]
    state['case_views']={**state.get('case_views',{}),number:view}
    state['case_sequence']=number
    return record


def clear_history(state):
    for key in ('case_history','case_views','case_sequence','compare_left','compare_right'):
        state.pop(key,None)


def comparison(state,left,right):
    history=state.get('case_history',[])
    if len(history)<2: raise ValueError('Analyse at least two images in this session to enable comparison.')
    if left==right: raise ValueError('Choose two different session cases.')
    records={r['case_number']:r for r in history}
    if left not in records or right not in records: raise ValueError('Selected session case is unavailable.')
    return dict(records=(records[left],records[right]),views=(state['case_views'][left],state['case_views'][right]),
                absolute_score_difference=abs(records[left]['score']-records[right]['score']))
