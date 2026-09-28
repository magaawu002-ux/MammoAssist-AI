"""Render existing attention values only; never call a model or change results."""
from src.explainability.overlay import overlay_heatmap


def attention_view(canonical,result,mode,opacity):
    if mode=='Original': return canonical.copy()
    if mode=='Heatmap': return result['heatmap'].copy()
    if mode=='Overlay': return overlay_heatmap(canonical,result['heatmap'],alpha=opacity)
    raise ValueError('Unknown attention display mode')
