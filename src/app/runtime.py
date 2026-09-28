"""Cache only the immutable model resource, never user images or results."""

import streamlit as st

from src.app.core import CHECKPOINT_HASH,AppError,load_model


@st.cache_resource(show_spinner=False)
def model_resource(checkpoint_hash,checkpoint_mtime_ns):
    if checkpoint_hash!=CHECKPOINT_HASH: raise AppError('Unexpected checkpoint request.')
    return load_model()
