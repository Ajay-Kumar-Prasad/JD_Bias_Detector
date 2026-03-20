"""
sidebar.py
Renders the Streamlit sidebar: about section, category legend, settings.
"""
import streamlit as st
try:
    from app.config import CATEGORY_COLORS, CATEGORY_LABELS
except ModuleNotFoundError:
    from config import CATEGORY_COLORS, CATEGORY_LABELS


def render_sidebar() -> dict:
    """
    Renders the sidebar and returns a settings dict.

    Returns:
        {
            "confidence_threshold": float,   # 0.0 – 1.0
            "show_diff": bool,
            "show_raw_json": bool,
        }
    """
    with st.sidebar:
        st.markdown("## 🔍 JD Bias Detector")
        st.caption(
            "Detects gender-coded, ageist, exclusionary, and "
            "ability-coded language in job descriptions."
        )
        st.divider()

        st.markdown("#### Settings")
        confidence_threshold = st.slider(
            "Min confidence to show flag",
            min_value=0.5, max_value=1.0, value=0.75, step=0.05,
            help="Spans below this confidence score are hidden."
        )
        auto_rewrite_threshold = st.slider(
            "Auto-rewrite threshold",
            min_value=0.5, max_value=1.0, value=0.85, step=0.05,
            help="Spans at or above this confidence are auto-replaced.",
        )
        suggestion_threshold = st.slider(
            "Suggestion threshold",
            min_value=0.3, max_value=0.9, value=0.70, step=0.05,
            help="Spans between suggestion and auto thresholds are suggestion-only.",
        )
        if suggestion_threshold > auto_rewrite_threshold:
            suggestion_threshold = auto_rewrite_threshold
            st.caption("Suggestion threshold capped to auto-rewrite threshold.")
        show_diff = st.toggle("Show side-by-side diff", value=True)
        show_raw_json = st.toggle("Show raw API response", value=False)

        st.divider()
        st.markdown("#### Bias categories")
        for cat, color in CATEGORY_COLORS.items():
            label = CATEGORY_LABELS[cat]
            st.markdown(
                f'<span style="display:inline-flex;align-items:center;gap:8px;'
                f'margin-bottom:6px">'
                f'<span style="width:14px;height:14px;border-radius:3px;'
                f'background:{color};display:inline-block;flex-shrink:0"></span>'
                f'<span style="font-size:13px">{label}</span></span>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown(
            "<div style='font-size:12px;color:gray'>"
            "Model: DeBERTa-v3-base · Fine-tuned on Gaucher et al. + synthetic JDs"
            "</div>",
            unsafe_allow_html=True,
        )

    return {
        "confidence_threshold": confidence_threshold,
        "auto_rewrite_threshold": auto_rewrite_threshold,
        "suggestion_threshold": suggestion_threshold,
        "show_diff": show_diff,
        "show_raw_json": show_raw_json,
    }
