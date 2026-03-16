"""
streamlit_app.py — JD Bias Detector UI

Run:
    streamlit run app/streamlit_app.py
"""
import json
import httpx
import streamlit as st

from app.config import API_URL, APP_TITLE, APP_ICON, CATEGORY_COLORS, CATEGORY_LABELS
from app.components.sidebar import render_sidebar
from app.components.highlighter import build_highlighted_html, render_legend
from app.components.diff_view import build_diff_html, change_summary

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
with open("app/assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
settings = render_sidebar()
conf_threshold = settings["confidence_threshold"]
show_diff      = settings["show_diff"]
show_raw_json  = settings["show_raw_json"]


# ── Header ────────────────────────────────────────────────────
st.markdown(f"# {APP_ICON} {APP_TITLE}")
st.caption(
    "Paste a job description below. The model flags biased language, "
    "explains each issue, and rewrites the JD to be more inclusive."
)
st.divider()


# ── Sample JDs for quick testing ─────────────────────────────
SAMPLES = {
    "High bias example": (
        "We are looking for a rockstar engineer who is young and hungry to crush it "
        "in our fast-paced, high-pressure environment. The ideal candidate is aggressive, "
        "independent, and a digital native who can dominate the competition."
    ),
    "Low bias example": (
        "We are seeking a skilled software engineer to join our collaborative team. "
        "You will design scalable systems, communicate clearly with stakeholders, "
        "and contribute to a supportive and inclusive work environment."
    ),
}

col_input, col_sample = st.columns([4, 1])
with col_sample:
    sample_choice = st.selectbox("Load a sample", ["— none —"] + list(SAMPLES.keys()))

default_text = SAMPLES.get(sample_choice, "") if sample_choice != "— none —" else ""

with col_input:
    jd_text = st.text_area(
        "Job description",
        value=default_text,
        height=220,
        placeholder="Paste your job description here...",
        label_visibility="collapsed",
    )

analyze_btn = st.button("Analyze for bias", type="primary", use_container_width=False)
st.divider()


# ── Analysis ──────────────────────────────────────────────────
def call_api(text: str) -> dict:
    try:
        resp = httpx.post(
            f"{API_URL}/analyze/",
            json={"text": text},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error(
            "Cannot connect to the API. Make sure the FastAPI server is running:\n"
            "```\nuvicorn api.main:app --reload\n```"
        )
        st.stop()
    except Exception as e:
        st.error(f"API error: {e}")
        st.stop()


if analyze_btn and jd_text.strip():
    with st.spinner("Analyzing..."):
        result = call_api(jd_text)

    # Filter spans by confidence threshold
    all_spans    = result.get("flagged_spans", [])
    shown_spans  = [s for s in all_spans if s["confidence"] >= conf_threshold]
    hidden_count = len(all_spans) - len(shown_spans)
    score        = result.get("inclusivity_score", 0)
    breakdown    = result.get("category_breakdown", {})
    rewritten    = result.get("rewritten_text", "")

    # ── Score + breakdown ──────────────────────────────────────
    ring_class = (
        "score-high"   if score >= 70 else
        "score-medium" if score >= 40 else
        "score-low"
    )
    c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])
    with c1:
        st.markdown(
            f'<div class="score-ring {ring_class}">{score}</div>'
            f'<p style="text-align:center;font-size:12px;color:gray">Inclusivity score</p>',
            unsafe_allow_html=True,
        )
    for col, cat in zip([c2, c3, c4, c5], ["GENDER_CODED", "AGEIST", "EXCLUSIONARY", "ABILITY_CODED"]):
        with col:
            color = CATEGORY_COLORS[cat]
            count = breakdown.get(cat, 0)
            st.markdown(
                f'<div style="border:1px solid {color};border-radius:10px;'
                f'padding:10px;text-align:center;background:{color}11">'
                f'<div style="font-size:24px;font-weight:700;color:{color}">{count}</div>'
                f'<div style="font-size:11px;color:gray;margin-top:2px">{CATEGORY_LABELS[cat]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if hidden_count:
        st.caption(f"ℹ️ {hidden_count} low-confidence flag(s) hidden (threshold: {conf_threshold:.0%}). Adjust in sidebar.")

    st.divider()

    # ── Highlighted original ───────────────────────────────────
    st.markdown("#### Flagged phrases")
    cats_present = [c for c in ["GENDER_CODED", "AGEIST", "EXCLUSIONARY", "ABILITY_CODED"]
                    if breakdown.get(c, 0) > 0]
    if cats_present:
        st.markdown(render_legend(cats_present), unsafe_allow_html=True)

    if shown_spans:
        st.markdown(
            build_highlighted_html(jd_text, shown_spans),
            unsafe_allow_html=True,
        )
    else:
        st.success("No bias flags above the confidence threshold. Looks good!")

    st.divider()

    # ── Flag detail cards ──────────────────────────────────────
    if shown_spans:
        st.markdown("#### What to fix")
        for span in shown_spans:
            cat   = span["category"]
            color = CATEGORY_COLORS.get(cat, "#ccc")
            label = CATEGORY_LABELS.get(cat, cat)
            st.markdown(
                f'<div class="flag-card">'
                f'<div class="flag-phrase" style="color:{color}">"{span["text"]}"</div>'
                f'<div class="flag-meta">{label} · {span["confidence"]:.0%} confidence</div>'
                f'<div class="flag-explanation">{span["explanation"]}</div>'
                f'<div>Rewrite as: <span class="flag-rewrite">{span["rewrite"]}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()

    # ── Diff view ─────────────────────────────────────────────
    if show_diff and rewritten:
        st.markdown("#### Side-by-side comparison")
        summary = change_summary(jd_text, rewritten)
        st.caption(
            f"{summary['replaced']} phrase(s) rewritten · "
            f"{summary['added']} word(s) added · "
            f"{summary['removed']} word(s) removed"
        )
        orig_html, rewrite_html = build_diff_html(jd_text, rewritten)
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(
                f'<div class="diff-col"><h4>Original</h4>{orig_html}</div>',
                unsafe_allow_html=True,
            )
        with d2:
            st.markdown(
                f'<div class="diff-col"><h4>Rewritten</h4>{rewrite_html}</div>',
                unsafe_allow_html=True,
            )

        st.download_button(
            "⬇ Download rewritten JD",
            data=rewritten,
            file_name="rewritten_jd.txt",
            mime="text/plain",
        )

    # ── Raw JSON ──────────────────────────────────────────────
    if show_raw_json:
        st.divider()
        with st.expander("Raw API response"):
            st.json(result)

elif analyze_btn and not jd_text.strip():
    st.warning("Please paste a job description before analyzing.")
