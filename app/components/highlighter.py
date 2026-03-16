"""
highlighter.py
Renders an HTML string with biased spans highlighted inline.
Each span gets a colored background badge matching its bias category.
"""
import html
from app.config import CATEGORY_COLORS, CATEGORY_LABELS


def build_highlighted_html(original_text: str, flagged_spans: list) -> str:
    """
    Returns an HTML string with flagged spans wrapped in colored <mark> tags.

    Args:
        original_text:  The raw job description string.
        flagged_spans:  List of dicts with keys: start, end, category, confidence, text.

    Returns:
        Safe HTML string ready for st.markdown(..., unsafe_allow_html=True).
    """
    if not flagged_spans:
        return f"<p style='line-height:1.8'>{html.escape(original_text)}</p>"

    # Sort spans by start position; handle overlaps by skipping nested ones
    spans = sorted(flagged_spans, key=lambda s: s["start"])
    result = []
    cursor = 0

    for span in spans:
        start = span["start"]
        end   = span["end"]
        cat   = span["category"]
        conf  = span["confidence"]

        if start < cursor:          # overlapping span — skip
            continue

        # Neutral text before this span
        if start > cursor:
            result.append(html.escape(original_text[cursor:start]))

        color     = CATEGORY_COLORS.get(cat, "#cccccc")
        label     = CATEGORY_LABELS.get(cat, cat)
        span_text = html.escape(original_text[start:end])
        tooltip   = f"{label} ({conf:.0%} confidence)"

        result.append(
            f'<mark title="{tooltip}" style="'
            f'background:{color}22;'
            f'border-bottom:2px solid {color};'
            f'border-radius:3px;'
            f'padding:1px 3px;'
            f'cursor:help;'
            f'font-weight:500'
            f'">{span_text}'
            f'<sup style="font-size:10px;color:{color};margin-left:2px">'
            f'{label}</sup></mark>'
        )
        cursor = end

    # Remaining text after last span
    if cursor < len(original_text):
        result.append(html.escape(original_text[cursor:]))

    body = "".join(result)
    return f"<p style='line-height:1.9;font-size:15px'>{body}</p>"


def render_legend(categories_present: list) -> str:
    """Returns an HTML legend row for the categories found in this JD."""
    items = []
    for cat in categories_present:
        color = CATEGORY_COLORS.get(cat, "#ccc")
        label = CATEGORY_LABELS.get(cat, cat)
        items.append(
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'margin-right:14px;font-size:12px">'
            f'<span style="width:12px;height:12px;border-radius:2px;'
            f'background:{color};display:inline-block"></span>'
            f'{label}</span>'
        )
    return "<div style='margin-bottom:10px'>" + "".join(items) + "</div>"
