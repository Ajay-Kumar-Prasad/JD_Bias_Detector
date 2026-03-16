"""
diff_view.py
Side-by-side word-level diff between original and rewritten JD.
Deletions shown in red, insertions in green — standard diff UX.
"""
import html
import difflib


def _tokenize(text: str) -> list[str]:
    """Split into words + whitespace tokens to enable word-level diff."""
    import re
    return re.findall(r'\S+|\s+', text)


def build_diff_html(original: str, rewritten: str) -> tuple[str, str]:
    """
    Returns (original_html, rewritten_html) with changed words highlighted.
    Deletions are marked in the original; insertions in the rewritten.
    """
    orig_tokens    = _tokenize(original)
    rewrite_tokens = _tokenize(rewritten)

    matcher = difflib.SequenceMatcher(None, orig_tokens, rewrite_tokens, autojunk=False)
    orig_html    = []
    rewrite_html = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        orig_chunk    = "".join(orig_tokens[i1:i2])
        rewrite_chunk = "".join(rewrite_tokens[j1:j2])

        if tag == "equal":
            orig_html.append(html.escape(orig_chunk))
            rewrite_html.append(html.escape(rewrite_chunk))

        elif tag == "replace":
            orig_html.append(
                f'<mark style="background:#ffd7d7;border-radius:3px;padding:1px 2px;'
                f'text-decoration:line-through;color:#a32d2d">'
                f'{html.escape(orig_chunk)}</mark>'
            )
            rewrite_html.append(
                f'<mark style="background:#d4f0d4;border-radius:3px;padding:1px 2px;'
                f'color:#27500a;font-weight:500">'
                f'{html.escape(rewrite_chunk)}</mark>'
            )

        elif tag == "delete":
            orig_html.append(
                f'<mark style="background:#ffd7d7;border-radius:3px;padding:1px 2px;'
                f'text-decoration:line-through;color:#a32d2d">'
                f'{html.escape(orig_chunk)}</mark>'
            )

        elif tag == "insert":
            rewrite_html.append(
                f'<mark style="background:#d4f0d4;border-radius:3px;padding:1px 2px;'
                f'color:#27500a;font-weight:500">'
                f'{html.escape(rewrite_chunk)}</mark>'
            )

    style = "line-height:1.9;font-size:14px"
    return (
        f"<p style='{style}'>{''.join(orig_html)}</p>",
        f"<p style='{style}'>{''.join(rewrite_html)}</p>",
    )


def change_summary(original: str, rewritten: str) -> dict:
    """Returns counts of words added, removed, and changed."""
    orig_tokens    = _tokenize(original)
    rewrite_tokens = _tokenize(rewritten)
    matcher = difflib.SequenceMatcher(None, orig_tokens, rewrite_tokens, autojunk=False)

    added = replaced = removed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            added += len(rewrite_tokens[j1:j2])
        elif tag == "delete":
            removed += len(orig_tokens[i1:i2])
        elif tag == "replace":
            replaced += 1
    return {"replaced": replaced, "added": added, "removed": removed}
