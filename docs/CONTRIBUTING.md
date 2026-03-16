# Contributing

Thanks for your interest in improving JD Bias Detector!

## Ways to contribute

- **Expand the bias lexicon** — Add new terms to `data/bias_lexicon/` with citations.
- **Improve annotations** — Fix labeling errors in `data/annotated/` via PR with explanation.
- **New bias categories** — Open an issue first to discuss scope (e.g. `RACIAL_CODED`).
- **Model improvements** — Experiment with different base models or training strategies.
- **UI/UX** — Improve the Streamlit app or add new visualizations.
- **Bug fixes** — Check the issues tab.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/jd-bias-detector.git
cd jd-bias-detector
pip install -r requirements-dev.txt
```

## Running tests

```bash
pytest tests/ -v
```

All tests must pass before opening a PR.

## Code style

```bash
black .       # auto-format
ruff check .  # lint
```

## PR guidelines

- Keep PRs focused — one concern per PR.
- Include a brief description of what changed and why.
- If adding new bias terms, include at least one academic or HR guideline reference.
- If changing the model or training pipeline, include before/after eval metrics.

## Lexicon contributions

New terms added to `data/bias_lexicon/` must include:

1. The term or phrase
2. Which group(s) it may exclude
3. A suggested neutral alternative
4. A citation (research paper, EEOC guideline, or reputable HR source)

## Code of conduct

Be kind. This project is about making hiring more equitable —
that spirit should extend to how contributors treat each other.
