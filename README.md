# RetailMind

Multi-constraint product search. A natural-language query such as "black running shoes for men
under $60, delivered in 3 days" goes through query understanding, hybrid retrieval (BM25 and
dense), knowledge-graph constraint filtering, cross-encoder re-ranking, and an agentic
self-check loop. Results are evaluated against a BM25 baseline on Amazon ESCI.

## Status

| Week | Stage | Status |
| --- | --- | --- |
| 1 | Repo setup, ESCI download, data preparation | In progress |
| 2 | BM25 baseline and evaluation harness | Not started |
| 3 | Dense retrieval and RRF fusion | Not started |
| 4 | Cross-encoder re-ranking | Not started |
| 5 | Query parser, decomposition, attribute extraction | Not started |
| 6 | KG constraint filter, synthetic constraints (SC-100) | Not started |
| 7 | Agent loop, grounded generation, Streamlit demo | Not started |
| 8 | Final evaluation, RAGAS, report | Not started |

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

Heavier packages are optional extras, installed when their week starts, for example
`pip install -e ".[retrieval]"` in Week 2.

## Layout

```text
configs/     default.yaml (paths, models, thresholds), ablations.yaml (components per run A0-A6)
data/        raw, processed, indexes (git-ignored; rebuilt by scripts)
scripts/     data preparation, index building, evaluation runs
src/retailmind/
  retrieval/ query/ kg/ rerank/ agent/ generate/ eval/
tests/       one test file per module
results/     committed run folders (config, run.json, metrics.json)
docs/        report and figures
app/         Streamlit demo
```

## Web UI

```powershell
pip install -e ".[demo]"
python app/server.py        # http://127.0.0.1:8000
```

With the ESCI data built (`data/processed/`), the UI runs the real pipeline for any ablation run
A0-A6 (LLM stages need `LLM_*` in `.env`). Without it, it falls back to a demo mode that runs the
real parser, BM25, knowledge graph and filter on a 60-product sample catalogue. Set
`RETAILMIND_MODE=demo` to force that. The Evaluation tab reads `results/`.

## Git workflow

- `main` always runs. Work happens on `feat/<stage>-<topic>` branches, merged by pull request
  (squash).
- Commit messages follow Conventional Commits, for example `feat(retrieval): add RRF fusion`.
- Milestones are tagged: `v0.1-baseline`, `v0.2-hybrid`, `v0.3-rerank`, `v0.4-constraints`,
  `v0.5-agent`, `v1.0-final`.
- Data, indexes, embeddings, and `.env` are never committed.
