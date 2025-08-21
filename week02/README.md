# Week 2 — RAG Eval & Robustness (Aug 18–24, JST)

This folder is a *starter scaffold* for your Week 2 plan. Each day ships a small, testable artifact.
Use `CHANGELOG.md` for daily notes and `EVAL_SUMMARY.md` for ablations.

## Structure
- `day08` — Cross-encoder reranker + ablation
- `day09` — Hybrid retrieval (BM25 + dense)
- `day10` — Prompt robustness harness
- `day11` — Context optimization (chunking, k)
- `day12` — Faithfulness + attributions
- `day13` — CI-lite (unit tests, latency guard, smoke eval)
- `day14` — Weekly revision + mock interview + demo polish
- `utils`  — Common helpers
- `week02_demo` — Gradio demo shell

## Quick start
1. Edit `config.yaml` (index paths, model names, weights).
2. Run a day's script, e.g. `python day09/hybrid_search.py --weights 0.6 0.4` (placeholders).
3. Record metrics in CSVs and summarize in `EVAL_SUMMARY.md`.
4. Update `CHANGELOG.md` with what changed, why, and result.
