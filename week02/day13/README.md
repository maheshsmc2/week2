---
title: D13 FAISS IVF+PQ (Gradio)
emoji: 🔎
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# D13 — FAISS IVF+PQ vs Flat (Gradio)

This Space provides an interactive **Gradio** UI to benchmark FAISS **IVF+PQ** against a **Flat** (brute force) baseline on synthetic vectors.

## How it works
- Choose dataset size (N), queries (Nq), dimension (d), and metric (cosine/L2).
- Configure IVF (`nlist`, `nprobe`) and PQ (`m`, `nbits`) parameters.
- Click **Run Benchmark** → See latency and **Recall@K** vs Flat.

> If you prefer CLI, keep `faiss_ivfpq_demo.py` in the repo and run it locally/Colab.
