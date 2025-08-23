---
title: D13 FAISS — IVF+PQ & HNSW (Gradio, CSV + Plot)
emoji: 📈
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# D13 — FAISS: IVF+PQ & HNSW vs Flat (with CSV + Plot)

Interactive **Gradio Space** to compare **IVF+PQ** and **HNSW** against a **Flat** baseline.

**What you get**
- UI sliders for dataset/index params
- Latency & **Recall@K** for each method
- **Downloadable CSV** of results
- **Matplotlib plot** (Recall vs Throughput)

**Requirements**: `faiss-cpu`, `numpy`, `pandas`, `matplotlib`, `gradio`.
