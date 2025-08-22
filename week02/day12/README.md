---
title: Day 12 — RAG Index Upgrade (FAISS + HNSW)
emoji: 🧭
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: "1.36.0"
app_file: app.py
pinned: false
license: mit
---

# RAG Index Upgrade — FAISS + HNSW (Streamlit)

This Space demonstrates upgrading a basic RAG retriever from brute‑force cosine to **FAISS** (ANN) and **HNSW** (via Chroma).  
Use the sidebar to **build indexes** from the sample corpus or upload your own text, then **query** either backend and compare speed.

**Tip:** For larger corpora, FAISS/HNSW provide major speedups over brute force with minimal accuracy trade‑off.
