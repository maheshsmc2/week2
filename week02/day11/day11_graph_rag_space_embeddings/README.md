---
title: Day 11 Graph-RAG Demo (Embeddings)
emoji: 🕸️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.31.0
app_file: app.py
pinned: false
---

# Day 11 — Mini Graph‑RAG (with Embeddings)

This is the upgraded version of the Day‑11 demo: now it uses **sentence‑transformers** to semantically match questions to graph facts, so it works even when keywords don't exactly match (e.g., *"Who brought in the ACA?"*).

## ✨ What it shows
- Facts stored as a **graph** (NetworkX).
- Rule‑based graph traversal for common patterns.
- **Embeddings fallback** (all‑MiniLM‑L6‑v2) for robust matching.
- Gradio UI with **Answer**, **Reasoning Trace**, and **Top semantic matches**.

## 🗂️ Files
- `app.py` — Gradio UI.
- `graph_rag.py` — Graph‑RAG logic with embeddings.
- `requirements.txt` — minimal deps including `sentence-transformers`.
- `docs/` — add a screenshot of your running Space (optional).

## ▶️ Run locally
```bash
pip install -r requirements.txt
python app.py
```

## 🚀 Deploy on Hugging Face Spaces
- Create a **Gradio** Space and upload these files (or the ZIP).
- Hardware: **CPU Basic** is fine. First build will download the embedding model.

## 🔧 Notes
- The embedding model is *all‑MiniLM‑L6‑v2* (fast, ~22M params).
- For larger graphs, consider a real graph DB (Neo4j) and offline precomputed embeddings.
