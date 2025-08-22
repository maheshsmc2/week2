
# Day 11 — Mini Graph‑RAG (Gradio + NetworkX)

A tiny, intern‑friendly **Graph‑RAG** demo you can deploy on **Hugging Face Spaces**.

## ✨ What it shows
- How to store facts as a **graph** (nodes & edges).
- How to answer questions via **relation lookup** + simple **multi‑hop traversal**.
- A lightweight UI with **Gradio**.

## 🗂️ Files
- `app.py` — Gradio UI.
- `graph_rag.py` — Graph-RAG logic with a small demo knowledge graph.
- `requirements.txt` — minimal deps.
- `docs/` — put a screenshot of the running app here (optional but recommended).

## ▶️ Run locally
```bash
pip install -r requirements.txt
python app.py
```

Then open the local URL shown by Gradio.

## 🚀 Deploy on Hugging Face Spaces
1. Go to **Hugging Face → New Space**.
2. **Space SDK:** Gradio. **Hardware:** CPU basic is enough.
3. Upload these files (or drag‑drop the ZIP below).
4. Wait for the build; then test with:
   - *Who introduced Obamacare?*
   - *Who was president when Obamacare was introduced?*

## 🧠 How it works (short)
- We create a directed graph of triples `(subject, relation, object)` using NetworkX.
- For “who introduced X”, we find edges with relation `introduced` that match `X` (with simple aliases).
- For “who was president when X was introduced”, we do **two hops**:
  1. `introduced(X)` → find the person.
  2. Check if that person has an edge `was_president_of` → return them.

> This is intentionally **minimal**. In a real app, replace the naive keyword checks with embeddings and add a proper graph DB (e.g., Neo4j).

## 📸 Screenshot
Add a screenshot of your Space UI to `docs/` and commit, so readers see it in the repo.
