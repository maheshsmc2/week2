# How to deploy on Hugging Face Spaces

1) Create a new **Space** → type: *Streamlit* (or just default) → name it `d12-rag-index-upgrade`.
2) Upload these files to the root of the Space:
   - `README.md` (with the YAML front matter at the top)
   - `requirements.txt`
   - `app.py`
   - `data/corpus.txt`
3) The Space will auto‑build. Open the **App** tab to use.
4) Use the sidebar to **build FAISS** and **HNSW/Chroma**, then ask questions and see timings.
