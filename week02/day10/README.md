# Day 10 – Track 2 (PLUS): Hybrid Search + Re-ranking + Streamlit

This upgraded mini‑project adds:
- **Hybrid Search**: BM25 (keyword) + Vector (FAISS IVF‑PQ / HNSW / Flat)
- **Optional Re‑ranker**: Cross‑Encoder for better top‑k precision
- **Streamlit UI**: Try queries, tweak params, inspect results
- **Results plot**: Latency vs Recall@K graph auto‑generated

## Install
```bash
pip install -U -r requirements.txt
```

## CLI (benchmark & build indexes)
```bash
python src/hybrid_benchmark.py --k 5 --nlist 64 --pq-m 16 --hnsw-m 32 --hnsw-ef 64 --queries 10 --use_reranker
```

Outputs:
- `results_plus.csv`
- `results_plot.png` (latency vs recall)

## Streamlit App
```bash
streamlit run streamlit_app.py
```

## Dataset Loader
Replace `data/sample_corpus.txt` (1 doc per line) or point `--corpus` to your own `.txt`/`.csv`.

