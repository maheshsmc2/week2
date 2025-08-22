
import os, time, numpy as np, streamlit as st
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss

CORPUS_PATH = os.path.join("data", "sample_corpus.txt")
with open(CORPUS_PATH, "r") as f:
    DOCS = [ln.strip() for ln in f if ln.strip()]

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def emb(model, texts):
    return model.encode(texts, normalize_embeddings=True).astype(np.float32)

@st.cache_resource
def build_bm25():
    toks = [d.lower().split() for d in DOCS]
    return BM25Okapi(toks)

def build_flat(X):
    d = X.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(X)
    return idx

st.title("Hybrid Search Playground")
model = load_model()
X = emb(model, DOCS)
bm25 = build_bm25()
index = build_flat(X)

q = st.text_input("Query", "vector search ANN vs BM25")
k = st.slider("Top-k", 1, 10, 5)
alpha = st.slider("Hybrid weight α", 0.0, 1.0, 0.6, 0.05)

if st.button("Search"):
    xq = emb(model, [q])
    D, I = index.search(xq, k)
    ids = I[0].tolist()

    bm_scores = bm25.get_scores(q.lower().split())
    bm_top = np.argsort(bm_scores)[::-1][:k]
    union = set(ids) | set(bm_top.tolist())
    fused = []
    for did in union:
        v_s = D[0][ids.index(did)] if did in ids else 0.0
        b_s = bm_scores[did]
        fused.append((did, alpha*v_s + (1-alpha)*b_s))
    fused.sort(key=lambda x: x[1], reverse=True)
    final_ids = [i for i,_ in fused[:k]]

    for rank, did in enumerate(final_ids, 1):
        st.markdown(f"**{rank}.** {DOCS[did]}")
