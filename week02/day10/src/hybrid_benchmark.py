
import argparse, os, time, random, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss

try:
    from sentence_transformers import CrossEncoder
    HAS_CE = True
except Exception:
    HAS_CE = False

def load_corpus(path):
    if path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        if "text" in df.columns:
            return df["text"].dropna().tolist()
        else:
            return df.iloc[:,0].dropna().astype(str).tolist()
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def tokenize_for_bm25(texts):
    return [t.lower().split() for t in texts]

def build_bm25(docs):
    return BM25Okapi(tokenize_for_bm25(docs))

def recall_at_k(true_ids, retrieved_ids, k=5):
    hits = 0
    for t, r in zip(true_ids, retrieved_ids):
        hits += int(t in r[:k])
    return hits / len(true_ids)

def build_flat(xb):
    d = xb.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(xb.astype(np.float32))
    return idx

def build_ivfpq(xb, nlist=64, m=16, nbits=8):
    d = xb.shape[1]
    quant = faiss.IndexFlatIP(d)
    idx = faiss.IndexIVFPQ(quant, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    idx.train(xb.astype(np.float32))
    idx.add(xb.astype(np.float32))
    idx.nprobe = max(4, nlist // 8)
    return idx

def build_hnsw(xb, M=32, efConstruction=200, efSearch=64):
    d = xb.shape[1]
    idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = efConstruction
    idx.hnsw.efSearch = efSearch
    idx.add(xb.astype(np.float32))
    return idx

def search(idx, xq, k):
    D, I = idx.search(xq.astype(np.float32), k)
    return D, I

def run(args):
    docs = load_corpus(args.corpus)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    xb = model.encode(docs, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    flat = build_flat(xb)
    q_ids = random.sample(range(len(docs)), min(args.queries, len(docs)))
    queries = [docs[i] for i in q_ids]
    xq = model.encode(queries, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    Dg, Ig = search(flat, xq, args.k)
    gold_top1 = [row[0] for row in Ig]

    ivfpq = build_ivfpq(xb, args.nlist, args.pq_m, 8)
    hnsw = build_hnsw(xb, args.hnsw_m, 200, args.hnsw_ef)
    bm25 = build_bm25(docs)

    rows = []
    def bench(label, idx):
        t0 = time.time()
        D, I = search(idx, xq, args.k)
        t = (time.time() - t0) / len(queries)
        r = recall_at_k(gold_top1, I, args.k)
        rows.append({"method": label, "avg_latency_s": t, "recall_vs_gold@k": r})
        return D, I

    bench("Flat (gold)", flat)
    D_ivf, I_ivf = bench("IVF-PQ", ivfpq)
    D_hnsw, I_hnsw = bench("HNSW", hnsw)

    # Hybrid
    def hybrid(I_vec, D_vec, alpha=0.6):
        hybrid_I = []
        for qi in range(len(queries)):
            vec_ids = I_vec[qi]
            vec_scores = D_vec[qi]
            bm_scores = bm25.get_scores(queries[qi].lower().split())
            bm_top = np.argsort(bm_scores)[::-1][:args.k]
            union_ids = set(vec_ids.tolist()) | set(bm_top.tolist())
            fused = []
            for did in union_ids:
                v_s = vec_scores[list(vec_ids).index(did)] if did in vec_ids else 0.0
                b_s = bm_scores[did]
                f = alpha * v_s + (1 - alpha) * b_s
                fused.append((did, f))
            fused.sort(key=lambda x: x[1], reverse=True)
            hybrid_I.append([d for d,_ in fused[:args.k]])
        return np.array(hybrid_I)

    I_hybrid_ivf = hybrid(I_ivf, D_ivf, args.alpha)
    I_hybrid_hnsw = hybrid(I_hnsw, D_hnsw, args.alpha)
    rows.append({"method": f"Hybrid IVF-PQ α={args.alpha}", "avg_latency_s": np.nan, "recall_vs_gold@k": recall_at_k(gold_top1, I_hybrid_ivf, args.k)})
    rows.append({"method": f"Hybrid HNSW α={args.alpha}", "avg_latency_s": np.nan, "recall_vs_gold@k": recall_at_k(gold_top1, I_hybrid_hnsw, args.k)})

    if args.use_reranker and HAS_CE:
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        def rerank(I_list):
            rer_I = []
            for qi, doc_ids in enumerate(I_list):
                pairs = [(queries[qi], docs[did]) for did in doc_ids]
                scores = ce.predict(pairs).tolist()
                order = np.argsort(scores)[::-1]
                rer_I.append([doc_ids[i] for i in order[:args.k]])
            return rer_I
        I_r_ivf = rerank(I_hybrid_ivf)
        I_r_hnsw = rerank(I_hybrid_hnsw)
        rows.append({"method": "Hybrid IVF-PQ + Reranker", "avg_latency_s": np.nan, "recall_vs_gold@k": recall_at_k(gold_top1, I_r_ivf, args.k)})
        rows.append({"method": "Hybrid HNSW + Reranker", "avg_latency_s": np.nan, "recall_vs_gold@k": recall_at_k(gold_top1, I_r_hnsw, args.k)})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out_csv = os.path.join(os.path.dirname(args.corpus), "..", "results_plus.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    # Plot
    fig, ax = plt.subplots()
    df_plot = df.dropna()
    ax.scatter(df_plot["avg_latency_s"], df_plot["recall_vs_gold@k"])
    for i, row in df_plot.iterrows():
        ax.annotate(row["method"], (row["avg_latency_s"], row["recall_vs_gold@k"]))
    ax.set_xlabel("Avg Latency (s)")
    ax.set_ylabel("Recall@K vs Gold")
    ax.set_title("Latency vs Recall@K")
    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(args.corpus), "..", "results_plot.png")
    plt.savefig(out_png)
    print(f"Saved plot to {out_png}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default=os.path.join("data", "sample_corpus.txt"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--nlist", type=int, default=64)
    ap.add_argument("--pq-m", type=int, default=16)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--hnsw-ef", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--use_reranker", action="store_true")
    args = ap.parse_args()
    run(args)
