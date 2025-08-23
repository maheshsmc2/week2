#!/usr/bin/env python3
# Day 13 — Track 2: FAISS IVF+PQ vs Flat (brute force)
# Intern-friendly, Colab-ready demo.
# Author: ChatGPT

import argparse
import time
import numpy as np

try:
    import faiss
except ImportError as e:
    raise SystemExit(
        "FAISS is not installed. Please run:\n"
        "  pip install faiss-cpu\n"
        "and try again."
    )

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / norms

def make_data(n: int, nq: int, d: int, metric: str, seed: int = 42):
    rng = np.random.default_rng(seed)
    xb = rng.normal(0, 1, size=(n, d)).astype('float32')
    xq = rng.normal(0, 1, size=(nq, d)).astype('float32')
    if metric == "cosine":
        xb = l2_normalize(xb)
        xq = l2_normalize(xq)
    return xb, xq

def build_flat(d: int, metric: str):
    if metric == "cosine":
        index = faiss.IndexFlatIP(d)  # cosine via inner product on unit vectors
    elif metric == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError("metric must be 'cosine' or 'l2'")
    return index

def build_ivfpq(d: int, nlist: int, m: int, nbits: int, metric: str):
    # quantizer for coarse clustering
    if metric == "cosine":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    else:  # l2
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_L2)
    return index

def recall_at_k(I_true: np.ndarray, I_test: np.ndarray, k: int) -> float:
    # Fraction of retrieved that are in the true top-k (intersection / k), averaged over queries
    # Ensure shapes: (nq, k)
    nq = I_true.shape[0]
    inter = 0
    for i in range(nq):
        inter += len(set(I_true[i, :k]).intersection(I_test[i, :k]))
    return inter / float(nq * k)

def time_search(index, queries: np.ndarray, k: int):
    t0 = time.perf_counter()
    D, I = index.search(queries, k)
    t1 = time.perf_counter()
    dt = t1 - t0
    nq = queries.shape[0]
    qps = nq / dt if dt > 0 else float('inf')
    ms_per_query = (dt / nq) * 1000 if nq > 0 else 0.0
    return D, I, dt, qps, ms_per_query

def fmt_time(dt: float) -> str:
    if dt < 1.0:
        return f"{dt*1000:.2f} ms"
    return f"{dt:.2f} s"

def main():
    p = argparse.ArgumentParser(description="FAISS IVF+PQ vs Flat (cosine or L2).")
    p.add_argument("--n", type=int, default=200_000, help="Number of database vectors")
    p.add_argument("--nq", type=int, default=1_000, help="Number of queries")
    p.add_argument("--d", type=int, default=128, help="Vector dimension")
    p.add_argument("--k", type=int, default=10, help="Top-k")
    p.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"], help="Distance metric")
    p.add_argument("--nlist", type=int, default=4096, help="IVF: number of clusters")
    p.add_argument("--m", type=int, default=16, help="PQ: number of subquantizers")
    p.add_argument("--nbits", type=int, default=8, help="PQ: bits per codebook index")
    p.add_argument("--nprobe", type=int, default=16, help="IVF: clusters to search per query")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    print("=== Settings ===")
    for k, v in vars(args).items():
        print(f"{k:>8}: {v}")
    print()

    # Data
    xb, xq = make_data(args.n, args.nq, args.d, args.metric, seed=args.seed)

    # 1) Flat (brute force) baseline
    flat = build_flat(args.d, args.metric)
    flat.add(xb)
    print("[Flat] Built. #vectors:", flat.ntotal)

    _, _, dt_add, qps_add, _ = time_search(flat, xq[:1], 1)  # warmup
    t0 = time.perf_counter()
    Df, If, dt_flat, qps_flat, ms_flat = time_search(flat, xq, args.k)
    print(f"[Flat] {args.nq} queries in {fmt_time(dt_flat)} → {qps_flat:.1f} qps ({ms_flat:.2f} ms/query)")
    print()

    # 2) IVF+PQ
    ivfpq = build_ivfpq(args.d, args.nlist, args.m, args.nbits, args.metric)
    # IVF requires training
    t_train0 = time.perf_counter()
    ivfpq.train(xb)
    t_train1 = time.perf_counter()
    ivfpq.add(xb)
    t_train2 = time.perf_counter()
    ivfpq.nprobe = args.nprobe
    print("[IVF+PQ] Trained in", fmt_time(t_train1 - t_train0), "; added in", fmt_time(t_train2 - t_train1))
    print("[IVF+PQ] nprobe =", ivfpq.nprobe, "| nlist =", args.nlist, "| m =", args.m, "| nbits =", args.nbits)

    # Warmup + search
    _ = ivfpq.search(xq[:1], args.k)
    Di, Ii, dt_ivf, qps_ivf, ms_ivf = time_search(ivfpq, xq, args.k)

    # Evaluate recall@k vs Flat (ground truth from Flat)
    r_at_k = recall_at_k(If, Ii, args.k)

    print(f"[IVF+PQ] {args.nq} queries in {fmt_time(dt_ivf)} → {qps_ivf:.1f} qps ({ms_ivf:.2f} ms/query); Recall@{args.k} = {r_at_k:.4f}")
    print()

    # Helpful tuning hints
    print("=== Hints ===")
    print("* Increase nprobe → higher recall, slower.")
    print("* Increase nlist → faster (smaller clusters), but needs more training data.")
    print("* Increase m or nbits → higher recall, more memory & slower codes.")
    print("* If accuracy is king and RAM is fine, compare with HNSW (IndexHNSWFlat).")
    print("* For cosine, keep vectors L2-normalized. For L2, do not normalize.\n")

if __name__ == "__main__":
    main()
