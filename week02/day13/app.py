# app.py — Gradio UI for FAISS IVF+PQ vs Flat
import gradio as gr
import numpy as np
import time

try:
    import faiss  # faiss-cpu
except Exception as e:
    raise RuntimeError("FAISS import failed. Make sure faiss-cpu is installed in requirements.txt") from e

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
        return faiss.IndexFlatIP(d)  # cosine via inner product on unit vectors
    else:
        return faiss.IndexFlatL2(d)

def build_ivfpq(d: int, nlist: int, m: int, nbits: int, metric: str):
    if metric == "cosine":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    else:
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_L2)
    return index

def recall_at_k(I_true: np.ndarray, I_test: np.ndarray, k: int) -> float:
    nq = I_true.shape[0]
    inter = 0
    for i in range(nq):
        inter += len(set(I_true[i, :k]).intersection(I_test[i, :k]))
    return inter / float(nq * k)

def run_demo(n, nq, d, k, metric, nlist, nprobe, m, nbits, seed):
    # Safety cap to avoid OOM on small Spaces
    cap = 1_000_000
    if n * d > cap * 8:
        return "Chosen N and d are too large for the demo space. Try reducing N or d.", ""

    xb, xq = make_data(n, nq, d, metric, seed)

    # Flat
    flat = build_flat(d, metric)
    flat.add(xb)
    _ = flat.search(xq[:1], 1)  # warmup
    t0 = time.perf_counter()
    Df, If = flat.search(xq, k)
    dt_flat = time.perf_counter() - t0
    qps_flat = nq / dt_flat if dt_flat > 0 else float("inf")
    ms_flat = (dt_flat / nq) * 1000 if nq > 0 else 0.0

    # IVF+PQ
    ivfpq = build_ivfpq(d, nlist, m, nbits, metric)
    t_train0 = time.perf_counter()
    ivfpq.train(xb)
    t_train1 = time.perf_counter()
    ivfpq.add(xb)
    t_train2 = time.perf_counter()
    ivfpq.nprobe = int(nprobe)
    _ = ivfpq.search(xq[:1], k)  # warmup
    t1 = time.perf_counter()
    Di, Ii = ivfpq.search(xq, k)
    dt_ivf = time.perf_counter() - t1
    qps_ivf = nq / dt_ivf if dt_ivf > 0 else float("inf")
    ms_ivf = (dt_ivf / nq) * 1000 if nq > 0 else 0.0

    r_at_k = recall_at_k(If, Ii, k)

    log = []
    log.append("=== Settings ===")
    log.append(f"N={n}, Nq={nq}, d={d}, k={k}, metric={metric}")
    log.append(f"IVF: nlist={nlist}, nprobe={nprobe} | PQ: m={m}, nbits={nbits}")
    log.append("")
    log.append(f"[Flat] {nq} queries in {dt_flat:.3f}s → {qps_flat:.1f} qps ({ms_flat:.2f} ms/query)")
    log.append(f"[IVF+PQ] Train={t_train1 - t_train0:.3f}s, Add={t_train2 - t_train1:.3f}s")
    log.append(f"[IVF+PQ] {nq} queries in {dt_ivf:.3f}s → {qps_ivf:.1f} qps ({ms_ivf:.2f} ms/query); Recall@{k} = {r_at_k:.4f}")
    log.append("")
    log.append("Hints: Increase nprobe for recall, increase nlist for speed (smaller cells), change m/nbits for compression vs accuracy.")

    return "\n".join(log), "Done ✅"

with gr.Blocks(title="FAISS IVF+PQ vs Flat (Day 13)") as demo:
    gr.Markdown("# 🔎 FAISS IVF+PQ vs Flat\nBenchmark approximate search vs brute force on synthetic vectors.")
    with gr.Row():
        with gr.Column():
            n = gr.Slider(10_000, 500_000, value=50_000, step=5_000, label="N (database vectors)")
            nq = gr.Slider(50, 2_000, value=300, step=50, label="Nq (query vectors)")
            d = gr.Slider(16, 256, value=64, step=8, label="d (dimension)")
            k = gr.Slider(1, 50, value=10, step=1, label="Top-k")
            metric = gr.Radio(choices=["cosine", "l2"], value="cosine", label="Metric")
        with gr.Column():
            nlist = gr.Slider(64, 8192, value=1024, step=64, label="IVF nlist (clusters)")
            nprobe = gr.Slider(1, 128, value=16, step=1, label="IVF nprobe (clusters/search)")
            m = gr.Slider(4, 64, value=16, step=4, label="PQ m (subquantizers)")
            nbits = gr.Slider(4, 12, value=8, step=1, label="PQ nbits (bits per subspace)")
            seed = gr.Slider(0, 10_000, value=42, step=1, label="Random seed")

    run_btn = gr.Button("Run Benchmark")
    out_text = gr.Textbox(lines=15, label="Results")
    status = gr.Textbox(label="Status", value="Ready")

    run_btn.click(
        run_demo,
        inputs=[n, nq, d, k, metric, nlist, nprobe, m, nbits, seed],
        outputs=[out_text, status]
    )

if __name__ == "__main__":
    demo.launch()
