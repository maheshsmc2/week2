# app.py — FAISS IVF+PQ & HNSW vs Flat with CSV + Plot
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tempfile
import os

try:
    import faiss  # faiss-cpu
except Exception as e:
    raise RuntimeError("FAISS import failed. Ensure 'faiss-cpu' is in requirements.txt") from e

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
    return faiss.IndexFlatIP(d) if metric == "cosine" else faiss.IndexFlatL2(d)

def build_ivfpq(d: int, nlist: int, m: int, nbits: int, metric: str):
    if metric == "cosine":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    else:
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_L2)
    return index

def build_hnsw(d: int, M: int):
    # Use L2 metric; for cosine we normalized, so rankings are equivalent.
    return faiss.IndexHNSWFlat(d, M)

def recall_at_k(I_true: np.ndarray, I_test: np.ndarray, k: int) -> float:
    nq = I_true.shape[0]
    inter = 0
    for i in range(nq):
        inter += len(set(I_true[i, :k]).intersection(I_test[i, :k]))
    return inter / float(nq * k)

def benchmark(n, nq, d, k, metric, select_ivfpq, select_hnsw,
              nlist, nprobe, m, nbits,
              M, efC, efS, seed):
    if not (select_ivfpq or select_hnsw):
        return "Select at least one ANN method.", None, None, None, "Select a method"
    # Modest safety to avoid OOM in small Spaces
    if n * d > 60_000_000:
        return "N and d are too large for the demo Space. Reduce N or d.", None, None, None, "Try smaller settings"

    xb, xq = make_data(int(n), int(nq), int(d), metric, int(seed))

    # Flat baseline
    flat = build_flat(int(d), metric)
    flat.add(xb)
    _ = flat.search(xq[:1], 1)  # warmup
    t0 = time.perf_counter()
    Df, If = flat.search(xq, int(k))
    dt_flat = time.perf_counter() - t0
    qps_flat = nq / dt_flat if dt_flat > 0 else float("inf")
    ms_flat = (dt_flat / nq) * 1000 if nq > 0 else 0.0

    rows = [{
        "Method": "Flat",
        "Train/Build (s)": 0.0,
        "Search Time (s)": round(dt_flat, 6),
        "QPS": round(qps_flat, 2),
        "ms/query": round(ms_flat, 3),
        f"Recall@{k}": 1.0
    }]

    log_lines = []
    log_lines.append("=== Settings ===")
    log_lines.append(f"N={n}, Nq={nq}, d={d}, k={k}, metric={metric}")
    log_lines.append(f"IVF: nlist={nlist}, nprobe={nprobe}, PQ: m={m}, nbits={nbits}")
    log_lines.append(f"HNSW: M={M}, efConstruction={efC}, efSearch={efS}")
    log_lines.append("")

    # IVF+PQ
    if select_ivfpq:
        ivfpq = build_ivfpq(int(d), int(nlist), int(m), int(nbits), metric)
        t_train0 = time.perf_counter()
        ivfpq.train(xb)
        t_train1 = time.perf_counter()
        ivfpq.add(xb)
        t_train2 = time.perf_counter()
        ivfpq.nprobe = int(nprobe)
        _ = ivfpq.search(xq[:1], int(k))
        t1 = time.perf_counter()
        Di, Ii = ivfpq.search(xq, int(k))
        dt_ivf = time.perf_counter() - t1
        qps_ivf = nq / dt_ivf if dt_ivf > 0 else float("inf")
        ms_ivf = (dt_ivf / nq) * 1000 if nq > 0 else 0.0
        r_at_k = recall_at_k(If, Ii, int(k))
        rows.append({
            "Method": "IVF+PQ",
            "Train/Build (s)": round((t_train1 - t_train0) + (t_train2 - t_train1), 6),
            "Search Time (s)": round(dt_ivf, 6),
            "QPS": round(qps_ivf, 2),
            "ms/query": round(ms_ivf, 3),
            f"Recall@{k}": round(r_at_k, 4)
        })
        log_lines.append(f"[IVF+PQ] Train={t_train1 - t_train0:.3f}s, Add={t_train2 - t_train1:.3f}s; "
                         f"Search={dt_ivf:.3f}s; Recall@{k}={r_at_k:.4f} (nprobe={nprobe})")

    # HNSW
    if select_hnsw:
        hnsw = build_hnsw(int(d), int(M))
        hnsw.hnsw.efConstruction = int(efC)
        hnsw.hnsw.efSearch = int(efS)
        t_b0 = time.perf_counter()
        hnsw.add(xb)
        t_b1 = time.perf_counter()
        _ = hnsw.search(xq[:1], int(k))
        t2 = time.perf_counter()
        Dh, Ih = hnsw.search(xq, int(k))
        dt_h = time.perf_counter() - t2
        qps_h = nq / dt_h if dt_h > 0 else float("inf")
        ms_h = (dt_h / nq) * 1000 if nq > 0 else 0.0
        r_h = recall_at_k(If, Ih, int(k))
        rows.append({
            "Method": "HNSW",
            "Train/Build (s)": round(t_b1 - t_b0, 6),
            "Search Time (s)": round(dt_h, 6),
            "QPS": round(qps_h, 2),
            "ms/query": round(ms_h, 3),
            f"Recall@{k}": round(r_h, 4)
        })
        log_lines.append(f"[HNSW] Build={t_b1 - t_b0:.3f}s; Search={dt_h:.3f}s; Recall@{k}={r_h:.4f} "
                         f"(M={M}, efSearch={efS}, efConstruction={efC})")

    import pandas as pd
    df = pd.DataFrame(rows)

    # Plot Recall vs QPS
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df["QPS"], df[f"Recall@{k}"])
    for idx, row in df.iterrows():
        ax.annotate(row["Method"], (row["QPS"], row[f"Recall@{k}"]), xytext=(5,5), textcoords="offset points")
    ax.set_xlabel("Queries per second (higher is better)")
    ax.set_ylabel(f"Recall@{k} (higher is better)")
    ax.set_title("Recall vs Throughput")

    # Save CSV for download
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    csv_path = os.path.join(tmpdir, "faiss_benchmark_results.csv")
    df.to_csv(csv_path, index=False)

    log_text = "\n".join(log_lines)
    return log_text, df, fig, csv_path, "Done ✅"

with gr.Blocks(title="FAISS — IVF+PQ & HNSW vs Flat") as demo:
    gr.Markdown("# 🔎 FAISS — IVF+PQ & HNSW vs Flat\nCompare approximate search methods with a Flat ground truth. Download CSV and view plots.")
    with gr.Row():
        with gr.Column():
            n = gr.Slider(10_000, 500_000, value=50_000, step=5_000, label="N (database vectors)")
            nq = gr.Slider(50, 2_000, value=300, step=50, label="Nq (query vectors)")
            d = gr.Slider(16, 256, value=64, step=8, label="d (dimension)")
            k = gr.Slider(1, 50, value=10, step=1, label="Top-k")
            metric = gr.Radio(choices=["cosine", "l2"], value="cosine", label="Metric")
            select_ivfpq = gr.Checkbox(value=True, label="Use IVF+PQ")
            select_hnsw = gr.Checkbox(value=True, label="Use HNSW")
        with gr.Column():
            gr.Markdown("### IVF+PQ")
            nlist = gr.Slider(64, 8192, value=1024, step=64, label="IVF nlist")
            nprobe = gr.Slider(1, 256, value=16, step=1, label="IVF nprobe")
            m = gr.Slider(4, 64, value=16, step=4, label="PQ m")
            nbits = gr.Slider(4, 12, value=8, step=1, label="PQ nbits")
            gr.Markdown("### HNSW")
            M = gr.Slider(4, 64, value=32, step=1, label="M (neighbors per node)")
            efC = gr.Slider(8, 400, value=100, step=1, label="efConstruction")
            efS = gr.Slider(1, 400, value=64, step=1, label="efSearch")
    run_btn = gr.Button("Run Benchmark")
    out_text = gr.Textbox(lines=14, label="Log")
    out_df = gr.Dataframe(label="Results table")
    out_plot = gr.Plot(label="Recall vs Throughput")
    out_file = gr.File(label="Download CSV")
    status = gr.Textbox(label="Status", value="Ready")
    run_btn.click(
        benchmark,
        inputs=[n, nq, d, k, metric, select_ivfpq, select_hnsw,
                nlist, nprobe, m, nbits,
                M, efC, efS,],
        outputs=[out_text, out_df, out_plot, out_file, status]
    )

if __name__ == "__main__":
    demo.launch()
