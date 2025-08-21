"""Skeleton for reranking pipeline. Replace stubs with actual models."""
import time, argparse
from utils.retrieval import bm25_search
from utils.eval_metrics import ndcg_at_k, recall_at_k

def fake_rerank(candidates):
    # Placeholder: pretend scores improved slightly
    return [(doc, s + 0.01) for doc, s in candidates]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top_k', type=int, default=10)
    args = ap.parse_args()

    q = "What is RAG?"
    t0 = time.time()
    base = bm25_search(q, top_k=args.top_k)
    reranked = fake_rerank(base)
    elapsed_ms = (time.time() - t0) * 1000

    # Fake relevances for illustration
    rels = [1,0,1,0,0,1,0,0,0,0]
    print({"nDCG@10": ndcg_at_k(rels, 10), "Recall@10": recall_at_k(rels, 10), "latency_ms": round(elapsed_ms,1)})
    print("Saved: day08/ablation.csv (fill with real numbers)")

if __name__ == "__main__":
    main()
