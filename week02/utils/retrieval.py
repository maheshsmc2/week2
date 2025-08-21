# Placeholder retrieval functions. Replace with your real BM25/dense code.
from typing import List, Tuple

def bm25_search(query: str, top_k: int = 10) -> List[Tuple[str, float]]:
    # TODO: integrate with your actual BM25 index
    return [(f"doc_{i}", 1.0/(i+1)) for i in range(top_k)]

def dense_search(query: str, top_k: int = 10) -> List[Tuple[str, float]]:
    # TODO: call your embedding model + ANN index
    return [(f"doc_{i}", 1.0 - i*0.05) for i in range(top_k)]

def rrf_merge(runs: List[List[Tuple[str, float]]], k: int = 10, k_rrf: int = 60):
    # Reciprocal Rank Fusion over multiple ranked lists
    from collections import defaultdict
    score = defaultdict(float)
    for run in runs:
        for rank, (doc_id, _) in enumerate(run):
            score[doc_id] += 1.0 / (k_rrf + rank + 1)
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]
