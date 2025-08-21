def ndcg_at_k(relevances, k=10):
    """Compute a simple nDCG@k given a list of binary/graded relevances sorted by rank."""
    import math
    rel = relevances[:k]
    dcg = sum((2**r - 1) / math.log2(i+2) for i, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum((2**r - 1) / math.log2(i+2) for i, r in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg

def recall_at_k(relevances, k=10):
    rel = relevances[:k]
    total_rel = sum(1 for r in relevances if r > 0)
    return 0.0 if total_rel == 0 else sum(1 for r in rel if r > 0) / total_rel
