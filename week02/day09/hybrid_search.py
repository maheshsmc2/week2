import argparse
from utils.retrieval import bm25_search, dense_search, rrf_merge

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=10)
    args = ap.parse_args()
    q = "What is vector search?"
    bm = bm25_search(q, args.k)
    dn = dense_search(q, args.k)
    fused = rrf_merge([bm, dn], k=args.k)
    print({"top": fused[:3]})
    print("Note: replace stubs with your actual indices and embedding models.")

if __name__ == "__main__":
    main()
