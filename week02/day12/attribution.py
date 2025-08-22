def add_citations(answer: str, sources):
    # Placeholder: wrap cited spans in [[...]] with (doc_id)
    return answer + " [CITE: doc_1]"

def main():
    ans = "RAG combines retrieval with generation."
    cited = add_citations(ans, sources=[("doc_1", "RAG...")])
    print(cited)

if __name__ == "__main__":
    main()
