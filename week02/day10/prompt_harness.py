TEMPLATES = [
    "Answer briefly: {q}",
    "You are a helpful assistant. {q}",
    "Explain step by step and cite sources: {q}"
]

def main():
    q = "Define retrieval-augmented generation."
    for t in TEMPLATES:
        prompt = t.format(q=q)
        print(f"RUN: {prompt}")
        print("(Call your LLM here; record outputs)")

if __name__ == "__main__":
    main()
