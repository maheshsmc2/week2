
import gradio as gr
from graph_rag import GraphRAG

rag = GraphRAG()

DESCRIPTION = """
# Mini Graph-RAG Demo (Day 11) — *Embeddings Upgrade*
Ask things like:
- **Who introduced Obamacare?**
- **Who was president when the Affordable Care Act was introduced?**
- **Who brought in the ACA?** (embedding synonym test)
"""

def answer(question: str):
    res = rag.simple_answer(question or "")
    ans = res.get("answer", "")
    reasoning = res.get("reasoning", [])
    top = res.get("top_matches", [])
    reasoning_text = "\n".join(f"- {step}" for step in reasoning) if reasoning else ""
    top_md = "\n".join(f"- {row['edge']}  (score={row['score']})" for row in top) if top else ""
    return ans, reasoning_text, top_md

with gr.Blocks(title="Mini Graph-RAG (Embeddings)") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        q = gr.Textbox(label="Your question", scale=3, placeholder="Try: Who brought in the Affordable Care Act?")
        btn = gr.Button("Ask", scale=1)
    a = gr.Textbox(label="Answer", interactive=False)
    r = gr.Textbox(label="Reasoning Trace", lines=8, interactive=False)
    m = gr.Markdown(value="", elem_id="top-matches")
    btn.click(fn=answer, inputs=q, outputs=[a, r, m])
    q.submit(fn=answer, inputs=q, outputs=[a, r, m])

if __name__ == "__main__":
    demo.launch()
