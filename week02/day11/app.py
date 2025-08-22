
import gradio as gr
from graph_rag import GraphRAG

rag = GraphRAG()

DESCRIPTION = """
# Mini Graph-RAG Demo (Day 11)
Ask questions like:
- **Who introduced Obamacare?**
- **Who was president when Obamacare was introduced?**
"""

def answer(question: str):
    res = rag.simple_answer(question or "")
    ans = res.get("answer", "")
    reasoning = res.get("reasoning", [])
    reasoning_text = "\n".join(f"- {step}" for step in reasoning)
    return ans, reasoning_text

with gr.Blocks(title="Mini Graph-RAG") as demo:
    gr.Markdown(DESCRIPTION)
    q = gr.Textbox(label="Your question")
    btn = gr.Button("Ask")
    a = gr.Textbox(label="Answer", interactive=False)
    r = gr.Textbox(label="Reasoning Trace", lines=8, interactive=False)
    btn.click(fn=answer, inputs=q, outputs=[a, r])
    q.submit(fn=answer, inputs=q, outputs=[a, r])

if __name__ == "__main__":
    demo.launch()
