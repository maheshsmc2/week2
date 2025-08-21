import os
import json
import uuid
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from chromadb.config import Settings

def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

def _pretty(rows):
    lines = []
    for r in rows:
        meta = r.get("meta", {})
        score = r.get("score", None)
        score_str = f" — score: {score:.4f}" if isinstance(score, (float, int)) else ""
        lines.append(f"- **{r.get('id', '')}**{score_str}\n  \n  {r.get('text','')}\n  \n  *metadata:* `{json.dumps(meta, ensure_ascii=False)}`")
    return "\n\n".join(lines) if lines else "_No results_"

class DualStore:
    def __init__(self):
        self.model_name = "intfloat/multilingual-e5-small"
        self.model = SentenceTransformer(self.model_name)
        self.docs = []
        self.embs = None
        self.id_to_idx = {}
        self.faiss_index = None
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.chroma_client.create_collection(name="dualstore", metadata={"hnsw:space":"cosine"})

    def _embed_texts(self, texts):
        return np.asarray(self.model.encode(texts, batch_size=32, show_progress_bar=False), dtype=np.float32)

    def seed_sample_docs(self):
        sample = [
            {"id": "doc1", "text": "passage: Law code 123: Property rights in urban areas.", "meta": {"license": "123", "code_type": "property", "jurisdiction": "JP", "year": 2020}},
            {"id": "doc2", "text": "passage: Law code 456: Regulations for vehicle licenses and registration.", "meta": {"license": "456", "code_type": "transport", "jurisdiction": "JP", "year": 2021}},
            {"id": "doc3", "text": "passage: Law code 789: Food safety guidelines for restaurants.", "meta": {"license": "789", "code_type": "health", "jurisdiction": "JP", "year": 2019}},
            {"id": "id001", "text": "passage: ID Card format: Citizen must carry ID with 12-digit number.", "meta": {"license": "ID-AAA", "code_type": "id_card", "jurisdiction": "IN", "year": 2018}},
            {"id": "lic987", "text": "passage: Driver license number rules: format AA-0000 valid across states.", "meta": {"license": "DL-987", "code_type": "licensing", "jurisdiction": "IN", "year": 2022}},
        ]
        self.add_documents(sample, rebuild=True)

    def rebuild_faiss(self):
        if not self.docs:
            self.faiss_index = None
            self.embs = None
            self.id_to_idx = {}
            return
        vecs = _normalize(self.get_embeddings())
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs.astype(np.float32))
        self.faiss_index = index

    def rebuild_chroma(self):
        try:
            self.chroma_client.delete_collection("dualstore")
        except Exception:
            pass
        self.collection = self.chroma_client.create_collection(name="dualstore", metadata={"hnsw:space":"cosine"})
        if not self.docs:
            return
        ids = [d["id"] for d in self.docs]
        docs = [d["text"] for d in self.docs]
        metas = [d.get("meta", {}) for d in self.docs]
        embs = self.get_embeddings().tolist()
        self.collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    def get_embeddings(self):
        if self.embs is None or self.embs.shape[0] != len(self.docs):
            texts = [d["text"] for d in self.docs]
            self.embs = self._embed_texts(texts)
            self.id_to_idx = {d["id"]: i for i, d in enumerate(self.docs)}
        return self.embs

    def add_documents(self, new_docs, rebuild=False):
        for d in new_docs:
            if "id" not in d or not d["id"]:
                d["id"] = str(uuid.uuid4())[:8]
            d.setdefault("meta", {})
        self.docs.extend(new_docs)
        new_texts = [d["text"] for d in new_docs]
        new_embs = self._embed_texts(new_texts)
        if self.embs is None:
            self.embs = new_embs
        else:
            self.embs = np.vstack([self.embs, new_embs])
        offset = len(self.docs) - len(new_docs)
        for i, d in enumerate(new_docs):
            self.id_to_idx[d["id"]] = offset + i
        if self.faiss_index is None or rebuild:
            self.rebuild_faiss()
        else:
            self.faiss_index.add(_normalize(new_embs).astype(np.float32))
        if rebuild:
            self.rebuild_chroma()
        else:
            self.collection.add(
                ids=[d["id"] for d in new_docs],
                documents=[d["text"] for d in new_docs],
                metadatas=[d.get("meta", {}) for d in new_docs],
                embeddings=new_embs.tolist(),
            )

    def search_faiss(self, query, top_k=5):
        if not self.docs or self.faiss_index is None:
            return []
        q = self._embed_texts([f"query: {query}"])
        q = _normalize(np.asarray(q, dtype=np.float32))
        scores, idxs = self.faiss_index.search(q, top_k)
        rows = []
        for i, sc in zip(idxs[0].tolist(), scores[0].tolist()):
            if i < 0 or i >= len(self.docs):
                continue
            d = self.docs[i]
            rows.append({"id": d["id"], "text": d["text"], "meta": d.get("meta", {}), "score": float(sc)})
        return rows

    def search_chroma(self, query, top_k=5, where=None):
        if not self.docs:
            return []
        emb = self._embed_texts([f"query: {query}"]).tolist()
        res = self.collection.query(query_embeddings=emb, n_results=top_k, where=where, include=["metadatas", "distances", "documents", "ids"])
        rows = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i in range(len(ids)):
            rows.append({"id": ids[i], "text": docs[i], "meta": metas[i], "score": float(1.0 - dists[i]) if dists[i] is not None else None})
        return rows

store = DualStore()
store.seed_sample_docs()

with gr.Blocks(title="FAISS vs Chroma — Multilingual Retriever", css="footer {visibility: hidden}") as demo:
    gr.Markdown("# 🔎 FAISS vs Chroma — Multilingual Retriever\\nUsing intfloat/multilingual-e5-small (supports English, Japanese, Hindi, etc.).")

    with gr.Tabs():
        with gr.Tab("Search"):
            query = gr.Textbox(label="Your query", placeholder="e.g., 車両登録のルール / vehicle registration rules", lines=2)
            topk = gr.Slider(1, 10, value=5, step=1, label="Top‑K")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### FAISS")
                    btn_faiss = gr.Button("Search FAISS")
                    out_faiss = gr.Markdown(value="_No results yet_")
                with gr.Column():
                    gr.Markdown("### Chroma (filters)")
                    lic = gr.Textbox(label="Filter: license")
                    juris = gr.Dropdown(choices=["", "JP", "IN"], value="", label="Filter: jurisdiction")
                    code_type = gr.Dropdown(choices=["", "property", "transport", "health", "id_card", "licensing"], value="", label="Filter: code_type")
                    btn_chroma = gr.Button("Search Chroma")
                    out_chroma = gr.Markdown(value="_No results yet_")

            btn_faiss.click(lambda q,k: _pretty(store.search_faiss(q,int(k))), [query, topk], [out_faiss])
            btn_chroma.click(lambda q,k,l,j,c: _pretty(store.search_chroma(q,int(k),where={"license":l} if l else None)), [query, topk, lic, juris, code_type], [out_chroma])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
