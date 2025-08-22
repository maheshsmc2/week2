import time, os
import streamlit as st

st.set_page_config(page_title="RAG Index Upgrade — FAISS + HNSW", layout="wide")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS, Chroma
except Exception:
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS, Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_DIR = "faiss_index"
CHROMA_DIR = "chroma_hnsw"
COLLECTION = "rag_hnsw"

@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = DEFAULT_MODEL):
    return HuggingFaceEmbeddings(model_name=model_name)

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    return splitter.create_documents([text])

def ensure_text(sample_path: str):
    uploaded = st.file_uploader("Upload a .txt file (optional)", type=["txt"])
    if uploaded is not None:
        text = uploaded.read().decode("utf-8", errors="ignore")
        return text, "(uploaded)"
    with open(sample_path, "r", encoding="utf-8") as f:
        return f.read(), sample_path

def build_faiss(docs, emb, out_dir=FAISS_DIR):
    db = FAISS.from_documents(docs, emb)
    db.save_local(out_dir)
    return out_dir

def build_chroma(docs, emb, out_dir=CHROMA_DIR, collection=COLLECTION):
    db = Chroma.from_documents(docs, emb, collection_name=collection, persist_directory=out_dir)
    db.persist()
    return out_dir

def load_faiss(emb, dir=FAISS_DIR):
    return FAISS.load_local(dir, emb, allow_dangerous_deserialization=True)

def load_chroma(emb, dir=CHROMA_DIR):
    return Chroma(persist_directory=dir, embedding_function=emb)

st.sidebar.header("Build or Load Index")
model = st.sidebar.text_input("Embedding model", value=DEFAULT_MODEL)
emb = get_embedder(model)

sample_path = os.path.join("data", "corpus.txt")
text, source = ensure_text(sample_path)
st.sidebar.caption(f"Using text from: {source}")

if st.sidebar.button("Build FAISS index"):
    with st.spinner("Building FAISS…"):
        docs = chunk_text(text)
        out = build_faiss(docs, emb)
    st.sidebar.success(f"FAISS saved to ./{out}")

if st.sidebar.button("Build HNSW/Chroma index"):
    with st.spinner("Building HNSW/Chroma…"):
        docs = chunk_text(text)
        out = build_chroma(docs, emb)
    st.sidebar.success(f"Chroma saved to ./{out}")

st.title("RAG Index Upgrade — FAISS vs HNSW")
backend = st.selectbox("Backend", options=["faiss", "chroma"], index=0)
k = st.number_input("Top‑k", min_value=1, max_value=10, value=3)
question = st.text_input("Your question", value="What are the symptoms of diabetes?")

col1, col2 = st.columns(2)
if st.button("Search"):
    with st.spinner("Searching…"):
        t0 = time.time()
        if backend == "faiss":
            db = load_faiss(emb, FAISS_DIR)
        else:
            db = load_chroma(emb, CHROMA_DIR)
        _ = time.time()
        results = db.similarity_search(question, k=k)
        t1 = time.time()
    with col1:
        for i, r in enumerate(results, 1):
            st.markdown(f"**Top {i}**")
            st.write(r.page_content)
    with col2:
        st.metric("Search time (ms)", f"{(t1 - _)*1000:.2f}")
