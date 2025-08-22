
from typing import List, Tuple, Dict, Any, Optional
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import torch

TRIPLES: List[Tuple[str, str, str]] = [
    ("Barack Obama", "was_president_of", "USA"),
    ("Barack Obama", "introduced", "Obamacare"),
    ("USA", "has_capital", "Washington D.C."),
    ("Joe Biden", "was_vice_president_of", "USA"),
    ("Joe Biden", "was_president_of", "USA"),
    ("Affordable Care Act", "aka", "Obamacare"),
    ("Obamacare", "aka", "Affordable Care Act"),
]

class GraphRAG:
    """
    Mini Graph-RAG with two retrieval modes:
      1) Rule-based graph traversal for specific patterns
      2) Embeddings fallback using sentence-transformers (all-MiniLM-L6-v2)
    """
    def __init__(self, triples: Optional[List[Tuple[str, str, str]]] = None, model_name: str = "all-MiniLM-L6-v2"):
        self.G = nx.DiGraph()
        self.triples = list(triples or TRIPLES)
        for s, r, o in self.triples:
            self.G.add_node(s, label=s)
            self.G.add_node(o, label=o)
            self.G.add_edge(s, o, relation=r)

        # Embedding model (CPU)
        self.model = SentenceTransformer(model_name)
        # Precompute embeddings for edges (stringify each triple)
        self.edge_strings: List[str] = [f"{s} {r.replace('_',' ')} {o}" for s, r, o in self.triples]
        self.edge_embs = self.model.encode(self.edge_strings, convert_to_tensor=True, normalize_embeddings=True)

    # ---------- Graph utilities ----------
    def relation_query(self, relation: str) -> List[Tuple[str, str]]:
        return [(u, v) for u, v, d in self.G.edges(data=True) if d.get("relation") == relation]

    def aliases(self, term: str) -> List[str]:
        """Return known aliases for a term following 'aka' edges (both directions)."""
        outs = {term}
        for u, v, d in self.G.edges(data=True):
            if d.get("relation") == "aka" and (u == term or v == term):
                outs.add(u); outs.add(v)
        return list(outs)

    def introduced_by(self, policy: str) -> List[str]:
        answers = set()
        cand_terms = {a.lower() for a in self.aliases(policy)} | {policy.lower()}
        for u, v, d in self.G.edges(data=True):
            if d.get("relation") == "introduced" and v.lower() in cand_terms:
                answers.add(u)
        return sorted(answers)

    def is_president(self, person: str) -> bool:
        return any(d.get("relation") == "was_president_of" and u == person for u, v, d in self.G.edges(data=True))

    # ---------- Embedding utilities ----------
    def semantic_search_edges(self, query: str, top_k: int = 5) -> List[Tuple[str, Tuple[str, str, str], float]]:
        """Return top_k edge strings with scores and original triples."""
        q_emb = self.model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        scores = util.pytorch_cos_sim(q_emb, self.edge_embs)[0]
        topk = torch.topk(scores, k=min(top_k, scores.shape[0]))
        results: List[Tuple[str, Tuple[str, str, str], float]] = []
        for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            triple = self.triples[idx]
            results.append((self.edge_strings[idx], triple, float(score)))
        return results

    # ---------- QA logic ----------
    def simple_answer(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip()
        trace: List[str] = []

        # 1) Rule-based patterns
        lower = q.lower()
        if "who" in lower and ("introduce" in lower or "introduced" in lower or "brought in" in lower or "rolled out" in lower):
            # Try to detect a policy term by matching any known object from "introduced" edges
            introduced_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d.get("relation") == "introduced"]
            for person, policy in introduced_edges:
                # if either the object or its alias is mentioned in the question
                if policy.lower() in lower or any(a.lower() in lower for a in self.aliases(policy)):
                    trace.append(f"Matched policy mention: {policy}")
                    return {"answer": person, "reasoning": trace + [f"Graph edge: {person} -introduced-> {policy}"], "top_matches": []}

        if "who" in lower and "president" in lower and ("when" in lower or "at the time" in lower or "introduced" in lower):
            # Find introducer, then check if they were president
            introduced_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d.get("relation") == "introduced"]
            candidates = []
            for person, policy in introduced_edges:
                if policy.lower() in lower or any(a.lower() in lower for a in self.aliases(policy)):
                    candidates.append((person, policy))
            if not candidates and introduced_edges:
                # fallback to first known policy if none explicitly mentioned
                candidates = [introduced_edges[0]]
                trace.append("No explicit policy mention found; using fallback introduced edge.")
            for person, pol in candidates:
                trace.append(f"Step1: {person} -introduced-> {pol}")
                if self.is_president(person):
                    trace.append(f"Step2: {person} -was_president_of-> USA")
                    return {"answer": person, "reasoning": trace, "top_matches": []}

        # 2) Embeddings fallback
        sem = self.semantic_search_edges(q, top_k=5)
        if sem:
            best_text, (s, r, o), score = sem[0]
            ans = s if r == "introduced" else s  # default to subject for "who"-style questions
            top_matches = [{"edge": t, "score": round(sc, 3)} for t, _, sc in sem]
            return {
                "answer": ans,
                "reasoning": [f"Semantic match → '{best_text}' (score={score:.3f})"],
                "top_matches": top_matches
            }

        return {"answer": "Not found", "reasoning": ["No rule or embedding match."], "top_matches": []}
