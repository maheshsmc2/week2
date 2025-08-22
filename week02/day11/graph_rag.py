
from typing import List, Tuple, Dict, Any, Optional
import networkx as nx

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
    def __init__(self, triples: Optional[List[Tuple[str, str, str]]] = None):
        self.G = nx.DiGraph()
        for s, r, o in (triples or TRIPLES):
            self.G.add_node(s, label=s)
            self.G.add_node(o, label=o)
            self.G.add_edge(s, o, relation=r)

    def relation_query(self, relation: str) -> List[Tuple[str, str]]:
        return [(u, v) for u, v, d in self.G.edges(data=True) if d.get("relation") == relation]

    def aliases(self, term: str) -> List[str]:
        # follow "aka" both directions to collect aliases
        aliases = {term}
        for u, v, d in self.G.edges(data=True):
            if d.get("relation") == "aka" and (u == term or v == term):
                aliases.add(u)
                aliases.add(v)
        return list(aliases)

    def introduced_by(self, policy: str) -> List[str]:
        answers = set()
        cand_terms = self.aliases(policy)
        for u, v, d in self.G.edges(data=True):
            if d.get("relation") == "introduced" and v in cand_terms:
                answers.add(u)
        return sorted(answers)

    def president_when_policy(self, policy: str) -> List[str]:
        people = self.introduced_by(policy)
        presidents = set()
        for p in people:
            for u, v, d in self.G.edges(data=True):
                if u == p and d.get("relation") == "was_president_of":
                    presidents.add(u)
        return sorted(presidents)

    def simple_answer(self, question: str) -> Dict[str, Any]:
        q = question.lower().strip()
        trace: List[str] = []
        if "who" in q and ("introduce" in q or "introduced" in q):
            # attempt to extract policy name by naive last token chunking
            # fallback: look through graph for any 'introduced' relation
            for u, v in self.relation_query("introduced"):
                if v.lower() in q or any(a.lower() in q for a in self.aliases(v)):
                    ans = {"answer": u, "reasoning": [f"Found edge: {u} -introduced-> {v}"]}
                    return ans
            # fallback: first introduced edge
            edges = self.relation_query("introduced")
            if edges:
                u, v = edges[0]
                return {"answer": u, "reasoning": [f"Fallback: {u} introduced {v}"]}
            return {"answer": "Not found", "reasoning": ["No introduced relations present."]}

        if "who" in q and "president" in q and ("when" in q or "at the time" in q):
            # super-naive: look for any policy mentioned and then confirm presidency
            candidates = []
            for u, v, d in self.G.edges(data=True):
                if d.get("relation") == "introduced" and (v.lower() in q or any(a.lower() in q for a in self.aliases(v))):
                    candidates.append((u, v))
            if not candidates:
                # fallback to most prominent policy we know
                candidates = [(u, v) for u, v, d in self.G.edges(data=True) if d.get("relation") == "introduced"]
            for person, pol in candidates:
                trace.append(f"Step1: {person} -introduced-> {pol}")
                # check presidency
                is_pres = any(d.get("relation") == "was_president_of" and u == person for u, v, d in self.G.edges(data=True))
                if is_pres:
                    trace.append(f"Step2: {person} -was_president_of-> USA")
                    return {"answer": person, "reasoning": trace}
            return {"answer": "Not found", "reasoning": trace + ["No candidate was president."]}

        # default: list some helpful relations
        edges_preview = [f"{u} -{d.get('relation')}-> {v}" for u, v, d in list(self.G.edges(data=True))[:5]]
        return {"answer": "Try asking about 'who introduced <policy>' or 'who was president when <policy> was introduced?'", "reasoning": ["Preview edges:"] + edges_preview}
