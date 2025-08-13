import logging
from typing import Any, Dict, List, Optional
import math

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, store: Any, embedder: Optional[Any] = None):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5, signals: Optional[Dict[str, Any]] = None) -> List[Any]:
        if not self.store:
            return []
        try:
            retriever = self.store.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(query)
            # Basic work-stress filtering and MMR-style dedup heuristic
            docs = docs or []
            scored: List[tuple] = []
            for d in docs:
                text = getattr(d, 'page_content', '') or ''
                score = 0.0
                wl_boost = 0.0
                for kw in [
                    'work', 'job', 'deadline', 'workload', 'boss', 'manager', 'office', 'overtime', 'project', 'deliverables', 'sprint'
                ]:
                    if kw in text.lower():
                        wl_boost += 0.05
                score += wl_boost
                scored.append((d, score))

            # Sort by boosted score desc while preserving base order using tie-breaker
            docs_boosted = [d for d, _ in sorted(scored, key=lambda x: x[1], reverse=True)]

            # Compute embeddings and apply cosine-similarity MMR
            try:
                if self.embedder is not None:
                    query_vec = self.embedder.embed_query(query)
                    doc_texts = [getattr(d, 'page_content', '') or '' for d in docs_boosted]
                    doc_vecs = self.embedder.embed_documents([t[:512] for t in doc_texts])
                    selected = self._mmr_select(docs_boosted, query_vec, doc_vecs, lambda_param=0.7, top_n=min(k, len(docs_boosted)))
                    return selected
            except Exception:
                logger.debug("MMR embedding failed; falling back to overlap heuristic")

            # Fallback: overlap heuristic
            selected_heur: List[Any] = []
            seen: List[str] = []
            for d in docs_boosted:
                t = (getattr(d, 'page_content', '') or '')[:400]
                grams = set(t.lower().split())
                if not grams:
                    selected_heur.append(d)
                    continue
                too_similar = False
                for s in seen:
                    base = set(s.split())
                    inter = len(base & grams)
                    union = len(base | grams) or 1
                    jacc = inter / union
                    if jacc > 0.6:
                        too_similar = True
                        break
                if not too_similar:
                    selected_heur.append(d)
                    seen.append(' '.join(list(grams)[:100]))
            return selected_heur
        except Exception:
            logger.exception("RAG retrieval failed")
            return []

    def build_context(self, docs: List[Any], max_docs: int = 2) -> str:
        final_docs = docs[:max_docs]
        return "\n\n".join([doc.page_content for doc in final_docs if getattr(doc, 'page_content', None)])

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        import math as _m
        if not a or not b:
            return 0.0
        num = sum(x*y for x, y in zip(a, b))
        da = _m.sqrt(sum(x*x for x in a)) or 1e-8
        db = _m.sqrt(sum(y*y for y in b)) or 1e-8
        return num / (da * db)

    def _mmr_select(self, docs: List[Any], query_vec: List[float], doc_vecs: List[List[float]], lambda_param: float = 0.7, top_n: int = 5) -> List[Any]:
        if not docs:
            return []
        relevance = [self._cosine(query_vec, v) for v in doc_vecs]
        selected_idx: List[int] = []
        candidates = set(range(len(docs)))
        # Start with most relevant
        first = max(candidates, key=lambda i: relevance[i])
        selected_idx.append(first)
        candidates.remove(first)
        while candidates and len(selected_idx) < top_n:
            def score(i: int) -> float:
                max_sim = max(self._cosine(doc_vecs[i], doc_vecs[j]) for j in selected_idx) if selected_idx else 0.0
                return lambda_param * relevance[i] - (1 - lambda_param) * max_sim
            nxt = max(candidates, key=score)
            selected_idx.append(nxt)
            candidates.remove(nxt)
        return [docs[i] for i in selected_idx]


