"""Retrieval module for FinRAG."""

import math
from typing import Any, Dict, List

from finrag.chunk_utils import build_candidate_chunks
from finrag.embeddings import EmbeddingStore
from finrag.tools import co


# Pure-Python BM25 implementation; no external dependencies
class BM25Okapi:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.N if self.N else 0
        df = {}
        for doc in corpus:
            seen = set()
            for term in doc:
                if term not in seen:
                    df[term] = df.get(term, 0) + 1
                    seen.add(term)
        self.idf = {
            term: math.log(1 + (self.N - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }
        self.freqs = [{t: doc.count(t) for t in set(doc)} for doc in corpus]

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = []
        for idx, doc in enumerate(self.corpus):
            score = 0.0
            doc_len = len(doc)
            for term in query_tokens:
                if term in self.freqs[idx]:
                    f = self.freqs[idx][term]
                    idf = self.idf.get(term, 0.0)
                    numerator = f * (self.k1 + 1)
                    denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * numerator / denominator
            scores.append(score)
        return scores


def retrieve_evidence(
    turn: Dict[str, Any], question: str, top_k: int = 8, bm25_k: int = 20
) -> List[str]:
    """Hybrid retrieval: BM25 + embedding fusion, returning top_k chunk_ids."""
    candidate_chunks = build_candidate_chunks(turn)
    texts = [c["text"] for c in candidate_chunks]
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(question.split())
    # Embedding similarity
    store = EmbeddingStore()
    q_embed = store.get_embedding(question)
    chunk_embeds = store.get_embeddings(texts)

    # cosine similarity via pure Python
    def cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return dot / norm if norm else 0.0

    embed_scores = [cosine(q_embed, e) for e in chunk_embeds]
    # Fusion of top bm25_k indices
    bm25_top = set(
        sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:bm25_k]
    )
    embed_top = set(
        sorted(range(len(embed_scores)), key=lambda i: embed_scores[i], reverse=True)[:bm25_k]
    )
    fused = list(bm25_top.union(embed_top))
    # Prepare fused chunks for rerank
    fused_chunks = [candidate_chunks[i] for i in fused]
    docs = [chunk["text"] for chunk in fused_chunks]
    # Cohere rerank API: returns results with .index attribute
    resp = co.rerank(
        model="rerank-english-v2.0",
        query=question,
        documents=docs,
        top_n=top_k,
    )
    return [fused_chunks[r.index]["chunk_id"] for r in resp.results]
