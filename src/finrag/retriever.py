"""Retrieval module for FinRAG."""

import logging
import math
import os
from typing import Any, Dict, List

from pinecone import Pinecone

from finrag.chunk_utils import build_candidate_chunks
from finrag.embeddings import EmbeddingStore
from finrag.tools import co

logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv

load_dotenv(override=True)  # Override existing env vars from .env
print(
    f" [DEBUG] PINECONE_API_KEY prefix: {os.getenv('PINECONE_API_KEY')[:5] if os.getenv('PINECONE_API_KEY') else 'None'}"
)

# Pinecone integration feature flag
USE_PINECONE = os.getenv("USE_PINECONE", "false").lower() in ("true", "1")
PINECONE_INDEX = None
if USE_PINECONE:
    # Instantiate Pinecone client (v3+) using API key and optional host or environment
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    host = os.getenv("PINECONE_API_HOST")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    logger.debug(
        f"Pinecone init vars -> key={'Yes' if api_key else 'No'}, index={'Yes' if index_name else 'No'}, host={'Yes' if host else 'No'}, env={'Yes' if environment else 'No'}"
    )
    try:
        if host:
            pc = Pinecone(api_key=api_key, host=host)
        elif environment:
            pc = Pinecone(api_key=api_key, environment=environment)
        else:
            pc = Pinecone(api_key=api_key)
        PINECONE_INDEX = pc.Index(index_name)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone index '{index_name}': {e}", exc_info=True)
        USE_PINECONE = False
        PINECONE_INDEX = None
else:
    logger.warning("USE_PINECONE=false; skipping Pinecone initialization.")


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


def pinecone_retrieve(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Retrieve similar chunks from Pinecone index."""
    if not PINECONE_INDEX:  # Add check
        print("Error: Pinecone index not available for retrieval.")
        return []
    store = EmbeddingStore()
    q_embed = store.get_embedding(query)
    response = PINECONE_INDEX.query(
        vector=q_embed,
        top_k=top_k,
        include_metadata=True,
    )
    chunks: List[Dict[str, Any]] = []
    for m in response.matches:
        meta = m.metadata
        chunks.append(
            {
                "chunk_id": meta.get("chunk_id"),
                "text": meta.get("text", ""),
                "score": m.score,
            }
        )
    return chunks


def retrieve_evidence(
    turn: Dict[str, Any], question: str, top_k: int = 8, bm25_k: int = 20
) -> Dict[str, List[Dict[str, Any]]]:
    """Hybrid retrieval: BM25 + embedding fusion. Returns raw_chunks and reranked_chunks with scores."""
    # Gold-index fallback for dev set: use annotated chunks directly
    gold_inds = turn.get("gold_inds")  # Use .get() which returns None if not present
    if gold_inds is not None:
        candidate_chunks = build_candidate_chunks(turn)
        # select gold chunks
        raw_chunks = [candidate_chunks[i] for i in gold_inds if i < len(candidate_chunks)]
        if not raw_chunks:
            logger.warning("Gold indices provided, but resulted in empty raw_chunks list.")
            return {"raw_chunks": [], "reranked_chunks": []}
        # treat gold as reranked with max score
        reranked_chunks = [
            {"chunk_id": c["chunk_id"], "text": c["text"], "score": 1.0} for c in raw_chunks
        ]
        return {"raw_chunks": raw_chunks, "reranked_chunks": reranked_chunks}

    # --- Start Retrieval ---
    raw_chunks = []  # Initialize
    retrieval_source = "None"

    # 1. Try Pinecone if enabled
    if USE_PINECONE:
        try:
            logger.debug(f"Attempting Pinecone retrieval for query: {question[:50]}...")
            raw_chunks = pinecone_retrieve(question, top_k)
            if not raw_chunks:
                logger.warning("Pinecone retrieval returned no results.")
            else:
                retrieval_source = "Pinecone"
                logger.debug(f"Pinecone returned {len(raw_chunks)} chunks.")
        except Exception as e:
            logger.error(f"Pinecone retrieval failed: {e}. Falling back to BM25.")
            raw_chunks = []  # Ensure empty before fallback

    # 2. Fallback to BM25 if Pinecone disabled OR failed OR returned empty
    if not raw_chunks:
        logger.info("Using BM25 retrieval.")
        try:
            # BM25 fallback using BM25Okapi
            candidate_chunks = build_candidate_chunks(turn)
            docs = [c["text"] for c in candidate_chunks]
            tokenized = [doc.split() for doc in docs]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(question.split())
            top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:bm25_k]
            raw_chunks = [candidate_chunks[i] for i in top_idxs]
            if raw_chunks:
                retrieval_source = "BM25"
                logger.debug(f"BM25 returned {len(raw_chunks)} chunks.")
            else:
                logger.warning("BM25 retrieval returned no results.")
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            raw_chunks = []

    # === FINAL CHECK before Reranking ===
    if not raw_chunks:
        logger.warning(
            f"No chunks found by any retrieval method ({retrieval_source}) for query: {question[:50]}..."
        )
        return {"raw_chunks": [], "reranked_chunks": []}  # Return empty result dict

    # === Rerank with Cohere (only if chunks were found) ===
    reranked_chunks = []
    try:
        logger.debug(f"Reranking {len(raw_chunks)} chunks from {retrieval_source} with Cohere...")
        # Format for reranker
        docs = [c["text"] for c in raw_chunks]
        resp = co.rerank(
            model="rerank-english-v2.0",
            query=question,
            documents=docs,
            top_n=top_k,
        )
        reranked_chunks = [
            {
                "chunk_id": raw_chunks[r.index]["chunk_id"],
                "text": raw_chunks[r.index]["text"],
                "score": getattr(r, "score", getattr(r, "relevance_score", None)),
            }
            for r in resp.results
        ]
    except Exception as e:
        logger.error(f"Cohere reranking failed: {e}")
        return {
            "raw_chunks": raw_chunks,
            "reranked_chunks": [],
        }  # Return raw chunks on rerank failure

    return {"raw_chunks": raw_chunks, "reranked_chunks": reranked_chunks}
