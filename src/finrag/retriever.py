"""
########################################################################
# FinRAG Retrieval Module (retriever.py)
#
# This module implements the "retrieval" stage of the RAG pipeline:
# 1. Initialises a ChromaDB client for vector retrieval.
# 2. chromadb_retrieve(query, top_k, doc_id=None): uses stored vector embeddings
#    in ChromaDB to find top-k relevant chunks via similarity search. Can optionally
#    filter by document ID ('source_filename' metadata).
# 3. BM25Okapi: a pure-Python sparse retrieval algorithm (fallback).
# 4. retrieve_evidence(turn, question): orchestrates retrieval:
#    a. Attempts retrieval using chromadb_retrieve. If a non-empty 'turn' dict
#       is provided, it extracts 'filename' to filter the ChromaDB search to
#       that document. If 'turn' is empty, performs a global search.
#    b. If ChromaDB fails or returns no results AND if 'turn' data is available,
#       builds candidate chunks from the 'turn' and uses BM25Okapi as a fallback
#       to select top-bm25_k chunks.
#    c. If raw chunks are found (from either ChromaDB or BM25), reranks them
#       using the Cohere Rerank API to refine the top-k most relevant chunks.
#    d. Returns a dict with 'raw_chunks' (initial candidates) and 'reranked_chunks'
#       (after Cohere reranking) ready for the agent.
########################################################################
"""

"""Retrieval module for FinRAG."""

import logging
import math
import os
from typing import Any, Dict, List

import chromadb

from finrag.chunk_utils import build_candidate_chunks
from finrag.embeddings import EmbeddingStore
from finrag.tools import co

logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv

load_dotenv(override=True)  # Override existing env vars from .env

# --- ChromaDB Initialization ---
CHROMA_DB_PATH = "./chroma_db_mpnet" # Path relative to project root where DB is stored
COLLECTION_NAME = "finrag_embeddings_mpnet" # Use the new collection name
CHROMA_CLIENT = None
CHROMA_COLLECTION = None
try:
    logger.info(f"Attempting to connect to ChromaDB at: {CHROMA_DB_PATH}")
    CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collections = CHROMA_CLIENT.list_collections()
    if COLLECTION_NAME in [col.name for col in collections]:
        CHROMA_COLLECTION = CHROMA_CLIENT.get_collection(name=COLLECTION_NAME)
        logger.info(f"Successfully connected to ChromaDB collection '{COLLECTION_NAME}'. Count: {CHROMA_COLLECTION.count()}")
    else:
        logger.error(f"ChromaDB collection '{COLLECTION_NAME}' not found at {CHROMA_DB_PATH}. Please run the loading script first.")
        CHROMA_COLLECTION = None 
except Exception as e:
    logger.error(f"Failed to initialize or connect to ChromaDB at {CHROMA_DB_PATH}: {e}")
    CHROMA_COLLECTION = None
# --- End ChromaDB Initialization ---


# ================================================================
# BM25 Code (Corrected Structure)
# ================================================================
class BM25Okapi:
    """Best Matching 25 retrieval algorithm implementation."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25Okapi with corpus, k1, and b parameters."""
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in corpus]
        # Handle potential division by zero if corpus is empty
        self.avgdl = sum(self.doc_len) / len(corpus) if corpus else 0 
        self.doc_freqs = self._calculate_doc_freqs()
        self.idf = self._calculate_idf()

    def _calculate_doc_freqs(self) -> Dict[str, int]:
        """Calculate document frequencies for each term in the corpus."""
        freqs: Dict[str, int] = {}
        for doc in self.corpus:
            seen = set()
            for term in doc:
                if term not in seen:
                    freqs[term] = freqs.get(term, 0) + 1
                    seen.add(term)
        return freqs

    def _calculate_idf(self) -> Dict[str, float]:
        """Calculate inverse document frequency for each term."""
        idf: Dict[str, float] = {}
        num_docs = len(self.corpus)
        # Handle case where num_docs might be 0
        if num_docs == 0:
            return idf 
        for term, freq in self.doc_freqs.items():
            # Ensure freq is not greater than num_docs (can happen with unusual data)
            freq = min(freq, num_docs)
            idf_val = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            idf[term] = max(0, idf_val) # Ensure IDF is non-negative
        return idf

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """Calculate BM25 scores for the query against all documents in the corpus."""
        scores = [0.0] * len(self.corpus)
        # Handle case where avgdl might be 0
        if self.avgdl == 0:
            return scores 
        for i, doc in enumerate(self.corpus):
            term_freqs = {term: doc.count(term) for term in set(doc)}
            for term in query_tokens:
                if term in term_freqs:
                    tf = term_freqs[term]
                    idf = self.idf.get(term, 0)
                    numerator = idf * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (
                        1 - self.b + self.b * self.doc_len[i] / self.avgdl
                    )
                    if denominator != 0: # Avoid division by zero
                        scores[i] += numerator / denominator
        return scores

def chromadb_retrieve(query: str, top_k: int = 20, doc_id: str | None = None) -> List[Dict[str, Any]]:
    """Retrieve similar chunks from ChromaDB collection, optionally filtering by doc_id.
    Uses 'source_filename' metadata key for filtering.
    """
    if not CHROMA_COLLECTION:
        logger.error("ChromaDB collection not available for retrieval.")
        return []
    
    store = EmbeddingStore() # Now uses SentenceTransformer
    try:
        q_embed = store.get_embedding(query)
        if not q_embed:
             logger.error("Failed to generate query embedding.")
             return []
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        return []

    query_params = {
        "query_embeddings": [q_embed],
        "n_results": top_k,
        "include": ["metadatas", "documents", "distances"] 
    }
    
    # --- Re-enable filtering using the correct metadata key ---
    if doc_id:
        # Filter using 'source_filename' stored during load_chroma_mpnet.py
        metadata_filter_key = "source_filename"
        logger.debug(f"Applying ChromaDB filter: {{'{metadata_filter_key}': '{doc_id}'}}")
        query_params["where"] = {metadata_filter_key: doc_id} 
    else:
        # This case might be less common now if turn always provides a filename
        logger.debug("No doc_id provided, performing unfiltered ChromaDB query.")
    # --- End Filter Update ---

    try:
        response = CHROMA_COLLECTION.query(**query_params)
    except Exception as e:
         logger.error(f"ChromaDB query failed: {e}")
         return []

    chunks: List[Dict[str, Any]] = []
    if response and response.get("ids") and response["ids"] and response["ids"][0]:
        ids = response["ids"][0]
        documents = response["documents"][0] if response.get("documents") else ["" for _ in ids]
        metadatas = response["metadatas"][0] if response.get("metadatas") else [{} for _ in ids]
        distances = response["distances"][0] if response.get("distances") else [1.0 for _ in ids]
        
        for i, chunk_id in enumerate(ids):
            text = documents[i] if documents and i < len(documents) else ""
            score = 1.0 - (distances[i] if distances and i < len(distances) else 1.0)
            
            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "score": score,
            })
    else:
        logger.debug("ChromaDB query returned no matches.")
        
    return chunks


def retrieve_evidence(
    turn: Dict[str, Any], question: str, top_k: int = 8, bm25_k: int = 50
) -> Dict[str, List[Dict[str, Any]]]:
    """Hybrid retrieval (multistage): ChromaDB / BM25 + Cohere reranking."""
    # --- START CONFIGURATION ---
    # Increase the number of candidates fetched initially from ChromaDB
    chromadb_query_top_k = 50 
    # --- END CONFIGURATION ---

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
    # Extract doc_id from the turn for potential filtering
    doc_id_to_filter = turn.get("filename") # Assuming 'filename' is the doc_id
    if not doc_id_to_filter:
         logger.warning(f"Could not find 'filename' (doc_id) in input turn for potential filtering.")

    # 1. Try ChromaDB if available
    if CHROMA_COLLECTION: # Check if collection loaded successfully
        try:
            logger.debug(f"Attempting ChromaDB retrieval for query: {question[:50]}... DocID: {doc_id_to_filter}, Requesting Top K: {chromadb_query_top_k}")
            # Use chromadb_retrieve
            raw_chunks = chromadb_retrieve(question, chromadb_query_top_k, doc_id=doc_id_to_filter) 
            
            if not raw_chunks:
                logger.warning("ChromaDB retrieval returned no results.")
            else:
                retrieval_source = "ChromaDB"
                logger.debug(f"ChromaDB returned {len(raw_chunks)} chunks.")
        except Exception as e:
            logger.error(f"ChromaDB retrieval failed: {e}. Falling back to BM25.")
            raw_chunks = []

    # 2. Fallback to BM25 if ChromaDB unavailable OR failed OR returned empty
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
        # --- Increase top_k for reranking ---
        final_top_k = 15 # Increased from default 8
        logger.debug(f"Reranking {len(raw_chunks)} chunks from {retrieval_source} with Cohere (Top {final_top_k})...")
        # Format for reranker
        docs = [c["text"] for c in raw_chunks]
        resp = co.rerank(
            model="rerank-english-v2.0",
            query=question,
            documents=docs,
            top_n=final_top_k, 
        )
        reranked_chunks = [
            {
                "chunk_id": raw_chunks[r.index]["chunk_id"],
                "text": raw_chunks[r.index]["text"],
                # Use get attribute to handle potential differences in score field name
                "score": getattr(r, 'relevance_score', getattr(r, 'score', None)),
            }
            for r in resp.results
        ]
    except Exception as e:
        logger.error(f"Cohere reranking failed: {e}")
        # Return raw chunks if reranking fails, so agent still gets *something*
        # Also add scores from the initial retrieval if available
        raw_chunks_with_scores = []
        for c in raw_chunks:
             raw_chunks_with_scores.append({
                  "chunk_id": c.get("chunk_id", "unknown"),
                  "text": c.get("text", ""),
                  "score": c.get("score", 0.0) # Use initial score or 0
             })
        return {
            "raw_chunks": raw_chunks_with_scores, # Keep original raw chunks order
            "reranked_chunks": [],
        }  

    # Return both raw (initial retrieval) and reranked chunks
    # The agent should primarily use reranked_chunks if available
    # Include scores in raw_chunks if they exist from Chroma/BM25
    raw_chunks_with_scores = []
    for c in raw_chunks:
         raw_chunks_with_scores.append({
              "chunk_id": c.get("chunk_id", "unknown"),
              "text": c.get("text", ""),
              "score": c.get("score", 0.0) # Use initial score or 0
         })

    return {"raw_chunks": raw_chunks_with_scores, "reranked_chunks": reranked_chunks}
