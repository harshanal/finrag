"""Embedding utilities for FinRAG."""
import os
from typing import Dict, List

import jsonlines
import openai
from finrag.utils import clean_env


class EmbeddingStore:
    """Cache and retrieve text embeddings using OpenAI."""

    def __init__(self, model: str = None):
        """
        Initialize the embedding store with a model name. If model is not provided,
        fallback to EMBEDDING_MODEL from environment variables.
        """
        self.model = model or clean_env("EMBEDDING_MODEL")
        self.cache_path = os.path.join(os.getcwd(), "data", "embeddings.jsonl")
        self._cache: Dict[str, List[float]] = {}
        if os.path.exists(self.cache_path):
            with jsonlines.open(self.cache_path) as reader:
                for obj in reader:
                    self._cache[obj["text"]] = obj["embedding"]

    def get_embedding(self, text: str) -> List[float]:
        if text not in self._cache:
            # migrate to OpenAI Python v1: use embeddings endpoint
            resp = openai.Embedding.create(model=self.model, input=[text])
            embed = resp.data[0].embedding
            self._cache[text] = embed
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with jsonlines.open(self.cache_path, mode="a") as writer:
                writer.write({"text": text, "embedding": embed})
        return self._cache[text]

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # determine which texts aren't cached yet
        missing = [t for t in texts if t not in self._cache]
        if missing:
            # batch‐fetch missing embeddings
            resp = openai.Embedding.create(model=self.model, input=missing)
            for t, d in zip(missing, resp.data):
                emb = d.embedding
                self._cache[t] = emb
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                with jsonlines.open(self.cache_path, mode="a") as w:
                    w.write({"text": t, "embedding": emb})
        # return embeddings in original order
        return [self._cache[t] for t in texts]
