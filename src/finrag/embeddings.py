"""Embedding utilities for FinRAG."""
import os
from typing import List, Optional

# Remove OpenAI and jsonlines imports
# import jsonlines
# import openai

# Import SentenceTransformer
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingStore:
    """Generate embeddings using a Sentence Transformer model."""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """Initializes the Sentence Transformer model."""
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        try:
            # Load the model during initialization
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Sentence Transformer model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model '{self.model_name}': {e}", exc_info=True)
            # You might want to raise the exception or handle it depending on desired behavior
            # For now, self.model remains None, and methods will return empty lists.

        # Remove cache logic
        # self.cache_path = os.path.join(os.getcwd(), "data", "embeddings.jsonl")
        # self._cache: Dict[str, List[float]] = {}
        # if os.path.exists(self.cache_path):
        #     with jsonlines.open(self.cache_path) as reader:
        #         for obj in reader:
        #             self._cache[obj["text"]] = obj["embedding"]

    def get_embedding(self, text: str) -> List[float]:
        """Generates an embedding for a single text string."""
        if not self.model:
            logger.error("Embedding model not loaded. Cannot generate embedding.")
            return [] # Return empty list or raise error
        
        if not text:
             logger.warning("Received empty string for embedding, returning empty list.")
             return []
             
        try:
            # model.encode() returns a numpy array, convert to list
            embedding = self.model.encode(text, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for text: '{text[:50]}...': {e}", exc_info=True)
            return []

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of text strings."""
        if not self.model:
            logger.error("Embedding model not loaded. Cannot generate embeddings.")
            return [] # Return empty list for all
        
        if not texts:
            return []

        # Filter out any empty strings before encoding
        valid_texts = [t for t in texts if t]
        if len(valid_texts) != len(texts):
             logger.warning(f"Removed {len(texts) - len(valid_texts)} empty strings before embedding.")
             if not valid_texts:
                  return [[] for _ in texts] # Return empty lists for all original texts if all were empty
        
        try:
            # Generate embeddings for the valid texts
            embeddings_np = self.model.encode(valid_texts, show_progress_bar=False)
            embeddings_list = embeddings_np.tolist()
            
            # Reconstruct the result list to match the original input list length,
            # inserting empty lists for originally empty strings.
            result_embeddings = []
            valid_idx = 0
            for original_text in texts:
                if original_text:
                    result_embeddings.append(embeddings_list[valid_idx])
                    valid_idx += 1
                else:
                    result_embeddings.append([])
            return result_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings for batch of size {len(valid_texts)}: {e}", exc_info=True)
            # Return empty list for all in case of batch error
            return [[] for _ in texts]
