"""Retrieval module for FinRAG."""

from typing import List


def retrieve_evidence(
    doc_id: str, question: str, top_k_text: int = 4, top_k_table: int = 4
) -> List[str]:
    """Retrieve evidence chunk IDs for a question."""
    raise NotImplementedError
