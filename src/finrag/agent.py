"""Agent module for FinRAG."""

from typing import List, Dict


def answer(question: str, conversation_history: List[Dict], doc_id: str) -> Dict:
    """Answer a question given history and document ID."""
    raise NotImplementedError
