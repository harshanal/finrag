# flake8: noqa E501,E231
"""Utility to convert a conversation turn into candidate chunks for retrieval."""

from typing import Dict, List


def build_candidate_chunks(turn: Dict) -> List[Dict]:
    """
    Convert a turn dict into a list of chunks with unique IDs.
    Each chunk is a dict: {"chunk_id": str, "text": str}.
    """
    candidate_chunks: List[Dict] = []
    # Pre-text lines
    for idx, line in enumerate(turn.get("pre_text", [])):
        candidate_chunks.append({"chunk_id": f"text:pre:{idx}", "text": line})
    # Table rows
    for idx, row in enumerate(turn.get("table", [])):
        text = " | ".join(row)
        candidate_chunks.append({"chunk_id": f"table:row:{idx}", "text": text})
    # Post-text lines
    for idx, line in enumerate(turn.get("post_text", [])):
        candidate_chunks.append({"chunk_id": f"text:post:{idx}", "text": line})
    return candidate_chunks

def chunk_text_content(text: str, min_chunk_len: int = 20) -> List[str]:
    """Chunks text content into paragraphs (splitting by double newline).

    Args:
        text: The input text string.
        min_chunk_len: Minimum character length for a chunk to be included.

    Returns:
        A list of text chunks.
    """
    chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
    # Filter out very short chunks
    return [chunk for chunk in chunks if len(chunk) >= min_chunk_len]
