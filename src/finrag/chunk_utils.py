"""Utility to convert a conversation turn into candidate chunks for retrieval."""

from typing import List, Dict


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
