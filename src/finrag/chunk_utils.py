# flake8: noqa E501,E231
"""Utility to convert a conversation turn into candidate chunks for retrieval."""

from typing import Dict, List
import logging

# Setup basic logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def format_table_as_markdown(table_data: List[List[str]]) -> str:
    """Converts a list-of-lists table into a Markdown formatted string."""
    if not table_data or not isinstance(table_data, list) or not all(isinstance(row, list) for row in table_data):
        # logger.warning("Invalid table_data format received for Markdown conversion. Skipping table.")
        return "" # Return empty string for invalid input, suppress log for cleaner output if needed
    try:
        # Escape pipe characters within cells to avoid breaking Markdown table structure
        # Need double escape for literal backslash in final string
        escaped_table = [[str(cell).replace("|", "\\|") for cell in row] for row in table_data]

        header = escaped_table[0]
        # Basic separator length, handle potential non-string headers gracefully
        separator = ["-" * len(str(h).strip()) if isinstance(h, (str, int, float)) else "-" * 3 for h in header]
        body = escaped_table[1:]

        markdown = "| " + " | ".join(map(str, header)) + " |\n" # Ensure headers are strings
        markdown += "| " + " | ".join(separator) + " |\n"
        for row in body:
            # Ensure row has the same number of columns as header, pad if necessary
            padded_row = row + [""] * (len(header) - len(row)) if len(row) < len(header) else row[:len(header)]
            # Ensure all cells in padded_row are strings before joining
            markdown += "| " + " | ".join(map(str, padded_row)) + " |\n"
        return markdown.strip()
    except Exception as e:
        logger.error(f"Error formatting table to Markdown: {e}", exc_info=True)
        return "" # Return empty string on error

def build_candidate_chunks(turn: Dict) -> List[Dict]:
    """
    Convert a turn dict into a list of chunks with unique IDs.
    Tables are converted into a single Markdown chunk.
    Each chunk is a dict: {"chunk_id": str, "text": str}.
    """
    candidate_chunks: List[Dict] = []
    doc_id = turn.get("filename", "unknown_doc") # Get document ID for context

    # Pre-text lines
    for idx, line in enumerate(turn.get("pre_text", [])):
         if isinstance(line, str): # Basic check
            candidate_chunks.append({"chunk_id": f"{doc_id}::text:pre:{idx}", "text": line}) # Add doc_id prefix

    # Table processing - Use table_ori if available and looks cleaner, else fallback to table
    # Prioritize 'table_ori' as it might be less processed/noisy
    table_content = turn.get("table_ori", turn.get("table", []))
    if table_content:
        md_table_text = format_table_as_markdown(table_content)
        if md_table_text: # Only add if Markdown conversion was successful
            # --- START MODIFICATION: Add Title to Table Chunk --- 
            # Prepend a generic title to give more context for embedding/LLM
            chunk_text_with_title = f"Financial Table Data:\n\n{md_table_text}"
            candidate_chunks.append({"chunk_id": f"{doc_id}::table:full", "text": chunk_text_with_title}) # Add doc_id prefix
            # --- END MODIFICATION ---
        else:
             logger.warning(f"Markdown table generation failed for doc: {doc_id}")
    # OLD TABLE LOGIC (REMOVE/COMMENT OUT)
    # for idx, row in enumerate(turn.get("table", [])):
    #     text = " | ".join(row)
    #     candidate_chunks.append({"chunk_id": f"table:row:{idx}", "text": text})

    # Post-text lines
    for idx, line in enumerate(turn.get("post_text", [])):
         if isinstance(line, str): # Basic check
            candidate_chunks.append({"chunk_id": f"{doc_id}::text:post:{idx}", "text": line}) # Add doc_id prefix

    # Add doc_id to all chunks for potential metadata filtering
    for chunk in candidate_chunks:
        chunk['doc_id'] = doc_id

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
