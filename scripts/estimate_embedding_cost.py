import json
import os
import logging
from typing import Dict, List, Any

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken library not found.")
    print("Please install it using: pip install tiktoken")
    exit()

# --- Copied from src/finrag/chunk_utils.py (for self-containment) ---
# Setup basic logging for the copied functions
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("ChunkUtilsEstimate")

def format_table_as_markdown(table_data: List[List[str]]) -> str:
    """Converts a list-of-lists table into a Markdown formatted string."""
    if not table_data or not isinstance(table_data, list) or not all(isinstance(row, list) for row in table_data):
        logger.warning("Invalid table_data format received for Markdown conversion. Skipping table.")
        return ""
    try:
        # Escape pipe characters within cells to avoid breaking Markdown table structure
        escaped_table = [[str(cell).replace("|", "\\\\") for cell in row] for row in table_data]

        header = escaped_table[0]
        # Basic separator length, handle potential non-string headers gracefully
        separator = ["-" * len(str(h).strip()) if isinstance(h, (str, int, float)) else "-" * 3 for h in header]
        body = escaped_table[1:]

        markdown = "| " + " | ".join(map(str, header)) + " |\\n" # Ensure headers are strings
        markdown += "| " + " | ".join(separator) + " |\\n"
        for row in body:
            # Ensure row has the same number of columns as header, pad if necessary
            padded_row = row + [""] * (len(header) - len(row)) if len(row) < len(header) else row[:len(header)]
            # Ensure all cells in padded_row are strings before joining
            markdown += "| " + " | ".join(map(str, padded_row)) + " |\\n"
        return markdown.strip()
    except Exception as e:
        logger.error(f"Error formatting table to Markdown: {e}", exc_info=True)
        return "" # Return empty string on error

def build_candidate_chunks(turn: Dict) -> List[Dict]:
    """
    Convert a turn dict into a list of chunks with unique IDs.
    Tables are converted into a single Markdown chunk.
    Each chunk is a dict: {"chunk_id": str, "text": str}.
    (Adapted for estimation script)
    """
    candidate_chunks: List[Dict] = []
    doc_id = turn.get("filename", "unknown_doc") # Get document ID for context

    # Pre-text lines
    for idx, line in enumerate(turn.get("pre_text", [])):
        if isinstance(line, str): # Basic check
             candidate_chunks.append({"chunk_id": f"{doc_id}::text:pre:{idx}", "text": line})

    # Table processing - Use table_ori if available and looks cleaner, else fallback to table
    table_content = turn.get("table_ori", turn.get("table", []))
    if table_content:
        md_table_text = format_table_as_markdown(table_content)
        if md_table_text: # Only add if Markdown conversion was successful
            candidate_chunks.append({"chunk_id": f"{doc_id}::table:full", "text": md_table_text})
        else:
             logger.warning(f"Markdown table generation failed for doc: {doc_id}")

    # Post-text lines
    for idx, line in enumerate(turn.get("post_text", [])):
         if isinstance(line, str): # Basic check
            candidate_chunks.append({"chunk_id": f"{doc_id}::text:post:{idx}", "text": line})

    # Add doc_id to all chunks for potential metadata filtering
    for chunk in candidate_chunks:
        chunk['doc_id'] = doc_id

    return candidate_chunks
# --- End of copied functions ---

def estimate_tokens_and_chunks(data_path: str, tokenizer_name: str = "cl100k_base") -> Dict[str, int]:
    """Loads data, generates chunks, and counts tokens."""
    total_entries = 0
    total_chunks = 0
    total_tokens = 0

    try:
        # Select the tokenizer based on the model you intend to use
        # cl100k_base is common for text-embedding-3-small, text-embedding-3-large, gpt-4, gpt-3.5-turbo
        # p50k_base might be used for older models like text-embedding-ada-002
        encoding = tiktoken.get_encoding(tokenizer_name)
        print(f"Using tokenizer: {tokenizer_name}")
    except Exception as e:
        print(f"Error getting tokenizer '{tokenizer_name}': {e}")
        print("Please ensure the tokenizer name is correct and tiktoken is installed.")
        return {}

    print(f"Processing file: {data_path}...")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {data_path}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred loading the file: {e}")
        return {}

    num_entries = len(data)
    print(f"Found {num_entries} entries in the JSON file.")

    for i, entry in enumerate(data):
        total_entries += 1
        if (i + 1) % 100 == 0: # Print progress every 100 entries
             print(f"Processing entry {i + 1}/{num_entries}...")

        if not isinstance(entry, dict):
            logger.warning(f"Skipping entry {i+1} as it is not a dictionary.")
            continue

        try:
            chunks = build_candidate_chunks(entry)
            for chunk in chunks:
                total_chunks += 1
                text_to_encode = chunk.get("text", "")
                if isinstance(text_to_encode, str):
                    tokens = encoding.encode(text_to_encode)
                    total_tokens += len(tokens)
                else:
                    logger.warning(f"Chunk text is not a string in entry {i+1}, chunk_id: {chunk.get('chunk_id', 'N/A')}. Skipping token count for this chunk.")
        except Exception as e:
            logger.error(f"Error processing entry {i+1} (ID: {entry.get('id', 'N/A')}): {e}", exc_info=True)


    print("Processing complete.")
    return {
        "total_entries": total_entries,
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
    }

if __name__ == "__main__":
    # --- Configuration ---
    DATA_FILE = os.path.join("data", "train.json")
    # IMPORTANT: Choose the tokenizer corresponding to your embedding model
    # e.g., "cl100k_base" for text-embedding-3 models, "p50k_base" for text-embedding-ada-002
    TOKENIZER = "cl100k_base"
    # --- End Configuration ---

    metrics = estimate_tokens_and_chunks(DATA_FILE, TOKENIZER)

    if metrics:
        print("\n--- Embedding Cost Estimation Metrics ---")
        print(f"Total Entries Processed: {metrics.get('total_entries'):,}")
        print(f"Total Chunks Generated:  {metrics.get('total_chunks'):,}")
        print(f"Total Tokens Estimated:  {metrics.get('total_tokens'):,}")
        print("-----------------------------------------")
        print("\nTo estimate cost:")
        print("1. Find the pricing for your chosen embedding model (e.g., OpenAI pricing page).")
        print("   Look for the cost per 1,000 or 1,000,000 tokens.")
        print("2. Calculate: (Total Tokens Estimated / Tokens per Price Unit) * Price per Unit")
        print(f"   Example: If price is $0.0001 per 1k tokens: ({metrics.get('total_tokens'):,} / 1000) * $0.0001")
    else:
        print("\nMetrics calculation failed.") 