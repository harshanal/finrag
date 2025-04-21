import os
from dotenv import load_dotenv
import json
import pinecone
from typing import List, Dict, Any
import logging # Import logging

# Ensure src is on path if running as script - adjust if needed
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from finrag.chunk_utils import build_candidate_chunks
    from finrag.embeddings import EmbeddingStore
except ImportError as e:
    print(f"Error importing FinRAG modules: {e}")
    print("Ensure the script is run from the project root or paths are correct.")
    sys.exit(1)


# Setup basic logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Pinecone Initialization ---
API_KEY = os.getenv("PINECONE_API_KEY")
ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set.")
if not ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT environment variable not set.")
if not INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME environment variable not set.")

# Initialize the Pinecone client
try:
    pc = pinecone.Pinecone(api_key=API_KEY, environment=ENVIRONMENT)
    print(f"Pinecone client initialized for environment: {ENVIRONMENT}")
except Exception as e:
    print(f"Error initializing Pinecone client: {e}")
    sys.exit(1)

# --- START: Configuration for Limiting Entries ---
# Set to a positive integer to limit entries per split, or None/0 to process all.
MAX_ENTRIES_PER_SPLIT = 100  # <-- Set this to your desired limit (e.g., 200) or None/0
# --- END: Configuration for Limiting Entries ---


# --- Check/Get Index ---
# Note: Index creation is often done separately or once.
# This script assumes the index exists. Add creation logic if needed.
try:
    print(f"Checking for index '{INDEX_NAME}'...")
    index_list = pc.list_indexes()
    if INDEX_NAME not in index_list.names():
         # Example: Create index if it doesn't exist (adjust spec as needed)
         # This requires knowing the embedding dimension beforehand
         # embedding_dim = EmbeddingStore().dimension # Get dimension if available
         # from pinecone import ServerlessSpec # or PodSpec
         # print(f"Index not found. Creating index '{INDEX_NAME}'...")
         # pc.create_index(
         #     name=INDEX_NAME,
         #     dimension=embedding_dim,
         #     metric="cosine", # Or your preferred metric
         #     spec=ServerlessSpec(cloud='aws', region='us-west-2') # Example spec
         # )
         # print("Index created. Wait a moment for initialization before upserting.")
         # import time
         # time.sleep(60) # Wait for index to be ready (adjust time as needed)
        raise ValueError(f"Pinecone index '{INDEX_NAME}' not found in environment '{ENVIRONMENT}'. Please create it first or add creation logic to the script.")

    index = pc.Index(INDEX_NAME)
    print(f"Connected to index '{INDEX_NAME}'.")
    # Optionally print index stats
    print(index.describe_index_stats())

except Exception as e:
    print(f"Error connecting to or checking Pinecone index '{INDEX_NAME}': {e}")
    sys.exit(1)
# --- End Check/Get Index ---


def index_chunks(split: str = "dev") -> None:
    """Index chunks from the specified data split into Pinecone."""
    data_path = os.path.join(os.getcwd(), "data", f"{split}.json")
    print(f"\nProcessing data file: {data_path}") # Added newline for clarity
    if not os.path.isfile(data_path):
        print(f"Data file not found: {data_path}. Skipping split.")
        return

    try:
        with open(data_path, "r", encoding='utf-8') as f: # Added encoding
            data = json.load(f)
    except Exception as e:
        print(f"Error loading or parsing {data_path}: {e}")
        return # Skip this split on error

    # Limit the data based on the configuration
    num_entries_total = len(data)
    effective_limit = MAX_ENTRIES_PER_SPLIT # Capture the configured limit

    if effective_limit is not None and effective_limit > 0:
        if num_entries_total > effective_limit:
             print(f"Limiting to first {effective_limit} entries out of {num_entries_total}.")
             data = data[:effective_limit]
        else:
             print(f"Processing all {num_entries_total} entries (limit >= total entries).")
    else:
        print(f"Processing all {num_entries_total} entries (limit is None or 0).")

    try:
        store = EmbeddingStore() # Initialize embedding store
    except Exception as e:
        print(f"Error initializing EmbeddingStore: {e}. Cannot proceed.")
        return

    batch: List[Dict[str, Any]] = []
    total_chunks_indexed_split = 0
    entries_processed_count = 0 # Counter for processed entries in this split
    batch_size = 100 # Pinecone recommends batches <= 100

    print(f"Starting chunk generation and embedding for {len(data)} entries in split '{split}'...")
    for i, sample in enumerate(data): # Use enumerate for progress tracking
        entries_processed_count += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(data): # Print progress every 50 entries and at the end
             print(f"  Processing entry {i + 1}/{len(data)} for split '{split}'...")

        if not isinstance(sample, dict):
            logger.warning(f"Skipping entry {i+1} in split '{split}' as it is not a dictionary.")
            continue

        try:
            chunks = build_candidate_chunks(sample)
            for c in chunks:
                # Ensure text is valid before embedding
                text_to_embed = c.get("text", "")
                chunk_id = c.get("chunk_id")

                if not isinstance(text_to_embed, str) or not text_to_embed:
                    logger.warning(f"Skipping chunk with invalid/empty text: ID {chunk_id} in split '{split}'")
                    continue
                if not chunk_id:
                     logger.warning(f"Skipping chunk with missing ID in entry {i+1} for split '{split}'")
                     continue

                # Generate embedding
                try:
                     embed = store.get_embedding(text_to_embed)
                except Exception as emb_err:
                     logger.error(f"Error getting embedding for chunk {chunk_id} in split '{split}': {emb_err}")
                     continue # Skip this chunk if embedding fails

                # Prepare metadata - keep it minimal if possible
                metadata = {
                    "doc_id": c.get("doc_id", ""),
                    # "source": c.get("source", ""), # Filename/source often part of doc_id
                    "text": text_to_embed[:200] + "..." if len(text_to_embed) > 200 else text_to_embed, # Store truncated text for quick checks? Limit size.
                }

                batch.append({"id": chunk_id, "values": embed, "metadata": metadata})
                total_chunks_indexed_split += 1

                # Upsert in batches
                if len(batch) >= batch_size:
                    print(f"    Upserting batch of {len(batch)} vectors...")
                    try:
                        index.upsert(vectors=batch)
                    except Exception as upsert_err:
                         logger.error(f"Error upserting batch to Pinecone: {upsert_err}")
                         # Optional: Add retry logic or handle partial failures
                    batch = [] # Reset batch

        except Exception as entry_err:
            logger.error(f"Error processing entry {i+1} (ID: {sample.get('id', 'N/A')}) in split '{split}': {entry_err}", exc_info=True)
            # Decide if you want to continue or stop on error
            # continue

    # Upsert any remaining vectors in the last batch
    if batch:
        print(f"    Upserting final batch of {len(batch)} vectors...")
        try:
            index.upsert(vectors=batch)
        except Exception as upsert_err:
            logger.error(f"Error upserting final batch to Pinecone: {upsert_err}")

    print(f"Indexed {total_chunks_indexed_split} total chunks from {entries_processed_count} entries for split '{split}' into Pinecone index '{INDEX_NAME}'.")


if __name__ == "__main__":
    # Decide which splits to index - MODIFY THIS LIST AS NEEDED
    # splits_to_index = ["dev"] # Good for quick testing
    splits_to_index = ["train"] # Focus on train as requested
    # splits_to_index = ["dev", "train", "test"] # To index all relevant splits

    print("-" * 50)
    print(f"Starting indexing run")
    print(f"Target Pinecone Index: {INDEX_NAME}")
    print(f"Splits to process: {splits_to_index}")
    if MAX_ENTRIES_PER_SPLIT is not None and MAX_ENTRIES_PER_SPLIT > 0:
        print(f"Processing up to {MAX_ENTRIES_PER_SPLIT} entries per split.")
    else:
        print("Processing all entries per split.")
    print("-" * 50)


    for split in splits_to_index:
        index_chunks(split)

    print("\nIndexing process complete.")
    print("-" * 50)