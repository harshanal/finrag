# scripts/upsert_to_pinecone.py
import argparse
import json
import logging
import os
import sys
import uuid

from dotenv import find_dotenv, load_dotenv
from pinecone import Pinecone
from tqdm import tqdm

# Add src directory to path to import finrag modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

# Import necessary modules from finrag (adjust paths/names if needed)
try:
    # Import logging_conf just to ensure basicConfig is called
    import finrag.logging_conf
    from finrag.data_loaders import (
        load_files_from_directory,  # Corrected data_loader import
    )
    from finrag.embeddings import EmbeddingStore
    from finrag.utils import clean_env  # Corrected clean_env import
except ImportError as e:
    print(f"Error importing FinRAG modules: {e}")
    print("Please ensure the src directory is in the Python path and necessary modules exist.")
    sys.exit(1)

# Get a logger instance for this script
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_DIR = os.path.join(project_root, "data")  # Assuming data is in project_root/data
PINECONE_UPSERT_BATCH_SIZE = 100
# Add any chunking parameters if needed, e.g., CHUNK_SIZE = 1000

# --------- CLI args for record limit and question log ---------
parser = argparse.ArgumentParser(
    description="Pinecone upsert with record limit and question logging"
)
parser.add_argument(
    "--max-records", type=int, default=500, help="Maximum number of chunks to upsert"
)
parser.add_argument(
    "--question-log",
    type=str,
    default=os.path.join(project_root, "upsert_questions.jsonl"),
    help="Path to write sample_id and question log",
)
args = parser.parse_args()
MAX_RECORDS = args.max_records
QUESTION_LOG_PATH = args.question_log


def upsert_data_to_pinecone():
    """Loads data, chunks it, gets embeddings, and upserts to Pinecone."""
    logger.info("Starting Pinecone upsert process...")

    # --- Load Environment Variables ---
    load_dotenv(find_dotenv(), override=True)
    api_key = clean_env("PINECONE_API_KEY")
    index_name = clean_env("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        logger.error("PINECONE_API_KEY or PINECONE_INDEX_NAME not found in environment variables.")
        return

    logger.info(f"Target Pinecone index: {index_name}")

    # --- Initialize Pinecone ---
    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        logger.info("Pinecone initialized successfully.")
        stats = index.describe_index_stats()
        logger.info(f"Initial index stats: {stats}")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone or connect to index: {e}")
        return

    # --- Initialize Embedding Store ---
    try:
        embed_store = EmbeddingStore()
        logger.info("EmbeddingStore initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize EmbeddingStore: {e}")
        return

    # --- Load and Process Files ---
    logger.info(f"Loading files from directory: {DATA_DIR}")
    try:
        all_chunks = load_files_from_directory(DATA_DIR)
        if not all_chunks:
            logger.warning(f"No chunks found in {DATA_DIR}.")
            return
        logger.info(f"Found {len(all_chunks)} chunks to process.")
    except Exception as e:
        logger.error(f"Failed to load chunks from {DATA_DIR}: {e}")
        return

    # Apply record limit
    if MAX_RECORDS is not None:
        all_chunks = all_chunks[:MAX_RECORDS]
        logger.info(f"Limiting to first {len(all_chunks)} chunks (max_records={MAX_RECORDS}).")

    # Open question log file
    try:
        qlog = open(QUESTION_LOG_PATH, "w", encoding="utf-8")
        logger.info(f"Question log file: {QUESTION_LOG_PATH}")
    except Exception as e:
        logger.error(f"Cannot open question log file {QUESTION_LOG_PATH}: {e}")
        qlog = None

    vectors_to_upsert = []

    # Iterate through the pre-processed chunks
    for chunk_data in tqdm(all_chunks, desc="Processing chunks"):
        # Create unique upsert ID to avoid overwriting existing vectors
        base_id = chunk_data.get("chunk_id", "")
        # Append short UUID suffix
        suffix = uuid.uuid4().hex[:8]
        chunk_id = f"{base_id}-{suffix}" if base_id else suffix

        # Log sample id and question
        if qlog:
            sample_id = chunk_data.get("sample_id", "")
            question = chunk_data.get("question", "")
            qlog.write(json.dumps({"sample_id": sample_id, "question": question}) + "\n")

        chunk_text = chunk_data.get("text", "")
        source_filename = chunk_data.get("source_filename", "unknown_source")

        if not chunk_text:
            logger.warning(f"Skipping empty content for chunk: {chunk_id}")
            continue

        # Embed the chunk text
        try:
            embedding = embed_store.get_embeddings([chunk_text])[
                0
            ]  # Get embedding for the single chunk
        except Exception as e:
            logger.error(f"Failed to get embedding for chunk {chunk_id}: {e}")
            continue

        # Prepare metadata (use text and source_filename from the chunk data)
        metadata = {"text": chunk_text, "source": source_filename}

        vectors_to_upsert.append((chunk_id, embedding, metadata))

        # Upsert in batches
        if len(vectors_to_upsert) >= PINECONE_UPSERT_BATCH_SIZE:
            logger.info(f"Upserting batch of {len(vectors_to_upsert)} vectors...")
            try:
                index.upsert(vectors=vectors_to_upsert)
                logger.info(f"Successfully upserted batch.")
            except Exception as e:
                logger.error(f"Error during batch upsert: {e}")
            vectors_to_upsert = []  # Reset batch

    # Upsert any remaining vectors
    if vectors_to_upsert:
        logger.info(f"Upserting final batch of {len(vectors_to_upsert)} vectors...")
        try:
            index.upsert(vectors=vectors_to_upsert)
            logger.info(f"Successfully upserted final batch.")
        except Exception as e:
            logger.error(f"Error during final batch upsert: {e}")

    logger.info(f"Processed {len(all_chunks)} chunks.")
    try:
        final_stats = index.describe_index_stats()
        logger.info(f"Final index stats: {final_stats}")
    except Exception as e:
        logger.error(f"Failed to get final index stats: {e}")

    # Close question log
    if qlog:
        qlog.close()

    logger.info("Pinecone upsert process finished.")


if __name__ == "__main__":
    upsert_data_to_pinecone()
