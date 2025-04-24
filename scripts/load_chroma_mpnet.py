import chromadb
import json
import logging
import time
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys

# --- Add project root to sys.path ---
# This allows importing modules from the 'src' directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Import from src ---
try:
    # Assuming build_candidate_chunks is in finrag.chunk_utils
    from finrag.chunk_utils import build_candidate_chunks
except ImportError as e:
    print(f"Error importing FinRAG modules: {e}")
    print("Ensure the 'src' directory and necessary modules exist.")
    sys.exit(1)


# --- Configuration ---
SOURCE_DATA_FILE = os.path.join(project_root, "data", "train.json")
CHROMA_DB_PATH = os.path.join(project_root, "chroma_db_mpnet") # New DB path
COLLECTION_NAME = "finrag_embeddings_mpnet" # New collection name
MODEL_NAME = 'all-mpnet-base-v2'
BATCH_SIZE_EMBEDDING = 32 # Batch size for generating embeddings (adjust based on GPU memory)
BATCH_SIZE_CHROMA = 100  # Batch size for adding to Chroma
LOG_INTERVAL_CHUNKS = 5000 # Log progress every N chunks

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Loading Logic ---
def load_data_and_embed_to_chroma():
    # --- 1. Initialize Sentence Transformer Model ---
    logging.info(f"Loading Sentence Transformer model: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load Sentence Transformer model: {e}")
        return

    # --- 2. Initialize ChromaDB Client ---
    logging.info(f"Initializing ChromaDB client at: {CHROMA_DB_PATH}...")
    try:
        # Ensure the directory exists
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        logging.info("ChromaDB client initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize ChromaDB client: {e}")
        return

    # --- 3. Get or Create ChromaDB Collection ---
    logging.info(f"Getting or creating collection: {COLLECTION_NAME}")
    try:
        # Specify the embedding function based on the model loaded IF using Chroma's auto-embedding
        # Since we provide embeddings directly, this isn't strictly necessary for add(),
        # but good practice if you later query without providing query embeddings.
        # from chromadb.utils import embedding_functions
        # sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
        # collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=sentence_transformer_ef)

        # Simpler: Get/create without specifying EF if we always provide embeddings
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        logging.info(f"Collection '{COLLECTION_NAME}' ready. Initial count: {collection.count()}")
        # Optional: Consider deleting collection if it exists for a clean load
        # client.delete_collection(name=COLLECTION_NAME)
        # collection = client.create_collection(name=COLLECTION_NAME)
        # logging.info(f"Collection '{COLLECTION_NAME}' created fresh.")

    except Exception as e:
        logging.error(f"Failed to get or create Chroma collection '{COLLECTION_NAME}': {e}")
        return

    # --- 4. Load Source Data ---
    logging.info(f"Loading source data from: {SOURCE_DATA_FILE}")
    try:
        with open(SOURCE_DATA_FILE, 'r', encoding='utf-8') as f:
            # Load the entire file as a JSON array
            all_samples = json.load(f)
        logging.info(f"Loaded {len(all_samples)} samples from source file.")
    except FileNotFoundError:
        logging.error(f"Source data file not found: {SOURCE_DATA_FILE}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from source file: {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred loading source data: {e}")
        return

    # --- 5. Process Samples, Generate Chunks & Embeddings, Add to Chroma ---
    logging.info("Starting chunk generation, embedding, and loading to ChromaDB...")
    total_chunks_processed = 0
    start_time = time.time()

    # Prepare lists for batching ChromaDB adds
    batch_ids_chroma = []
    batch_embeddings_chroma = []
    batch_documents_chroma = []
    batch_metadatas_chroma = []

    # Iterate through each sample (document)
    for sample_idx, sample in enumerate(tqdm(all_samples, desc="Processing Samples")):
        if not isinstance(sample, dict):
            logging.warning(f"Skipping sample {sample_idx} as it's not a dictionary.")
            continue

        try:
            # Generate candidate chunks for this sample
            candidate_chunks = build_candidate_chunks(sample)
            if not candidate_chunks:
                logging.warning(f"No chunks generated for sample {sample_idx} (ID: {sample.get('id', 'N/A')}).")
                continue

            texts_to_embed = []
            chunk_ids_for_batch = []
            metadatas_for_batch = []

            # Prepare chunk texts and metadata for embedding batch
            for chunk_idx, chunk in enumerate(candidate_chunks):
                text = chunk.get("text")
                original_chunk_id = chunk.get("chunk_id") # Get the ID from build_candidate_chunks

                if not text or not original_chunk_id:
                    logging.warning(f"Skipping chunk {chunk_idx} in sample {sample_idx} due to missing text or original_chunk_id.")
                    continue
                
                # --- Create Globally Unique ID for Chroma --- 
                # Combine sample index and original chunk id to ensure uniqueness
                chroma_id = f"sample_{sample_idx}_chunk_{original_chunk_id}" 
                # Alternative using sample ID if available: 
                # sample_id_str = sample.get('id', f"index_{sample_idx}")
                # chroma_id = f"{sample_id_str}::{original_chunk_id}"
                # --- End ID Generation ---

                texts_to_embed.append(text)
                chunk_ids_for_batch.append(chroma_id) # Use the generated unique ID
                # Create metadata for ChromaDB
                metadata = {
                    "original_chunk_id": original_chunk_id, # Keep original ID in metadata
                    "source_filename": sample.get("filename", "unknown"), # Store original filename
                    "sample_id": sample.get("id", "unknown"), # Store original sample ID
                    "text_length": len(text)
                }
                metadatas_for_batch.append(metadata)

            # Generate embeddings for the current batch of texts
            if texts_to_embed:
                try:
                    embeddings = model.encode(texts_to_embed, batch_size=BATCH_SIZE_EMBEDDING, show_progress_bar=False)

                    # Add generated embeddings and corresponding data to Chroma batch lists
                    batch_ids_chroma.extend(chunk_ids_for_batch) # Use the generated unique IDs
                    batch_embeddings_chroma.extend(embeddings.tolist()) # Convert numpy arrays to lists
                    batch_documents_chroma.extend(texts_to_embed)
                    batch_metadatas_chroma.extend(metadatas_for_batch)

                    total_chunks_processed += len(texts_to_embed)

                    # Log progress periodically based on total chunks
                    if total_chunks_processed % LOG_INTERVAL_CHUNKS < BATCH_SIZE_EMBEDDING : # Approximate logging
                         elapsed_time = time.time() - start_time
                         logging.info(f"Processed approx {total_chunks_processed} chunks... ({elapsed_time:.2f}s elapsed)")

                except Exception as e:
                    logging.error(f"Error encoding batch for sample {sample_idx}: {e}")
                    # Decide how to handle embedding errors - skip batch?

            # Add batch to ChromaDB when it reaches BATCH_SIZE_CHROMA
            while len(batch_ids_chroma) >= BATCH_SIZE_CHROMA:
                try:
                    # Take the first BATCH_SIZE_CHROMA items
                    ids_to_add = batch_ids_chroma[:BATCH_SIZE_CHROMA]
                    embeddings_to_add = batch_embeddings_chroma[:BATCH_SIZE_CHROMA]
                    docs_to_add = batch_documents_chroma[:BATCH_SIZE_CHROMA]
                    meta_to_add = batch_metadatas_chroma[:BATCH_SIZE_CHROMA]

                    collection.add(
                        ids=ids_to_add,
                        embeddings=embeddings_to_add,
                        documents=docs_to_add,
                        metadatas=meta_to_add
                    )

                    # Remove the added items from the front of the lists
                    batch_ids_chroma = batch_ids_chroma[BATCH_SIZE_CHROMA:]
                    batch_embeddings_chroma = batch_embeddings_chroma[BATCH_SIZE_CHROMA:]
                    batch_documents_chroma = batch_documents_chroma[BATCH_SIZE_CHROMA:]
                    batch_metadatas_chroma = batch_metadatas_chroma[BATCH_SIZE_CHROMA:]

                except Exception as e:
                    logging.error(f"Error adding batch to Chroma: {e}. Skipping this batch.")
                    # Clear the batch to prevent retrying the same problematic data immediately
                    batch_ids_chroma = []
                    batch_embeddings_chroma = []
                    batch_documents_chroma = []
                    batch_metadatas_chroma = []
                    break # Exit the inner while loop for this sample

        except Exception as e:
            logging.error(f"Unexpected error processing sample {sample_idx} (ID: {sample.get('id', 'N/A')}): {e}")
            # Continue to the next sample

    # Add any remaining chunks in the last batch
    if batch_ids_chroma:
        logging.info(f"Adding final batch of {len(batch_ids_chroma)} chunks...")
        try:
            collection.add(
                ids=batch_ids_chroma,
                embeddings=batch_embeddings_chroma,
                documents=batch_documents_chroma,
                metadatas=batch_metadatas_chroma
            )
        except Exception as e:
            logging.error(f"Error adding final batch to Chroma: {e}")

    # --- 6. Final Report ---
    final_count = collection.count()
    total_time = time.time() - start_time
    logging.info("--- Loading Complete ---")
    logging.info(f"Total samples processed: {len(all_samples)}")
    logging.info(f"Total chunks processed and attempted to load: {total_chunks_processed}")
    logging.info(f"Final collection count in '{COLLECTION_NAME}': {final_count}")
    logging.info(f"Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    load_data_and_embed_to_chroma() 