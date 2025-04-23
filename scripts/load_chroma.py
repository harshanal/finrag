import chromadb
import json
import logging
import time
import os

# --- Configuration ---
# Build paths relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

EMBEDDINGS_FILE = os.path.join(project_root, "data", "embeddings.jsonl")
CHROMA_DB_PATH = os.path.join(project_root, "chroma_db") # Store DB in the root

COLLECTION_NAME = "finrag_embeddings"
BATCH_SIZE = 100
LOG_INTERVAL_BATCHES = 5 # Log progress every 5 batches

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Loading Logic ---
def load_embeddings_to_chroma():
    logging.info("Initializing ChromaDB client...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    except Exception as e:
        logging.error(f"Failed to initialize ChromaDB client at {CHROMA_DB_PATH}: {e}")
        return

    logging.info(f"Getting or creating collection: {COLLECTION_NAME}")
    try:
        # Note: If the collection exists with a different embedding function/metadata,
        # this might cause issues. For a clean load, delete the chroma_db directory first.
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        logging.info(f"Collection '{COLLECTION_NAME}' ready. Current count: {collection.count()}")
    except Exception as e:
        logging.error(f"Failed to get or create Chroma collection '{COLLECTION_NAME}': {e}")
        return

    logging.info(f"Starting embedding loading from: {EMBEDDINGS_FILE}")

    batch_ids = []
    batch_embeddings = []
    batch_documents = []
    batch_metadatas = [] # Optional, adjust if your JSONL has metadata

    processed_count = 0
    batch_count = 0
    start_time = time.time()

    try:
        with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    data = json.loads(line)
                    
                    # --- Generate IDs since they don't exist in the file ---
                    doc_id = f"chunk_{line_idx}"
                    embedding = data.get('embedding')
                    text_content = data.get('text')
                    # Create basic metadata from text content
                    metadata = {
                        "length": len(text_content) if text_content else 0,
                        "index": line_idx
                    }
                    # --- End Adapt ---

                    if not all([embedding, text_content]):
                        logging.warning(f"Skipping line {line_idx} due to missing embedding or text")
                        continue
                        
                    if not isinstance(embedding, list) or not all(isinstance(n, (int, float)) for n in embedding):
                        logging.warning(f"Skipping line {line_idx} due to invalid embedding format. Type: {type(embedding)}")
                        continue

                    batch_ids.append(str(doc_id))
                    batch_embeddings.append(embedding)
                    batch_documents.append(text_content)
                    batch_metadatas.append(metadata)

                    if len(batch_ids) >= BATCH_SIZE:
                        try:
                            collection.add(
                                ids=batch_ids,
                                embeddings=batch_embeddings,
                                documents=batch_documents,
                                metadatas=batch_metadatas
                            )
                            processed_count += len(batch_ids)
                            batch_count += 1
                            
                            if batch_count % LOG_INTERVAL_BATCHES == 0:
                                elapsed_time = time.time() - start_time
                                logging.info(f"Processed {processed_count} embeddings... ({elapsed_time:.2f}s elapsed)")
                                
                        except Exception as e:
                            logging.error(f"Error adding batch to Chroma: {e}")
                            # Optional: Add retry logic or skip the batch
                        
                        # Clear batches
                        batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []

                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line: {line.strip()}")
                except Exception as e:
                    logging.error(f"Unexpected error processing line {line_idx}: {e}")

        # Add any remaining embeddings in the last batch
        if batch_ids:
            try:
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                processed_count += len(batch_ids)
                logging.info(f"Added final batch of {len(batch_ids)} embeddings.")
            except Exception as e:
                logging.error(f"Error adding final batch to Chroma: {e}")

    except FileNotFoundError:
        logging.error(f"Embeddings file not found at: {EMBEDDINGS_FILE}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during file reading: {e}")

    final_count = collection.count()
    total_time = time.time() - start_time
    logging.info(f"--- Loading Complete ---")
    logging.info(f"Total embeddings processed: {processed_count}")
    logging.info(f"Final collection count: {final_count}")
    logging.info(f"Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    load_embeddings_to_chroma() 