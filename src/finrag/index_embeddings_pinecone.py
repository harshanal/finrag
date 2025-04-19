import os
from dotenv import load_dotenv
import json
import pinecone
from typing import List, Dict, Any
from finrag.chunk_utils import build_candidate_chunks
from finrag.embeddings import EmbeddingStore

# Load environment variables
load_dotenv()
# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pinecone.Index(index_name)

def index_chunks(split: str = "dev") -> None:
    """Index chunks from the specified data split into Pinecone."""
    data_path = os.path.join(os.getcwd(), "data", f"{split}_turn.json")
    if not os.path.isfile(data_path):
        print(f"Data file not found: {data_path}")
        return
    with open(data_path, "r") as f:
        data = json.load(f)

    store = EmbeddingStore()
    batch: List[Dict[str, Any]] = []
    count = 0
    for sample in data:
        chunks = build_candidate_chunks(sample)
        for c in chunks:
            metadata = {
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id", ""),
                "source": c.get("source", ""),
                "text": c.get("text", ""),
            }
            embed = store.get_embedding(c.get("text", ""))
            batch.append({"id": c.get("chunk_id"), "values": embed, "metadata": metadata})
            count += 1
            if len(batch) >= 100:
                index.upsert(vectors=batch)
                batch = []
    if batch:
        index.upsert(vectors=batch)
    print(f"Indexed {count} chunks for split '{split}' into Pinecone index '{index_name}'.")

if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        index_chunks(split)
