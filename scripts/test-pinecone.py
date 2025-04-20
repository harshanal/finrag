import os
import pinecone
from dotenv import load_dotenv, find_dotenv

import unicodedata
import re

def clean_env(key):
    val = os.getenv(key, "")
    # Normalize to ASCII
    val = unicodedata.normalize("NFKC", val)
    # Replace any kind of Unicode dash with ASCII hyphen
    val = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", val)
    return val.strip()
    
# Load env
load_dotenv(find_dotenv(), override=True)



# Get values from .env
api_key = clean_env("PINECONE_API_KEY")
index_name = clean_env("PINECONE_INDEX_NAME")

print("API Key loaded:", "Yes" if api_key else "No")
print("INDEX repr:", repr(index_name))

# Use Pinecone class (v3+ style)
from pinecone import Pinecone
pc = Pinecone(api_key=api_key)

# Connect to index
index = pc.Index(index_name)

# Test describe index
try:
    stats = index.describe_index_stats()
    print("✅ Connected to Pinecone!")
    print(f"Raw stats object: {stats}") # Print the raw stats object
    print("Index stats:")
    if stats:
        # Safely iterate if stats is not None
        # Access attributes directly as per v3+ documentation (stats is an object, not dict)
        print(f"  dimension: {getattr(stats, 'dimension', 'N/A')}")
        print(f"  index_fullness: {getattr(stats, 'index_fullness', 'N/A')}")
        print(f"  namespaces: {getattr(stats, 'namespaces', 'N/A')}") 
        print(f"  total_vector_count: {getattr(stats, 'total_vector_count', 'N/A')}")
    else:
        print("  Stats object is None or empty.")
except Exception as e:
    print("❌ Failed:", e)
