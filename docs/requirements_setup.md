# FinRAG – Requirements & Setup Guide

This document provides guidance on dependency management, environment setup, and API key configuration to ensure reproducible development and deployment of the FinRAG system.

---

## Project Dependency Files

### 1. Poetry Setup (Preferred)

The project uses [Poetry](https://python-poetry.org/) for dependency and environment management.

**Key file:** `pyproject.toml`

Includes key dependencies such as:

```toml
[tool.poetry.dependencies]
python = "^3.10"
openai = "^1.12.0" # Or compatible version
cohere = "^4.0.0" # Or compatible version
chromadb-client = "^0.4.24" # Or compatible version for chromadb
sentence-transformers = "*" # For embeddings
pandas = "*" # Used in data loading/processing
tqdm = "*" # Progress bars
python-dotenv = "^1.0.0"
streamlit = "^1.31.1" # Or compatible version
pydantic = "^2.5.3"
pytest = "^7.4.4" # For testing
# Ensure other necessary dependencies like numpy, etc., are included
```
*(Note: This is illustrative; refer to the actual `pyproject.toml` for the exact list and versions.)*

To install dependencies:

```bash
poetry install
```

To activate the virtual environment shell:

```bash
poetry shell
```

---

### 2. requirements.txt (For Pip Users)

If not using Poetry, a `requirements.txt` file might be available (potentially generated via `poetry export`). Install using:

```bash
pip install -r requirements.txt
```
*(Note: Using Poetry is recommended for managing dependencies accurately.)*

---

## 🛠️ Environment Setup

### Python Version

- Python 3.10 or higher is recommended.
- Use `pyenv`, `poetry`, or `conda` to manage Python versions consistently.

---

## Environment Variables

### 1. .env File

API keys and potentially other configurations are loaded from a `.env` file in the project root.

Create your `.env` file by copying the example template:

```bash
cp .env.example .env
```

**Important:** Add `.env` to your `.gitignore` file to avoid committing secrets.

### 2. .env.example Template

The `.env.example` file should contain placeholders for required secrets:

```env
# OpenAI API Key (Required for agent LLM calls)
OPENAI_API_KEY=your_openai_api_key

# Cohere API Key (Required for reranking)
COHERE_API_KEY=your_cohere_api_key

# Optional: Specify different models for planning/extraction steps
# OPENAI_CHAT_MODEL_PLANNING=gpt-4o 
# OPENAI_CHAT_MODEL_EXTRACTION=gpt-4o-mini
```
*(Note: Pinecone variables are no longer needed as ChromaDB is used locally.)*

Fill in your actual API keys in the `.env` file.

---

## Data Setup: ChromaDB Vector Store

The system relies on a local ChromaDB vector store containing embeddings of the financial document chunks.

1.  **Prerequisite:** Ensure you have the `train.json` dataset file available (likely in a `data/` directory, based on ConvFinQA structure).
2.  **Run Indexing Script:** Execute the script provided to process the data, generate embeddings using `all-mpnet-base-v2`, and load them into the local ChromaDB instance.
    ```bash
    # Ensure you are in the activated poetry environment
    poetry run python scripts/load_chroma_mpnet.py 
    ```
    *This process can take a significant amount of time (~8 hours observed on a non-GPU laptop) and will create the `./chroma_db_mpnet` directory.* 
3.  **Verify:** Check for the presence of the `./chroma_db_mpnet` directory containing the database files.

---

## Streamlit Dashboard

After installing dependencies, configuring your `.env` file, and successfully setting up the ChromaDB data, you can launch the interactive UI:

```bash
poetry run streamlit run app.py
```

---

## Directory Overview (Simplified)

```
finrag/
├── pyproject.toml
├── poetry.lock
├── .env.example
├── .gitignore
├── data/                 # Likely location for train.json, etc.
│   └── train.json
├── scripts/
│   └── load_chroma_mpnet.py # Script to populate ChromaDB
├── src/
│   └── finrag/           # Core library code (retriever, agent, etc.)
├── docs/                 # Documentation files
├── app.py                # Streamlit application
├── chroma_db_mpnet/      # ChromaDB data directory (created by script)
└── ... (other files like tests, notebooks)
```

---
