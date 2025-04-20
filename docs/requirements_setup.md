# FinRAG – Requirements & Setup Guide

This document provides guidance on dependency management, environment setup, and API key configuration to ensure reproducible development and deployment of the FinRAG system.

---

## Project Dependency Files

### 1. Poetry Setup (Preferred)

The project uses [Poetry](https://python-poetry.org/) for dependency and environment management.

**Key file:** `pyproject.toml`

Includes dependencies such as:

```toml
[tool.poetry.dependencies]
python = "^3.10"
openai = "^1.12.0"
cohere = "^4.0.0"
rank_bm25 = "^0.2.2"
tqdm = "*"
python-dotenv = "^1.0.0"
streamlit = "^1.31.1"
pydantic = "^2.5.3"
pytest = "^7.4.4"
pinecone-client = "^2.2.4"
```

To install:

```bash
poetry install
```

To activate shell:

```bash
poetry shell
```

---

### 2. requirements.txt (For Pip Users)

If not using Poetry, install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

This file is auto-generated from `poetry export` for pip compatibility.

---

## 🛠️ Environment Setup

### Python Version

- Python 3.10 or higher is recommended.
- Use `pyenv`, `poetry`, or `conda` to manage versions.

---

## Environment Variables

### 1. .env File

All API keys and credentials are loaded from a `.env` file.

Create a `.env` file from the example template:

```bash
cp .env.example .env
```

### 2. .env.example Template

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4

# Cohere
COHERE_API_KEY=your_cohere_api_key

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=finrag-index
PINECONE_ENVIRONMENT=us-west1-gcp
```

These variables are loaded automatically via `python-dotenv`.

---

## Streamlit Dashboard

After installing dependencies and configuring your `.env`, launch the interactive UI:

```bash
poetry run streamlit run app.py
```

---

## Notes

- Always commit `pyproject.toml` and `poetry.lock` but not the `.env` file.
- Add `outputs/`, `.env`, `.venv/` to `.gitignore`.
---

## Directory Overview

```
finrag/
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
└── src/
    └── finrag/
        └── ...
```

---
