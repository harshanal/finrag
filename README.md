# FinRAG

FinRAG is a Conversational QA Agent prototype for multi‑step quantitative questions on financial reports (text + tables).

## Setup

```bash
pipx install poetry
git clone https://github.com/harshanal/finrag.git finrag
cd finrag
poetry install
poetry shell
```

## Environment Variables

This project requires API keys from OpenAI, Pinecone, and Cohere.

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Fill in the API keys in the `.env` file.

    ```bash
    OPENAI_API_KEY=your-openai-key
    PINECONE_API_KEY=your-pinecone-key
    COHERE_API_KEY=your-cohere-key
    ```

## Dataset
This project uses data from the ConvFinQA dataset, which provides conversational question-answer pairs over financial documents.

Download it from: [https://github.com/yangji9181/ConvFinQA](https://github.com/yangji9181/ConvFinQA)

## Source File Overview

Below is a brief description of each module in `src/finrag`:

- **agent.py**: Orchestrates the LLM-based planning and execution. Defines `generate_tool_call` and `plan_and_execute` to construct DSL programs, invoke the math tool, and format results.
- **calculator.py**: Implements the arithmetic DSL execution engine. Parses and evaluates `add`, `subtract`, `multiply`, `divide` functions and manages intermediate results.
- **chunk_utils.py**: Contains logic to split raw transcript turns into candidate text chunks, filtering by length and metadata.
- **cli.py**: Provides a command-line interface for simple data loading, embedding, and retrieval operations outside of the web UI.
- **data_loaders.py**: Loads JSON files from disk, applies `build_candidate_chunks`, and returns structured chunks with metadata.
- **dsl.py**: Defines the DSL grammar and utilities for serializing/deserializing programs to and from JSON.
- **embeddings.py**: Wraps the OpenAI/Cohere embedding APIs to produce vector representations for texts.
- **eval.py**: Runs end-to-end evaluation on a dataset split, measuring execution accuracy, program match rate, and tool usage rate.
- **index_embeddings_pinecone.py**: Standalone script to compute embeddings and build a Pinecone index from raw data chunks.
- **logging_conf.py**: Configures Python logging settings (format, level, handlers) for consistency across modules.
- **retriever.py**: Implements hybrid retrieval: first tries Pinecone, falls back to BM25 if needed, then reranks with Cohere.
- **tools.py**: Defines wrappers around external tool calls (e.g., `run_math_tool`) and any helper integrations.
- **utils.py**: Utility functions such as loading environment variables, cleaning paths, and other small helpers.

## Running Tests

```bash
poetry run pytest
```

## Evaluation

```bash
poetry run python eval.py --split dev --sample 5
