# FinRAG: Financial Reasoning Assistant

FinRAG is a Retrieval-Augmented Generation (RAG) system tailored for quantitative reasoning over financial reports, built on the ConvFinQA dataset. It orchestrates evidence retrieval, DSL planning, and numeric execution to answer complex questions.

## RAG Pipeline Overview

### 1. Data Loading & Chunking
- **src/finrag/data_loaders.py**: Loads JSON ‘turn’ files, applies `build_candidate_chunks` to extract structured text and table chunks.
- **src/finrag/chunk_utils.py**: Splits conversation turns into pre-text, table rows, and post-text chunks with unique IDs.

### 2. Embedding & Indexing
- **src/finrag/embeddings.py**: Wraps OpenAI/Cohere embedding APIs and caches embeddings in `data/embeddings.jsonl`.
- **scripts/upsert_to_pinecone.py**: CLI script to batch process chunk embeddings and upsert them into a Pinecone index. Supports:
  - `--max-records <N>` to limit chunks uploaded
  - `--question-log <path>` to log sample IDs and questions
  - Automatic unique ID suffixes to avoid vector collisions
Requires the following environment variables (e.g., in `.env`): `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`.
Example usage:
```bash
python scripts/upsert_to_pinecone.py --max-records 1000 --question-log queries.jsonl
```

### 3. Retrieval
- **src/finrag/retriever.py**: Implements hybrid retrieval:
  1. **Pinecone** vector search when `USE_PINECONE=true`.
  2. Fallback **BM25Okapi** scoring on tokenized chunks.
  3. Cohere reranking to refine the top-k candidates.
  Returns `raw_chunks` and `reranked_chunks` for downstream planning.

### 4. Planning & Generation
- **src/finrag/agent.py**: Manages LLM-based planning via function calling:
  - `generate_tool_call()`: Formats question + evidence into a JSON function_call for the math DSL.
  - `plan_and_execute()`: Invokes the tool call, runs the returned DSL through the math tool, captures intermediates, and formats the final answer.

### 5. DSL Parsing & Execution
- **src/finrag/dsl.py**: Defines the DSL grammar and helper functions:
  - `parse_program()`, `serialize_program()`, and `execute_program()` to run DSL operations.
- **src/finrag/calculator.py**: Core math operations (`add`, `subtract`, `multiply`, `divide`) and percentage formatting.

### 6. Evaluation & CLI
- **src/finrag/eval.py**: End-to-end evaluation script measuring execution accuracy, program match rate, and tool usage.
- **src/finrag/cli.py**: Lightweight CLI for data inspection, embedding, and retrieval outside the Streamlit UI.

## Models

### OPENAI_CHAT_MODEL

Default: `gpt-4.1-mini-2025-04-14`  
Used by the LLM (`openai.ChatCompletion`) for planning and generating DSL-based math programs. Selected for its robust function-calling support, strong reasoning capabilities, and balanced latency/cost profile.

### EMBEDDING_MODEL

Default: `text-embedding-ada-002`  
Employed to compute semantic embeddings for text and table chunks. Chosen for high-quality embeddings, efficient performance, and cost-effectiveness in large-scale retrieval scenarios.

These variables can be customised in your `.env` file to experiment with different OpenAI models.

## Documentation
FinRAG's supplementary guides and logs are available in the `docs/` folder:

| Document                   | Path                                    | Description                                           |
|----------------------------|-----------------------------------------|-------------------------------------------------------|
| Product Requirements       | [docs/PRD.md](docs/PRD.md)                             | Objectives, features, non-functional requirements     |
| Design & Architecture      | [docs/design_architecture.md](docs/design_architecture.md) | System design, data flow, component overview          |
| Engineering Log            | [docs/engineering_log.md](docs/engineering_log.md)         | Development milestones, decisions, challenges         |
| Evaluation                 | [docs/evaluation.md](docs/evaluation.md)                   | Evaluation metrics, CLI usage, sample outputs         |
| Production Planning        | [docs/productionisation_plan.md](docs/productionisation_plan.md) | Production readiness, observability, cost strategies  |
| Requirements & Setup       | [docs/requirements_setup.md](docs/requirements_setup.md)   | Setup guide: dependencies, env vars, directory layout |
| Streamlit UI Walkthrough   | [docs/streamlit_ui_walkthrough.md](docs/streamlit_ui_walkthrough.md) | Streamlit interface guide with screenshots            |
| Test Suite Summary         | [docs/test_suite_summary.md](docs/test_suite_summary.md)   | Automated test overview: coverage, mocks, outputs     |

For a concise pipeline summary, see the PRD and Design docs.

## Dataset
This project uses data from the ConvFinQA dataset, which provides conversational question-answer pairs over financial documents.

Download it from: [https://github.com/czyssrs/ConvFinQA](https://github.com/czyssrs/ConvFinQA)

## Source File Overview

Below is a brief description of each module in `src/finrag`:

- **agent.py**: Orchestrates the LLM-based planning and execution. Defines `generate_tool_call` and `plan_and_execute` to construct DSL programs, invoke the math tool, and format results.
- **calculator.py**: Implements the arithmetic DSL execution engine. Parses and evaluates `add`, `subtract`, `multiply`, `divide` functions and manages intermediate results.
- **chunk_utils.py**: Contains logic to split raw transcript turns into candidate text chunks, filtering by length and metadata.
- **cli.py**: Provides a command-line interface for simple data loading, embedding, and retrieval operations outside of the web UI.
- **data_loaders.py**: Loads JSON files from disk, applies `build_candidate_chunks`, and returns structured chunks with metadata.
- **dsl.py**: Defines the DSL grammar and utilities for serialising/deserialising programs to and from JSON.
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
See the [Test Suite Summary](docs/test_suite_summary.md) for further details about tests.

## Evaluation

```bash
poetry run python eval.py --split dev --sample 5
```
See the [Evaluation](docs/evaluation.md) section for details about evals methodology.
