# FinRAG – Test Suite Summary

This document provides an overview of the testing strategy implemented to ensure the stability, reliability, and correctness of the FinRAG system. It focuses on core module coverage, the use of mocks and stubs, and the intended scope of the tests.

---

## Testing Strategy

A modular and mock-driven testing approach is intended. Each functional unit of the FinRAG pipeline should be tested in isolation using synthetic inputs and stubbed dependencies (like external APIs or vector stores), minimizing external calls during testing.

---

## Test Files and Coverage (Reflecting Current Architecture)

| File                   | Intended Functionality Tested                                                                                         |
|------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `test_chunk_utils.py`  | Chunk builder (`build_candidate_chunks`) splitting conversation turns into pre-text, table rows, and post-text chunks.    |
| `test_retriever.py`    | Multi-stage retrieval pipeline: Mocked ChromaDB search, BM25 fallback logic (if applicable), and Cohere reranking simulation. |
| `test_agent.py`        | Two-step agent logic: Mocked LLM calls for `specify_and_generate_expression` (planning) and `extract_all_values_with_llm` (extraction), orchestration (`plan_and_execute`), and safe `eval()` execution. |
| `test_tools.py`        | (Minimal/Potentially Redundant) Basic tool initialisation or utility functions, if any remain outside agent/retriever logic. |
| `test_eval.py`         | Evaluation pipeline logic: Computation of execution accuracy (`answers_match`) and related metrics based on evaluation logs. |

---

## Mocks and Stubs Used (Intended)

- **Retrieval Tests** (`test_retriever.py`):
  - Mocked `chromadb.PersistentClient` and collection query results.
  - `DummyBM25` or similar logic for testing BM25 fallback path.
  - Mocked `cohere.Client.rerank` responses.
- **Agent Tests** (`test_agent.py`):
  - Mocked `openai.ChatCompletion.create` responses for both planning (Step 1) and extraction (Step 2) LLM calls, returning structured JSON payloads.
  - Testing of error handling for different failure modes (`specify_failed`, `extract_failed`, `eval_failed`).
- **Chunk Utility Tests** (`test_chunk_utils.py`):
  - Pure unit tests using sample input dictionaries.
- **Evaluation Tests** (`test_eval.py`):
  - Synthetic datasets/log entries to verify metric calculations.

---

## Example Test Output (Illustrative of Current Files)

```bash
$ poetry run pytest -v

tests/test_chunk_utils.py::... PASSED
tests/test_retriever.py::... PASSED
tests/test_agent.py::... PASSED
tests/test_tools.py::... PASSED 
tests/test_eval.py::... PASSED 

=== X passed in Y.ZZs === 
```
---

