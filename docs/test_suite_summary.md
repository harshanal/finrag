# FinRAG – Test Suite Summary

This document provides an overview of the testing strategy implemented to ensure the stability, reliability, and correctness of the FinRAG system. It focuses on core module coverage, use of mocks and stubs, and examples of expected test outputs.

---

## ✅ Testing Philosophy

A modular and mock-driven testing approach is used. Each functional unit of the FinRAG pipeline is tested in isolation using synthetic inputs and stubbed dependencies, minimizing external API calls.

---

## 🧪 Test Files and Coverage

| File                   | Functionality Tested                                                                                       |
|------------------------|------------------------------------------------------------------------------------------------------------|
| `test_calculator.py`   | Arithmetic operations (`add`, `subtract`, `multiply`, `divide`), percentage formatting (`format_percentage`), and direct `execute_program` on `Operation` lists. |
| `test_dsl.py`          | DSL parsing (`parse_program`), serialization (`serialize_program`), execution flow including percent formatting and error cases. |
| `test_chunk_utils.py`  | Chunk builder (`build_candidate_chunks`) splitting conversation turns into pre-text, table rows, and post-text chunks. |
| `test_retriever.py`    | Hybrid retrieval pipeline: vector search fallback, BM25 ranking, and dummy reranking.                     |
| `test_agent.py`        | LLM tool-call generation (`generate_tool_call`) and fallback extraction logic for malformed JSON responses. |
| `test_tools.py`        | Math tool integration: `run_math_tool` end-to-end processing of DSL programs.                                |
| `test_eval.py`         | Evaluation pipeline logic: computation of execution accuracy and related metrics.                          |

---

## 🧰 Mocks and Stubs Used

- **Retrieval Tests** (`test_retriever.py`):
  - Dummy Pinecone index and vector responses.
  - `DummyBM25` for BM25 ranking.
  - Stubbed rerank function returning fixed order.
- **Agent Tests** (`test_agent.py`):
  - Mocked LLM responses via predefined function call payloads.
  - Regex fallback tested with simulated malformed content.
- **Calculator & DSL Tests** (`test_calculator.py`, `test_dsl.py`):
  - Pure unit tests without external dependencies.
- **Tool Tests** (`test_tools.py`):
  - Direct invocation of `run_math_tool`, using real calculator logic.
- **Evaluation Tests** (`test_eval.py`):
  - Synthetic datasets to verify metric calculations.

---

## 🔍 Example Test Output

```bash
$ poetry run pytest -v

tests/test_calculator.py::test_add_subtract_multiply_divide PASSED
tests/test_calculator.py::test_format_percentage PASSED
tests/test_calculator.py::test_execute_program_simple PASSED
tests/test_dsl.py::test_parse_and_serialize PASSED
tests/test_dsl.py::test_execute_program PASSED
tests/test_dsl.py::test_division_by_zero PASSED
tests/test_dsl.py::test_invalid_ref PASSED
tests/test_chunk_utils.py::test_build_candidate_chunks_empty PASSED
tests/test_chunk_utils.py::test_build_candidate_chunks_all_fields PASSED
tests/test_retriever.py::test_retrieve_evidence_rerank PASSED
tests/test_agent.py::test_generate_tool_call_with_fallback PASSED
tests/test_tools.py::test_run_math_tool PASSED
tests/test_eval.py::test_exec_accuracy_computation PASSED

=== 13 passed in 11.22s ===
```

> All tests pass, ensuring module-level reliability and regression safety.

---

FinRAG’s test suite validates individual components—retrieval, DSL parsing, math execution, agent orchestration, and evaluation—providing confidence in maintainable AI solution development.
