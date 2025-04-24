# Product Requirements Document (PRD): FinRAG – Financial Reasoning Assistant

## Document Version
- **Version:** 1.1 (Reflecting Final Prototype State)
- **Last Updated:** 2025-04-23
- **Author:** Harshana Liyanage

---

## 1. Objective

To build a prototype LLM-powered system (FinRAG) that answers complex financial questions based on financial documents (text and tables), demonstrating multi-stage retrieval, a robust two-step reasoning agent, Python expression execution, experiment tracking, and accuracy reporting.

---

## 2. Background

The task is to create a question-answering system that works on the ConvFinQA dataset, showcasing:
- Retrieval from unstructured and semi-structured documents using vector search and reranking.
- Execution of mathematical operations derived from LLM-generated Python expression templates.
- A modular RAG architecture suitable for iterative development.
- Evaluation metrics tracking and experiment logging for explainability.

---

## 3. Core Features and Functional Requirements (Reflecting Final State)

### P1 – Retrieval MVP
- Build multi-stage retriever using **ChromaDB** for dense vector search (primary) and **BM25Okapi** (fallback, context-dependent).
- Use locally computed embeddings (`all-mpnet-base-v2`) for ChromaDB indexing.
- Integrate **Cohere rerank** API to re-rank top-k candidates retrieved.
- Evaluate using Recall@k on a dev subset.

### P2 – Agent Core: Two-Step Reasoning
- Implement a two-step agent process:
    - **Step 1 (`specify_and_generate_expression`):** LLM analyzes question + evidence to generate a calculation plan, including required items mapped to placeholders (VAL_1, VAL_2...) and a standard **Python expression template**.
    - **Step 2 (`extract_all_values_with_llm`):** LLM extracts numerical values for the required items from evidence, applying scaling (millions, %, etc.).
- Use OpenAI function-calling API to structure LLM outputs for both steps.
- Validate key outputs from LLM steps (e.g., presence of required fields, basic safety checks on templates).

### P3 – Agent Orchestration & Execution
- Implement `plan_and_execute()` logic to orchestrate the two-step agent flow.
- Substitute extracted values into the Python expression template.
- Execute the populated Python expression using a safe `eval()` context.
- Ensure full traceability through logs for the plan, extracted values, final expression, and answer.

### P4 – Evaluation Framework & Experiment Tracking
- Provide `run_evaluation.py` CLI to evaluate system on dev/test split.
- Compute:
  - Execution Accuracy
  - Failure analysis (e.g., `substitute_failed`, `eval_failed` rates)
- Log full run details (plan, values, answer, gold standard) in JSONL format.
- Integrate **Weights & Biases (W&B)** for logging metrics, parameters, and evaluation results across runs.

### P5 – Streamlit Interface
- Build interactive UI (`app.py`) for live demonstrations of the **global search** pipeline.
- Components:
  - Question input
  - Retrieved context display (pre/post rerank)
  - Agent results: Final Answer (prominent), Python Expression Template, Intermediate Values (plan + extracted values), Raw Debug Output.
- Implement two-button flow (Retrieve -> Run Agent).

### P6 – Vector Store Implementation: ChromaDB
- Index the full `train.json` dataset (~10k+ chunks) into a local, persistent **ChromaDB** store (`./chroma_db_mpnet`).
- Use `all-mpnet-base-v2` embeddings computed locally for indexing.
- Provide script (`scripts/load_chroma_mpnet.py`) for repeatable indexing.

--- 

## 4. Non-Functional Requirements

- **Performance**: Agent execution latency acceptable for interactive use (typically < 10-15 seconds per query, depending on LLM response times).
- **Reliability**: Graceful handling of LLM errors (e.g., invalid JSON, failed extraction) and retrieval failures.
- **Modularity**: Maintain logical separation between retrieval, agent steps (planning, extraction), and execution.
- **Explainability**: Each step's inputs and outputs are logged and displayed in the UI where appropriate.

--- 

## 5. Evaluation Metrics

- **Execution Accuracy**: Does the final computed answer match the gold answer (within tolerance)?
- **Failure Rate Analysis**: Track rates of failures at different stages (e.g., `specify_or_express_failed`, `substitute_failed`, `eval_failed`).
- **Tool Usage Rate:** (Less relevant with Python eval) Monitor successful completion of the `plan_and_execute` flow.

--- 

## 6. Risks & Mitigations (Reflecting Final State)

| Risk                                     | Mitigation                                                                                             |
|------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Global retrieval lacks precision         | Use Cohere reranker; Log retrieved chunks for analysis; Acknowledge limitation in documentation.         |
| LLM value extraction errors/hallucinations| Detailed prompts with scaling instructions & negative constraints; Handle `null` returns; Log intermediates. |
| `eval()` security concerns               | Use `eval()` with restricted `globals` and `locals` to prevent execution of arbitrary code.           |
| Long indexing time for ChromaDB          | Document time taken; Suggest GPU use for faster re-indexing; Perform indexing as a separate setup step. |
| LLM API errors / costs                   | Implement basic error handling; Use efficient models (e.g., GPT-4o-mini); Manage API keys via `.env`.      |

--- 

## 7. Stretch Goals / Future Work (Updated)

- **Hybrid Search Fusion**: Implement fusion of sparse (BM25) and dense (ChromaDB) results *before* reranking.
- **Advanced Chunking/Retrieval**: Explore proposition-based indexing, smaller/overlapping chunks, or recursive retrieval.
- **Improved Value Extraction**: Investigate dedicated NER/parsing models or more sophisticated LLM prompting techniques for numerical extraction and scaling.
- **Agent Self-Correction/Refinement**: Add logic for the agent to retry steps or refine its plan upon encountering errors.
- **Broader Tool Integration**: Explore tools beyond basic calculation (e.g., web search for current data, plotting).
- **OCR/Multimodal Input**: Integrate tools for processing tables/figures directly from images/PDFs (e.g., LlamaParse, vendor solutions).
- **Productionisation Readiness**: Enhance observability, monitoring, and cost tracking.

--- 

## 8. Appendix

- Dataset: [ConvFinQA GitHub Repo](https://github.com/czyssrs/ConvFinQA)
- Gold Example (Illustrative of Planning Step):
  ```json
  {
    "question": "What was the percentage change in net cash from operating activities from 2008 to 2009?",
    "answer": "14.1%",
    "// Step 1 Output (Illustrative)": {
        "calculation_type": "percentage_change",
        "required_items_map": {
            "VAL_1": "Net cash from operating activities 2009",
            "VAL_2": "Net cash from operating activities 2008"
         },
        "python_expression_template": "(VAL_1 - VAL_2) / VAL_2",
        "output_format": "percent"
    }
  }
  ```

