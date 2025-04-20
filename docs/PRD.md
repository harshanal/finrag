
# Product Requirements Document (PRD): FinRAG – Financial Reasoning Assistant

## Document Version
- **Version:** 1.0
- **Last Updated:** 2025-04-20
- **Author:** Harshana Liyanage

---

## 1. Objective

To build a prototype LLM-powered system (FinRAG) that answers complex financial questions based on financial documents (text and tables), demonstrating tool-use, numerical reasoning, conversational memory, and accuracy reporting.

---

## 2. Background

The task is to create a question-answering system that works on the ConvFinQA dataset, showcasing:
- Retrieval from unstructured and semi-structured documents.
- Execution of DSL-based mathematical operations.
- Robust architecture for production-level LLM solutions.
- Evaluation metrics and explainability.

---

## 3. Core Features and Functional Requirements

### P1 – Retrieval MVP
- Build hybrid retriever using BM25 + OpenAI embeddings.
- Integrate Cohere rerank to re-rank top-k candidates.
- Evaluate using Recall@k on a dev subset (target: >80%).

### P2 – Tool Schema & Executor
- Implement structured DSL parser (add, subtract, divide, percentage).
- Use OpenAI function-calling API with structured output.
- Validate tool programs using Pydantic schema.
- Execute DSL using a secure Python interpreter or function dispatcher.

### P3 – Agent Orchestration
- Implement `plan_and_execute()` logic to tie together retrieval, planning, and execution.
- Include regex fallback for malformed outputs.
- Ensure full traceability through logs for DSL, context, results.

### P4 – Evaluation Framework
- Provide CLI to evaluate system on dev/test split.
- Compute:
  - Execution Accuracy
  - Program Match Rate
  - Tool Usage Rate
- Log full run with gold answers and generated DSL for inspection.

### P5 – Streamlit Interface
- Build interactive UI for live demonstrations.
- Components:
  - Question input
  - Retrieved context (pre/post rerank)
  - DSL program + result
  - Comparison with ground-truth answer
- Include dropdown for sample questions and context highlighting.

### P6 – Pinecone Integration
- Upsert 500+ chunks to Pinecone for semantic search.
- Add fallback to BM25 when Pinecone fails or returns empty.
- Track upload limits due to Pinecone's free tier (570 records max).
- Measure and compare accuracy vs BM25 baseline.

---

## 4. Non-Functional Requirements

- **Performance**: Tool execution latency < 2 seconds per query.
- **Reliability**: Fallback mechanisms ensure tool execution and retrieval never fail silently.
- **Scalability**: Modular retrieval and agent logic for cloud-scale expansion.
- **Explainability**: Each step (retrieval, planning, execution) is exposed for traceability.

---

## 5. Evaluation Metrics

- **Execution Accuracy**: Does the output value match the answer?
- **Program Match Rate**: Does the generated DSL match gold program?
- **Tool Usage Rate**: Is the planner consistently producing valid DSL programs?

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Limited Pinecone index capacity | Use top-performing 570 records, fallback to BM25 |
| LLM generating malformed outputs | Enforce JSON schema via Pydantic + regex fallback |
| Low recall affecting accuracy | Use reranking and chunk ID traceability |
| Model hallucination | Evidence-aware prompts, numerical grounding |

---

## 7. Stretch Goals / Future Work

- OpenAI Evals or W&B integration for visual eval tracking.
- Document-level retrieval using FAISS/HNSW.
- DSL expansion (multi-hop chaining).
- OCR and figure analysis integration (in production scenario consider multimodal tools such as Landing.ai's Agentic Document Extraction or LlamaParse )
- Productionisation readiness (observability, consolidated cost monitoring across 3rd party services).

---

## 8. Appendix

- Dataset: [ConvFinQA GitHub Repo](https://github.com/czyssrs/ConvFinQA)
- Gold Example:
  ```json
  {
    "question": "What was the percentage change in net cash from operating activities from 2008 to 2009?",
    "answer": "14.1%",
    "program": "subtract(2009_value, 2008_value), divide(#0, 2008_value)"
  }
  ```

