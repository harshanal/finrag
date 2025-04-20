
# FinRAG – System Design Document

This document outlines the architectural design and system flow for FinRAG, an LLM-powered financial question-answering engine. It includes component descriptions, flow diagrams, and variants of the MVP pipeline.

---

## 🎯 Design Objectives

- Modular components for retrieval, reasoning, and execution.
- Clear abstraction between data, logic, and model behavior.
- Multiple deployment variants to showcase flexibility.

---

## 🧩 System Components

### 1. Input Interface
- Accepts financial question (optionally conversational).
- Selects or matches context from a financial document set.

### 2. Retriever
- Hybrid retrieval:
  - **BM25**: Classical keyword-based scoring.
  - **Embeddings**: Semantic similarity (OpenAI, text-embedding-ada-002).
  - **Fusion**: Combine top-k BM25 and semantic matches.
- **Reranker**: Cohere ReRank to improve quality of final selected evidence.

### 3. Planner (LLM Agent)
- LLM (OpenAI GPT-4 / O4-mini / Claude) receives:
  - Current question
  - Chat history (if available)
  - Retrieved evidence
- Outputs a DSL program (e.g., `subtract(100, 80), divide(#0, 80)`).

### 4. Tool Executor
- Parses and executes the DSL using a Python sandbox.
- Applies numeric operations and formatting (e.g., `format_percentage`).

### 5. Output Layer
- Returns:
  - Final answer
  - Retrieved context
  - Generated DSL program
  - Comparison with gold answer (if available)

---

## 📊 Architecture Diagram

```
+-----------------------+     +--------------------+
|    Question Input     | --> |     Retriever      |
|  (UI or Evaluation)   |     |  (BM25 + Embedding)|
+-----------------------+     +--------------------+
                                      |
                                      v
                             +------------------+
                             |     Reranker     |
                             |   (Cohere API)   |
                             +------------------+
                                      |
                                      v
                            +---------------------+
                            |   Planner (LLM)     |
                            | Generates DSL Logic |
                            +---------------------+
                                      |
                                      v
                            +----------------------+
                            |   DSL Executor       |
                            | add, subtract, etc.  |
                            +----------------------+
                                      |
                                      v
                            +----------------------+
                            |  Answer + Program +  |
                            |     Retrieved Text   |
                            +----------------------+
```

---

## 🔁 MVP Variants

### MVP 1: Page-Based Retrieval (Default)
- Uses a selected document or page as context.
- Higher accuracy due to constrained retrieval.

### MVP 2: Global Search
- Freeform question selects across all indexed records.
- Uses Cohere Rerank to improve retrieval.
- Accuracy drops due to retrieval noise.

### MVP 3: Pinecone Vector Search
- Replaces in-memory vector DB with Pinecone.
- Enables scalable retrieval across thousands of records.
- Accuracy limited by small data size and free-tier upload caps.

---

## ⚙️ Engineering Notes

- Retrieval and execution are decoupled from LLM, improving explainability.
- Each module is testable in isolation.
- Feature flags allow switching between modes (BM25, Pinecone, etc).
- Prompt versioning enables fine-tuning over time.

---

## 📌 Summary

FinRAG’s architecture emphasises clean boundaries between retrieval, reasoning, and execution. It is flexible to support multiple strategies, scalable for larger datasets, and testable to maintain robust QA behavior.

