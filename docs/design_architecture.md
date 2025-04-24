# FinRAG – System Design Document

This document outlines the architectural design and system flow for FinRAG, an LLM-powered financial question-answering engine. It includes component descriptions, flow diagrams, and variants of the RAG pipeline.

---

## Design Objectives

- Modular components for retrieval, planning, extraction, and execution.
- Clear abstraction between data, logic, and model behavior.
- Flexible architecture adaptable to different retrieval and agent strategies.

---

##  System Components

### 1. Input Interface
- Accepts a financial question.
- Can operate in global search mode or potentially be adapted for sample-scoped search (though the default UI is global).

### 2. Retriever (`src/finrag/retriever.py`)
- Performs multi-stage retrieval:
  - **Primary Retrieval**: Dense vector search using **ChromaDB** based on question embeddings.
  - **Fallback Retrieval**: If ChromaDB fails/returns empty (and if operating in a sample-scoped context with necessary data), uses **BM25Okapi** (keyword-based) on chunks from the specific context.
  - **Reranker**: Uses **Cohere ReRank API** on the initial candidate chunks (from ChromaDB or BM25) to select the final top-k most relevant evidence chunks.

### 3. Agent (`src/finrag/agent.py`)
- Implements a two-step LLM process:
  - **Step 1: Specify Expression (`specify_and_generate_expression`)**: 
    - Receives the question and reranked evidence chunks.
    - Calls an LLM (e.g., GPT-4o-mini) to determine required financial items, maps them to placeholders (VAL_1, VAL_2...), generates a Python expression template (e.g., `(VAL_1 - VAL_2) / VAL_2`), and determines output format.
  - **Step 2: Extract Values (`extract_all_values_with_llm`)**: 
    - Receives the required items map (from Step 1) and evidence chunks.
    - Calls an LLM to extract the numerical values for each required item, applying scaling (millions, %, etc.) based on context within the evidence.

### 4. Executor (within `plan_and_execute` in `src/finrag/agent.py`)
- Substitutes the extracted numerical values (from Step 2) into the Python expression template (from Step 1).
- Executes the resulting standard Python expression using a safe `eval()` context.
- Formats the final numerical result based on the output format specified in Step 1 (e.g., formatting as a percentage).

### 5. Output Layer
- Returns a structured dictionary containing:
  - Final formatted answer (or error message).
  - The Python expression template used.
  - Intermediate results (Step 1 plan, Step 2 extracted values, substituted expression).
  - Tool used (`python_eval` or `none` on failure).
  - IDs of the evidence chunks used by the agent.

---

## Architecture Diagram

```
+-----------------------+      +-------------------------------------+      +------------------+
|    Question Input     | ---> |      Retriever (ChromaDB Primary)   | ---> |     Reranker     |
|      (Streamlit)      |      |      (Optional BM25 Fallback)       |      |   (Cohere API)   |
+-----------------------+      +-------------------------------------+      +------------------+
                                                                                     |
                                                                                     v
+------------------------------------+      +---------------------------------------------+
| Agent - Step 1: Specify Expression | ---> | Agent - Step 2: Extract Values (with Scale) |
|  (LLM: Required Items + Template)  |      |        (LLM: Finds Numbers in Evidence)     |
+------------------------------------+      +---------------------------------------------+
               |                                                |                      
               | (Template + Item Map)                          | (Extracted Numbers)  
               +------------------------------------------------+                      
                                      |
                                      v                      
                          +--------------------------+                     
                          | Executor (Python eval()) |                     
                          | Substitutes & Calculates |                     
                          +--------------------------+                     
                                      |
                                      v                      
                          +-----------------------------+                     
                          |   Final Answer + Details    |
                          |  (Formatted + Intermediates)|                     
                          +-----------------------------+                     
```

---

## Pipeline Variants & Modes

### Mode 1: Global Search (Current Streamlit Default)
- The `retrieve_evidence` function is called with an empty `turn` dictionary.
- ChromaDB performs a search across the entire indexed database based only on the question embedding.
- BM25 fallback is effectively disabled as it requires `turn` data.
- Suitable for general querying but may struggle to find highly specific data points without document context (as seen in UI testing).

### Mode 2: Sample-Scoped Search (Used in some Evaluations)
- The `retrieve_evidence` function receives a `turn` dictionary containing sample details (like `filename`).
- ChromaDB retrieval is filtered using the `doc_id` (filename) to search only within chunks from that specific source document.
- If ChromaDB fails for that document, BM25 fallback *can* function using `build_candidate_chunks(turn)`.
- Generally yields higher accuracy for dataset-specific questions due to the constrained search space.

### Vector Store
- Currently uses **ChromaDB** for persistent local vector storage (`./chroma_db_mpnet`).
- Previous iterations used Pinecone, but ChromaDB is the active implementation.

---

## Engineering Notes

- Retrieval, planning/extraction, and execution are logically separated components.
- The two-step agent allows for specialized prompts and potentially different models for planning vs. extraction (though currently uses the same model class by default).
- Use of standard Python `eval()` simplifies execution compared to a custom DSL.
- Relies on `python-dotenv` for managing API keys.
For detailed implementation history and technical decisions, see [@engineering_log.md](engineering_log.md).

---

## Summary

FinRAG's current architecture uses a multi-stage retrieval process (ChromaDB + Cohere Rerank) feeding into a two-step LLM agent (Specify Expression -> Extract Values) that culminates in the execution of a standard Python expression. It supports both global and potentially sample-scoped retrieval modes.

