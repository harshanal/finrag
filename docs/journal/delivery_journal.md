# Delivery Journal: PRD Feature Breakdown

This document provides a **chronological** account of how each core feature (“P”) was delivered, with detailed design discussions, encountered challenges, their resolutions, and metrics evolution.

---

## 1. Pinecone Integration

**When**: Initial development (Step 1135)

**Design Discussion:**  
To enable high-quality semantic retrieval of financial report chunks, we integrated Pinecone early in the pipeline. Pinecone offered a managed vector database that seamlessly handles large-scale embedding storage and similarity queries. We prioritized vector search to capture nuanced semantic relationships that keyword matching alone might miss.

**Design Details:**  
- Introduced `USE_PINECONE` flag in `retriever.py` to toggle vector search.  
- Initialized Pinecone client using environment variables (`PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_ENVIRONMENT`).  
- Wrapped all Pinecone calls in try/except and fallback logic to preserve flow if the service is unavailable.

**Issues:**  
- Misconfigured environment variables led to silent failures and empty results.  
- Initial logs lacked detail for diagnosing index initialization issues.

**Resolutions:**  
- Enhanced startup logging with explicit messages on flag status and client parameters.  
- Added warning logs when query returns zero vectors.

**Accuracy Impact:**  
- Without reliable retrieval, end-to-end accuracy remained at ~17%.

---

## 2. BM25 Fallback Mechanism

**When**: Directly after Pinecone integration

**Design Discussion:**  
Aware that external vector services may fail or increase latency, we implemented a local BM25Okapi fallback. BM25 provides a robust term-frequency–based ranking, ensuring retrieval continuity even when vectors fail. This hybrid strategy balances modern embeddings with classical IR.

**Design Details:**  
- Inlined BM25 logic in `retriever.py` using `BM25Okapi` from `rank_bm25`.  
- Tokenized candidate chunks from `chunk_utils.build_candidate_chunks`.  
- Selected top-k candidates based on BM25 scores when Pinecone is disabled or returns no results.

**Issues:**  
- A `NameError: bm25_retrieve` placeholder left our fallback path broken.  
- Initial BM25 results were unfiltered and sometimes low-quality.

**Resolutions:**  
- Replaced undefined calls with direct BM25 implementation.  
- Added result filtering and top-k reranking steps.

**Accuracy Impact:**  
- Activation of BM25 fallback raised execution accuracy from ~17% to **~21.6%**.

---

## 3. Structured Output Parsing (Pydantic)

**When**: Following retrieval stabilization

**Design Discussion:**  
To enforce consistency in LLM–to–tool communication, we used Pydantic for strict schema validation. This ensures that only well-formed DSL programs are executed, reducing silent errors from malformed JSON.

**Design Details:**  
- Defined a Pydantic model `RunMathToolArguments` requiring a single `program` string.  
- Updated `generate_tool_call` in `agent.py` to validate LLM outputs against the model.  
- Implemented a regex-based fallback extraction if validation fails.

**Issues:**  
- LLM occasionally returned extra text or incorrect JSON formats.  
- Regex fallback was brittle in edge cases.

**Resolutions:**  
- Added warning logs on validation failure with raw response dumps.  
- Retained regex fallback as a secondary safety net while encouraging strict outputs.

**Accuracy Impact:**  
- Program match rate improved; tool usage stabilized near 97–100%.

---

## 4. Upsert Script Enhancements

**When**: Data indexing phase

**Design Discussion:**  
To control cost and ensure data integrity in Pinecone, we enhanced the upsert script with batching limits and unique IDs. Recording questions alongside chunk IDs improved traceability.

**Design Details:**  
- Added CLI options `--max-records` and `--question-log` to `scripts/upsert_to_pinecone.py`.  
- Incorporated UUID suffixes to each `chunk_id` to prevent silent overwrites.  
- Logged sample IDs and questions to a separate file for auditing.

**Issues:**  
- Identical chunk IDs caused later embeddings to overwrite earlier ones.

**Resolutions:**  
- Implemented 8-character UUID postfix, guaranteeing unique IDs.  
- Verified 500+ records persisted correctly in Pinecone.

---

## 5. UI Conversation History Removal

**When**: Early UI iteration

**Design Discussion:**  
Feedback indicated the conversation history panel cluttered the interface for single-turn queries. We streamlined the UX by removing saved history, focusing on direct Q&A.

**Design Details:**  
- Deleted `st.session_state.chat_history` initialization and history display blocks.  
- Removed Clear button and history appending logic in `app.py`.

**Issues:**  
- Uninitialized session state warnings appeared when accessing history.

**Resolutions:**  
- Eliminated all history references, leaving only a single text input per run.

---

## 6. Streamlit UI Simplification

**When**: Final UI polish

**Design Discussion:**  
To reduce cognitive load, we refined titles and labels. A concise subtitle clarifies dataset context, and an unlabeled text area avoids redundancy.

**Design Details:**  
- Updated title to “FinRAG: Financial Reasoning Assistant”.  
- Changed subtitle to reference the ConvFinQA dataset.  
- Left the question input area label blank.

**Issues:**  
- Early labels were verbose and confusing.

**Resolutions:**  
- Condensed UI to a clean, two-line header and a blank prompt area.

---

## 7. Documentation & Code Comments

**When**: Parallel to refactoring

**Design Discussion:**  
Comprehensive documentation is essential for maintainability. We annotated each core module with pipeline role headers and expanded the README with a step-by-step overview.

**Design Details:**  
- Inserted detailed module-level docstrings in `agent.py`, `retriever.py`, `dsl.py`, `chunk_utils.py`, `data_loaders.py`, and `embeddings.py`.  
- Rewrote `README.md` to outline data flow, module responsibilities, and setup instructions.

**Issues:**  
- Initial README lacked cohesive pipeline context.

**Resolutions:**  
- Structured README into six RAG stages and added CLI examples.

---

## 8. Accuracy Tracking & Iterations

**When**: Throughout the project

**Design Discussion:**  
Continuous metric monitoring informed our priorities. Observing accuracy, program matching, and tool usage guided fallback and parsing enhancements.

**Key Metrics:**  
- **Execution Accuracy**: increased from **17.1%** to **21.6%** after BM25 fallback.  
- **Program Match Rate**: stabilized around 40–45%.  
- **Tool Usage**: consistently nearly 100%.

---

**Learnings:**  
- Unique IDs prevent silent data overwrites in vector stores.  
- Rich logging and error handling reveal pipeline blind spots early.  
- A hybrid retrieval approach ensures robust coverage when vectors fail.

---

**Next Steps:**  
1. Explore full-document vectorization.  
2. A/B test advanced reranking models.  
3. Extend DSL with aggregate functions and multi-step workflows.

*This journal is maintained to document the end-to-end engineering decisions and outcomes.*
