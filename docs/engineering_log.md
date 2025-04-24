# FinRAG – Engineering Log

## Overview

This log documents the technical evolution of the FinRAG prototype over a one-week development period, aligned with key milestones outlined in the product requirements document (PRD). Each section details decisions made, challenges faced, and lessons learned across the development cycle, aiming for transparency in the decision-making process.

---

## P1 – Retrieval MVP

### Objective

Develop a hybrid retriever combining lexical (BM25) and semantic (OpenAI embeddings) retrieval, enhanced with reranking via Cohere, initially focused on retrieving evidence within the context of a specific financial document (page-level retrieval).

### Deliverables

- `retriever.py` with initial hybrid search logic (BM25 + Embeddings + Reranker)
- Chunking and ID tagging via `chunk_utils.py`
- Evaluation notebook for top-k recall within document context

### Design Decisions

- Combined top-k BM25 and embedding-based retrieval intended to offset weaknesses of each method.
- Integrated Cohere as a reranking layer on fused retrieval results.
- Split text by paragraph and tables by row; tagged chunks with unique identifiers (`sample_id::chunk_type::position`).

### Key Issues

- Initial semantic-only retrieval showed low recall without BM25 fusion.
- Chunk ID generation needed refinement for uniqueness.

### Fixes and Enhancements

- Developed unit tests for recall on a fixed dev subset.
- Normalized token processing in BM25 and refined rerank cutoff thresholds.

### Outcome

Achieved good recall within the constrained page-level context. Provided a reliable base for the initial agent development focused on reasoning over provided evidence.

---

## P2 – Initial Agent Design: DSL Generation

### Objective

Develop an agent capable of planning and executing calculations based on retrieved evidence, initially using a custom Domain-Specific Language (DSL) and OpenAI's function-calling.

### Deliverables

- Initial DSL interpreter (`calculator.py`, `dsl.py` - *later deprecated*)
- Initial DSL planner logic in `agent.py` using function calling
- Tests for math operations and DSL execution

### Design Decisions

- Designed a simple DSL for common financial calculations (add, subtract, divide, percentage).
- Aimed to use LLM function-calling to generate DSL programs based on question and evidence.
- Offloaded execution to deterministic Python functions interpreting the DSL.

### Key Issues

- LLM output inconsistency: malformed JSON, incorrect function calls, extraneous text.
- DSL generation proved brittle; small changes in evidence or question phrasing led to invalid programs.
- Error handling for DSL execution was complex.

### Fixes

- Introduced Pydantic models for stricter validation of expected function call arguments (*partially successful*).
- Implemented regex fallbacks to extract DSL programs from malformed LLM responses.

### Outcome

Demonstrated feasibility of LLM generating calculation steps, but highlighted the fragility and complexity of the DSL approach. Tool execution success was inconsistent. **Decision:** Pivot agent design away from DSL generation towards a more robust two-step approach.

---

## P3 – Agent Architecture Pivot: Two-Step Python Expression

### Objective

Redesign the agent to improve robustness and simplify execution by using a two-step LLM process: 1) Specify a Python expression template and required variables, 2) Extract variable values from evidence.

### Deliverables

- `specify_and_generate_expression` function in `agent.py` (Step 1 LLM call).
- `extract_all_values_with_llm` function in `agent.py` (Step 2 LLM call).
- Updated `plan_and_execute` orchestrator to manage the two-step flow and execute the final Python expression using `eval()`.

### Design Decisions

- **Step 1 (Planning):** LLM determines calculation type, required items (e.g., "Net Sales 2021"), maps them to placeholders (VAL_1, VAL_2), and generates a standard Python expression template (e.g., `(VAL_1 - VAL_2) / VAL_2`). This simplifies the LLM's task compared to generating a full DSL program.
- **Step 2 (Extraction):** A separate LLM call focuses solely on extracting numerical values for the items identified in Step 1, applying scaling based on context (millions, %, etc.).
- **Execution:** Use Python's `eval()` in a controlled environment to execute the template populated with extracted values. This eliminates the need for a custom DSL interpreter.

### Key Issues

- Value Extraction Challenges: LLM sometimes failed to find the correct value, hallucinated values, or misinterpreted scaling units (e.g., millions, percentages).
- Prompt Engineering: Refining prompts for both steps to ensure accurate planning and reliable extraction required significant iteration.

### Fixes

- Added detailed instructions and examples to prompts, including negative constraints for extraction (e.g., don't extract values from descriptions of change).
- Implemented logic to handle `null` returns from the extraction step and fail gracefully.
- Added output formatting logic based on the plan from Step 1.

### Outcome

This two-step architecture proved significantly more robust and controllable than the DSL approach. Execution success rate improved, although extraction accuracy remained a challenge dependent on prompt quality and model capability.

---

## P4 – Evaluation Framework & Experiment Tracking

### Objective

Benchmark and monitor pipeline performance across samples using controlled metrics, and track experiments systematically.

### Deliverables

- `run_evaluation.py` CLI tool for evaluation.
- Metrics including execution accuracy, program template generation success (`substitute_failed` rate), and value extraction success.
- Evaluation logs in JSONL format for detailed traceability.
- Integration with **Weights & Biases (W&B)** for logging metrics, parameters (model names, prompt versions), and evaluation results.

### Design Decisions

- Evaluated each question individually against the pipeline.
- Captured key intermediate steps (plan, extracted values, final expression) alongside the final answer and gold standard.
- Used W&B to visualize accuracy trends across different configurations (models, prompts, retrieval strategies), aiding rapid iteration.
- Enabled sampling and split-based evaluation via CLI flags.

### Key Issues

- Defining reliable accuracy metrics (e.g., handling floating point differences, percentage formatting).
- Ensuring consistent environment setup for reproducible W&B runs.

### Fixes

- Implemented a flexible `answers_match` function handling numeric and percentage comparisons with tolerance.
- Documented environment variable usage (`WANDB_MODE=disabled`) for temporarily disabling W&B logging during debugging.

### Outcome

The evaluation framework coupled with W&B provided crucial visibility into performance bottlenecks and the impact of changes, enabling data-driven development and tuning.

---

## P5 – Streamlit Demo and UI Improvements

### Objective

Build a clear, interactive dashboard to demonstrate the RAG pipeline's reasoning and results, reflecting the latest architecture.

### Deliverables

- Streamlit app (`app.py`) with user input, global evidence retrieval, agent execution trigger, and results display.
- UI elements showing retrieved/reranked chunks, final answer, program template, and intermediate values.

### Design Decisions

- Shifted UI from sample-based interaction to **global search** to better reflect a real-world application scenario.
- Implemented a two-button flow: "Retrieve Evidence" followed by "Run Agent".
- Focused on explainability by displaying retrieved chunks (before/after reranking) and intermediate agent outputs (plan, extracted values, debug info).
- Used `st.metric` for prominent display of the final answer.

### Key Issues

- Initial attempts at global search highlighted poorer retrieval quality compared to page-scoped search, sometimes failing to find necessary data chunks.
- Ensuring smooth state management (`st.session_state`) between the retrieval and agent execution steps.

### Fixes

- Refined UI layout using columns and dividers for better readability.
- Added clear status messages (info, success, warning, error).

### Outcome

Fully functional demo showcasing the end-to-end global search RAG pipeline. Effectively demonstrated the two-step agent process and highlighted the challenges of global retrieval in this context.

---

## P6 – Vector Store Experiment: Pinecone

### Objective

Evaluate the use of a managed vector database (Pinecone) for potentially more scalable semantic retrieval compared to local BM25/embedding lookups, especially for future larger datasets.

### Deliverables

- Pinecone integration in `retriever.py` (*later removed*).
- CLI script for data upsert to Pinecone (*later adapted for ChromaDB*).
- Feature flag for switching retrieval backend.

### Design Decisions

- Selected Pinecone as a representative managed vector DB.
- Maintained BM25 as a fallback.

### Key Issues & Rationale for Removal

- **Pinecone Free Tier Limitations:** The primary blocker was the free tier's constraints on index size and the number of vectors. We could only index a small fraction (~570 records) of the `train.json` data, not the full dataset.
- **Inability to Test Global Search:** Due to the partial index, Pinecone could not access all necessary financial chunks during global search queries. This made it impossible to fairly evaluate or improve global retrieval performance using Pinecone under the project constraints.
- **Decision:** Given that evaluating and improving global search was a key goal, and the free tier prevented comprehensive indexing, Pinecone was removed. The focus shifted to a locally hosted vector store (ChromaDB) that allowed indexing the entire dataset.

### Outcome

Learned about the practical limitations of free tiers for vector databases on moderately sized datasets. Confirmed that partial indexing severely hinders the effectiveness and testability of global retrieval strategies. This justified the pivot to ChromaDB.

---

## P7 – Vector Store Implementation: ChromaDB & Performance Tuning

### Objective

Implement a persistent, locally hosted vector store using ChromaDB to enable full-dataset indexing and facilitate global search evaluation and tuning. Optimize the two-step agent pipeline with this setup.

### Deliverables

- ChromaDB integration in `retriever.py` (replacing Pinecone).
- Updated upsert script (`scripts/load_chroma_mpnet.py`) for ChromaDB.
- Iterative tuning of agent prompts (Step 1 & 2) and model selection (testing GPT-4o vs. GPT-4o-mini).
- W&B logs tracking performance with ChromaDB + two-step agent.

### Design Decisions

- Chose ChromaDB for its ease of local setup and persistence.
- **Embedding Model Selection:** Adopted the `all-mpnet-base-v2` Sentence Transformer model for generating embeddings. 
    - **Rationale:** This model offers a strong balance of performance (MTEB benchmarks) and local execution efficiency. Running embeddings locally avoided API costs and potential rate limits associated with embedding the entire dataset (~10k chunks) using external services like OpenAI's API, especially given the ~8 hour indexing time observed.
- Indexed the entire `train.json` dataset (~10,000+ vector records after chunking) to `./chroma_db_mpnet`.
- Continued refining the two-step agent, experimenting with different models (GPT-4o vs. GPT-4o-mini) for planning and extraction steps based on W&B results.
- Tuned retrieval parameters (`chromadb_query_top_k`, reranker `final_top_k`).

### Key Issues

- **Indexing Time:** Indexing the full dataset with embeddings took considerable time (~8 hours) on a laptop without GPU acceleration.
- **Global Retrieval Accuracy:** While ChromaDB allowed full indexing, achieving high accuracy on global search remained challenging. Retrieved chunks were often relevant but sometimes lacked the specific table/sentence containing the exact numerical value needed for extraction.
- **Agent Extraction Errors:** The extraction step (Step 2) continued to be a source of errors, requiring careful prompt engineering and model selection.

### Fixes

- Reverted model choices based on W&B experiments (e.g., finding GPT-4o didn't always improve planning and sometimes hurt extraction compared to mini, settling back on mini or a consistent model for both steps).
- Iteratively refined extraction prompts with scaling instructions and negative constraints.

### Outcome

Successfully implemented ChromaDB, enabling full-dataset indexing and proper testing of the global search pipeline. Achieved a peak **execution accuracy of ~62.5%** on the `train_sample20` evaluation set with the optimized ChromaDB + two-step agent configuration. Identified global retrieval and robust value extraction as key areas for future improvement.

---

## Experimental Enhancements Summary

- **Vector Store:** Transitioned from attempted Pinecone integration (limited by free tier) to successful **ChromaDB** implementation (allowing full dataset indexing).
- **Agent Architecture:** Shifted from a fragile DSL-based agent to a more robust **two-step LLM process** (Specify Expression -> Extract Values).
- **Structured Output Validation:** Used schema validation for LLM function calls (both for original DSL attempt and current two-step agent) to improve reliability.
- **Chat Memory:** Explored briefly but removed to simplify pipeline execution for this phase.
- **Experiment Tracking:** Leveraged **Weights & Biases (W&B)** throughout for systematic evaluation and tuning.

