# FinRAG: Financial Reasoning Assistant

FinRAG is a Retrieval-Augmented Generation (RAG) system tailored for quantitative reasoning over financial reports, built on the ConvFinQA dataset. It uses a two-step LLM agent approach to answer complex numerical questions by:

1. Retrieving relevant evidence using ChromaDB vector search and Cohere reranking
2. Planning calculations through an LLM that specifies required values and generates Python expression templates
3. Extracting numerical values from evidence using a second LLM step that handles scaling and units
4. Executing the populated expressions to produce final answers

The system achieves ~62.5% accuracy on evaluation sets and provides full transparency into its reasoning process through detailed intermediate outputs.

## RAG Pipeline Overview

The following diagram illustrates the core workflow of FinRAG:

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

For a detailed explanation of the system architecture, data flow, and component interactions, please refer to the [Design & Architecture Document](docs/design_architecture.md).

## Models

### OPENAI_CHAT_MODEL

Default: `gpt-4o-mini` (or the latest recommended mini version)
Used by the LLM agent (`src/finrag/agent.py`) for the two-step process:
1.  **Specify Expression**: Determines required financial items, maps them to placeholders, generates a Python expression template, and specifies the output format.
2.  **Extract Values**: Extracts numerical values for required items from the evidence, applying scaling as needed.

Selected for its strong reasoning, instruction following, function-calling capabilities (though currently using Python execution), and excellent balance of cost, latency, and performance.

### EMBEDDING_MODEL

Default: `all-mpnet-base-v2`
Employed by `src/finrag/embeddings.py` using the `sentence-transformers` library to compute local semantic embeddings for text and table chunks stored in ChromaDB. This model runs locally and is chosen for its strong performance on semantic search tasks without requiring external API calls.

Note: Initial development phases utilised OpenAI's `text-embedding-ada-002` model before transitioning to the local `sentence-transformers` approach.

These model identifiers can be customised in your `.env` file (referencing `OPENAI_CHAT_MODEL` and `EMBEDDING_MODEL` variables) to experiment with different compatible models.

## Documentation
FinRAG's supplementary guides and logs are available in the `docs/` folder:

| Document                   | Path                                    | Description                                           |
|----------------------------|-----------------------------------------|-------------------------------------------------------|
| Streamlit UI Walkthrough   | [docs/streamlit_ui_walkthrough.md](docs/streamlit_ui_walkthrough.md) | Prototype streamlit interface guide with screenshots            |
| Product Requirements (PRD) | [docs/PRD.md](docs/PRD.md)              | Objectives, features, non-functional requirements     |
| Design & Architecture      | [docs/design_architecture.md](docs/design_architecture.md) | System design, component overview          |
| Engineering Log            | [docs/engineering_log.md](docs/engineering_log.md)         | Log of architecture, decisions, experiments, challenges and outcomes during development         |
| Evaluation                 | [docs/evaluation.md](docs/evaluation.md)                   | Evaluation strategy, metrics, sample outputs         |
| Test Suite Summary         | [docs/test_suite_summary.md](docs/test_suite_summary.md)   | Automated test overview: coverage, mocks, outputs     |
| Requirements & Setup       | [docs/requirements_setup.md](docs/requirements_setup.md)   | Setup guide: dependencies, env vars, directory layout |
| Production Planning        | [docs/productionisation_plan.md](docs/productionisation_plan.md) | Thoughts about scaling this prototype in to production  |



## Dataset
This project uses data from the ConvFinQA dataset, which provides conversational question-answer pairs over financial documents.

Download it from: [https://github.com/czyssrs/ConvFinQA](https://github.com/czyssrs/ConvFinQA)

## Example Questions

Here are a few examples of the types of financial reasoning questions FinRAG can handle:

- what is the percentage change in the total gross amount of unrecognized tax benefits from 2013 to 2014?
- what is the percentage change net provision for interest and penalties from 2015 to 2016?
- how much more return was given for investing in the overall market rather than applied materials from 2009 to 2014 ?

## Running Tests

```bash
poetry run pytest
```
See the [Test Suite Summary](docs/test_suite_summary.md) for further details about tests.

## Evaluation

```bash
poetry run python src/finrag/eval.py --split dev --sample 5
```
See the [Evaluation](docs/evaluation.md) section for details about the evaluation methodology.

---

## Accuracy Evolution (Approximate on `train_sample20`)

| Stage                                | Execution Accuracy | Program Match Rate | Tool Usage Rate | Notes                                                                 |
| ------------------------------------ | ------------------ | ------------------ | --------------- | --------------------------------------------------------------------- |
| Initial (P1/P2 - DSL Attempt)        | <10%               | <10%               | ~50-70%         | Highly unstable DSL generation                                        |
| Post Two-Step Agent Intro (P3)       | ~20%               | N/A (Template)     | ~95%+           | Improved stability, extraction issues surfaced                        |
| Tuned Two-Step Agent                 | ~50%               | N/A (Template)     | ~85%            | Peak before vector store changes, likely benefiting from page-context |
| ChromaDB + Tuned Agent (P7 - Global) | **~62.5%**         | N/A (Template)     | ~85%            | Peak accuracy with full indexing and global search                    |

*Note: Program Match Rate less relevant for template-based approach; focus shifted to execution accuracy & failure modes (specify vs. extract vs. eval).* 

---

## Key Findings 

- **Global vs. Scoped Retrieval:** Global retrieval is significantly harder than document-scoped retrieval. While ChromaDB enabled full indexing, finding the *exact* data chunk remains a challenge. Hybrid approaches fusing keyword/sparse search (like BM25) *with* dense vector search *before* reranking could be beneficial.
- **Agent Design:** The two-step agent (Plan -> Extract) is more robust than direct DSL generation for this task. However, the extraction step is sensitive to evidence quality and prompt details.
- **Value Extraction:** Reliably extracting and scaling numerical values (millions, %, etc.) is non-trivial and a major source of errors. More sophisticated parsing or targeted extraction models might be needed.
- **Chunking Strategy:** The current chunking (paragraphs, table rows) might not be optimal. Exploring different chunk sizes, overlap, or semantic chunking could improve retrieval quality.
- **Experiment Tracking:** W&B was invaluable for rapidly iterating on prompts, models, and retrieval parameters.
- **Trade-offs:** The project demonstrated the trade-offs between development speed, complexity (DSL vs. Python eval), performance, and the challenges inherent in open-domain financial QA within a limited timeframe.
- **Transparency:** Documenting the journey, including abandoned approaches (Pinecone, DSL) and performance fluctuations, provides a realistic view of the iterative AI development process.

---

## Shortcomings of the Current RAG Pipeline

Based on the development and evaluation conducted within the one-week timeframe, the current FinRAG pipeline exhibits several limitations:

1.  **Global Retrieval Precision:** While the pipeline successfully retrieves generally relevant documents using global ChromaDB search + Cohere reranking, it often fails to pinpoint the *single specific chunk* (especially tables or graphs) containing the exact numerical data required for the agent's extraction step. This leads to `substitute_failed` errors even when relevant documents are found.
2.  **Value Extraction Fragility:** The LLM-based value extraction step (Step 2) is prone to errors. It can struggle with accurately interpreting scaling units (millions, billions, %), hallucinating values not present in the evidence, or failing to find a value even if it exists in the retrieved chunks. This step is highly sensitive to prompt wording and the quality/clarity of the evidence provided.
3.  **Lack of True Hybrid Search Fusion:** The retriever uses ChromaDB primarily and BM25 only as a fallback in specific (currently unused in the UI) scenarios. It does not fuse results from both sparse and dense retrieval methods before reranking, potentially missing out on the benefits of combining keyword and semantic relevance signals early on.
4.  **Basic Chunking Strategy:** The current approach of chunking by paragraphs and table rows is relatively simple. More advanced strategies (e.g., smaller/overlapping chunks, recursive retrieval, proposition-based indexing) might improve the retriever's ability to isolate specific facts.
5.  **Performance Ceiling:** Despite extensive tuning within the week, the peak accuracy achieved (~62.5% on the sample set) indicates significant room for improvement. Addressing the retrieval and extraction shortcomings would be necessary to push performance higher.
6.  **Scalability Concerns (Indexing Time):** While ChromaDB enabled full dataset indexing locally, the ~8-hour indexing time highlights potential challenges for significantly larger datasets or scenarios requiring frequent updates without GPU acceleration.

