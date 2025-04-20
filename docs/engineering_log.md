# FinRAG – Engineering Log

## Overview

This log documents the technical evolution of the FinRAG prototype, aligned with key milestones outlined in the product requirements document (PRD). Each section details decisions made, challenges faced, and lessons learned across the development cycle.

---

## P1 – Retrieval MVP

### Objective

Develop a hybrid retriever combining lexical (BM25) and semantic (OpenAI embeddings) retrieval, enhanced with reranking via Cohere.

### Deliverables

- `retriever.py` with hybrid search logic
- Chunking and ID tagging via `chunk_utils.py`
- Evaluation notebook for top-k recall

### Design Decisions

- Combined top-k BM25 and embedding-based retrieval to offset the weakness of either method individually.
- Integrated Cohere as a reranking layer on fused retrieval results.
- Split text by paragraph and tables by row; tagged all chunks with unique identifiers.

### Key Issues

- Semantic-only retrieval initially returned low-recall results.
- Errors in chunk ID generation caused ambiguous or duplicate mappings.

### Fixes and Enhancements

- Developed unit tests to evaluate recall on a fixed dev subset.
- Normalized token processing in BM25 and refined rerank cutoff thresholds.

### Outcome

Achieved recall above 80% on benchmark. Provided a reliable base for context-aware DSL generation.

---

## P2 – Tool Schema and Executor

### Objective

Formalise a DSL execution engine and integrate tool calling via OpenAI’s function-calling API.

### Deliverables

- DSL interpreter (`calculator.py`, `dsl.py`)
- DSL planner logic in `agent.py`
- Tests for math operations, formatting, and execution chaining

### Design Decisions

- Designed a simple DSL for unary and binary operations (add, subtract, divide, percentage).
- Used LLM to generate tool instructions and offloaded execution to deterministic Python functions.
- Structured DSL validation using a tool schema before execution.

### Key Issues

- Inconsistent LLM outputs included malformed JSON, extra formatting, and irregular function\_call usage.
- Errors in percentage formatting led to rounding mismatches.

### Fixes

- Introduced `format_percentage` for clean and consistent result formatting.
- Added structured parsing using Pydantic models to validate tool instructions prior to execution.
- Integrated a regex-based fallback mechanism to recover DSL program strings when structured output was unavailable.
- This two-layered strategy significantly reduced malformed program errors and increased execution success rates.

### Outcome

Tool execution succeeded for all supported DSL operations.&#x20;

---

## P3 – Agent Orchestration

### Objective

Combine retriever, planner, and executor into a unified LLM-driven orchestration flow.

### Deliverables

- `plan_and_execute()` orchestrator function
- Prompt engineering for few-shot DSL generation
- Unit and integration tests for end-to-end QA

### Design Decisions

- Applied chain-of-thought prompting to encourage step-wise tool plan generation.
- Integrated regex fallback in case function\_call fields were missing.
- Logged full trace including evidence, LLM response, and executed answer.

### Key Issues

- LLM occasionally omitted or misformatted tool call responses.
- Multiline tool plans failed to parse due to syntax errors.

### Fixes

- Enforced `temperature=0` in all OpenAI calls.
- Added structured parsing using Pydantic models to validate tool instructions.

### Outcome

Stable tool generation with 100% usage success rate on test samples. Improved program match rate and trace transparency.

**LLM prompt developed through iterative refinement:**\
\
IMPORTANT: Reply ONLY with a JSON function\_call using the provided schema; do NOT include any explanations or plain text.

You are an expert financial analyst AI. Your task is to answer questions based on provided evidence by generating a program for a simple math DSL.&#x20;

Follow these steps carefully:

1\. Understand the Question: Identify the core calculation needed (e.g., sum, difference, percentage change).

2\. Analyze Evidence: Scan the provided evidence text snippets (e.g., '[c1] Some text with numbers 123 and 456') to find the specific numerical values required by the question. Pay close attention to units (e.g., millions, thousands) and time periods (e.g., 2015, 2016).

&#x20;  \- Ensure numbers extracted account for units (e.g., if evidence says '2.2 billion' and other numbers are in millions, use 2200).

3\. Select Arguments: Extract the \*exact\* numerical values from the evidence that correspond to the concepts in the question. DO NOT use years (like 2015) as calculation arguments unless the question specifically asks for calculations on the years themselves. Use the numbers \*associated\* with those years in the evidence.

4\. Formulate DSL Program: Construct a program using ONLY the allowed DSL functions: \`add\`, \`subtract\`, \`multiply\`, \`divide\`. The program must be a flat sequence of comma-separated calls, like \`func1(arg1, arg2), func2(#0, arg3)\`. Use \`#N\` to reference the result of the N-th previous step (0-indexed).&#x20;

&#x20;  \- Example Percentage Change (New - Old) / Old: \`subtract(NewValue, OldValue), divide(#0, OldValue)\`

&#x20;  \- Example Sum: \`add(Value1, Value2), add(#0, Value3)\`

5\. Output Format: Respond ONLY with a JSON object containing the function call, like \`{"function\_call": {"name": "run\_math\_tool", "arguments": "{\\"program\\": \\"YOUR\_DSL\_PROGRAM\_HERE\\"}"}}\`. Do not include any other text, explanations, or markdown formatting.

Constraints:

\- DO NOT nest function calls (e.g., \`divide(subtract(a,b), b)\` is INVALID).

\- Use only numbers found/derived from the evidence (after adjusting for units) or simple constants (like 100) as arguments.

\- Ensure the program logically answers the question posed.

---

## P4 – Evaluation Framework

### Objective

Benchmark and monitor pipeline performance across samples using controlled metrics.

### Deliverables

- `eval.py` CLI tool for evaluation
- Metrics including execution accuracy, program match rate, and tool usage
- Evaluation logs in JSONL format for traceability

### Design Decisions

- Evaluated each conversational turn individually.
- Captured and logged retrieved context, generated DSL, and execution results.
- Enabled sampling and split-based evaluation using flags.

### Key Issues

- Generated programs occasionally used the wrong inputs due to weak evidence retrieval.
- Execution accuracy lagged behind program match rate.

### Fixes

- Adjusted prompt templates to clarify required references.
- Strengthened program parsing to prevent malformed execution.

### Outcome

Tool usage rate stabilized above 97%. Program match rate reached approximately 45%. Execution accuracy improved incrementally to over 21%.

---

## P5 – Streamlit Demo and UI Improvements

### Objective

Build a clear, interactive dashboard to demonstrate model reasoning and results.

### Deliverables

- Streamlit app with user input, evidence preview, DSL display, and final answer
- Support for top-k retrieval panels before and after reranking



### Design Decisions

- Emphasised explainability by exposing retrieval context and toolchain steps.



  Simplified UI labels for improved readability and interaction.

### Fixes

- Improved formatting of titles and subtitles.
- Added comparison against gold answer where available.

### Outcome

Fully functional demo enabling step-through analysis of QA behavior. Enhanced user understanding of the LLM-planning-execution pipeline.

---

## P6.1 – Pinecone Integration

### Objective

Introduce scalable semantic retrieval using a vector database service (Pinecone) and ensure fallback resilience.

### Deliverables

- Pinecone integration in `retriever.py`
- CLI script for data upsert to vector store
- Feature flag for hybrid retrieval mode

### Design Decisions

- Selected Pinecone as a managed vector database to enable semantic search across financial document chunks.
- Integrated a fallback path using BM25 for resilience if Pinecone is unavailable.
- Capped uploads to 570 records to stay within the free tier usage limits.

### Key Issues

- Pinecone-based retrieval did not improve accuracy, potentially due to limited dataset size and document overlap. Additionally, the full dataset could not be uploaded due to the constraints of the Pinecone free-tier plan, which limited the number of vector inserts and total index size. As a result, only 570 records were included in the index, preventing the retriever from having access to all the necessary financial chunks during inference and evaluation.
- Misconfigured chunk IDs caused accidental overwriting during embedding uploads.

### Fixes

- Introduced unique chunk ID suffixes using UUID to prevent collisions.
- Logged uploads to a file for auditing and reproducibility.
- Implemented warning and exception handling for empty or failed responses from Pinecone.

### Outcome

Pipeline supported vector-based semantic retrieval, but effectiveness was limited under the current dataset constraints. BM25 fallback ensured consistent behavior across failures.


---

## P6.2 – Loading data to vector store

**Objective**  
Build a repeatable CLI to embed and upsert data chunks into Pinecone with traceability.

**Design Rationale**  
- Parameterize record limits (`--max-records`) to control costs.  
- Log question metadata for each chunk to audit.  
- Prevent ID collisions in Pinecone.

**Implementation Notes**  
- In `scripts/upsert_to_pinecone.py`, added CLI args `--max-records` and `--question-log`.  
- Utilised `EmbeddingStore` for batched text embeddings.

**Challenges & Resolutions**  
- **Overwritten vectors**: identical IDs caused silent overwrites.  
  *Fix:* Suffix IDs with random UUID.  
---


## Experimental Enhancements

### Pinecone Integration

- Enabled via feature flag.
- Used for semantic vector search across chunks.
- Accuracy benefits were limited due to small dataset size.

### Structured Output via Pydantic

- Used to validate LLM-generated tool instructions.
- Reduced invalid program execution and improved robustness.

### Chat Memory for Multi-turn QA

- Maintained chat history across dialogue turns (but chat history was subsequently removed to reduce the  complexity of pipeline execution)

---

## Accuracy Evolution

| Stage                     | Execution Accuracy | Program Match Rate | Tool Usage Rate |
| ------------------------- | ------------------ | ------------------ | --------------- |
| Initial (P1)              | 0%                 | 0%                 | \~50%           |
| Post-Retrieval Fix        | \~17%              | \~35%              | \~90%           |
| After Parser Improvements | \~21.6%            | \~45%              | 97–100%         |

---

## Key Learnings

- Strong retrieval quality is essential for accurate DSL execution.
- Structured validation of LLM outputs reduces downstream execution errors.
- Tool-based reasoning frameworks improve transparency and debugging.
- A debug-oriented UI helps expose model and system behavior effectively.
  - The UI was enhanced during the development process to add the top\_k records before and after reranking. Also displayed program generated vs gold standard program from data. 
- Modular feature flags aid in testing experimental features without disrupting core functionality.
  - (e.g. feature flag for inclusion of Pinecone to the RAG pipeline)

---

_Note: Throughout this project, I employed AI tools to streamline different phases. OpenAI o3 model assisted with early project scoping, requirement extraction, and documentation workflows. GPT-o4-mini-high was integrated via Windsurf IDE to accelerate iterative development and code review. All strategic decisions, testing, debugging, and architecture were led and verified by me to ensure quality and coherence._

_Using these tools allowed me to move quickly without sacrificing depth. I was able to experiment with multiple architectural paths (page-based vs global search vs vector store) and evolve the pipeline over several iterations in just a few days. The assistants helped convert complex ideas into code faster, enabling me to focus more time on evaluation, metric design, and error analysis. They also supported test-driven development, automating repetitive tasks and generating scaffolding code like BM25 ranking utilities and DSL validators. The ability to offload boilerplate implementation work made space for deeper reasoning, rapid UI integration, and higher-level design decisions within time constraints._
