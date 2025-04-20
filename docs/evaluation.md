
# FinRAG – Evals

This document presents a structured analysis of the FinRAG system’s performance across its different retrieval and execution pipelines. It highlights metrics, test scenarios, error patterns, and interpretable examples to support iterative improvements.

---

## Evaluation Objectives

- Measure the effectiveness of the pipeline in extracting, reasoning over, and answering financial questions.
- Compare performance across versions (MVP1: page-based, MVP2: global search, MVP3: Pinecone).
- Understand and document system limitations through examples and metrics.

---

## Metric Definitions

| Metric | Definition |
|--------|------------|
| **Execution Accuracy** | Percentage of QA pairs where the system's final numerical answer matches the gold answer. |
| **Program Match Rate** | Percentage of turns where the generated DSL program matches the gold reasoning program. |
| **Tool Usage Rate** | Frequency at which the planner correctly calls the execution tool instead of failing silently. |
| **Recall@k** | Whether the retrieved evidence includes the gold-referenced content within the top-k candidates. Used in retrieval benchmarking. |

Execution and program matching were computed with rounding tolerance (e.g., ±0.1) for numeric results and string normalization for programs.

---

## Evaluation Artifacts

All evaluations were performed using CLI and notebook-based experiments.

- 🔗 [retrieval_benchmark.ipynb](../notebooks/retrieval_benchmark.ipynb): Calculates Recall@k metrics across BM25 + embedding fusion and rerank variants.
- 🔗 [program_accuracy_eval.ipynb](../notebooks/program_accuracy_eval.ipynb): Measures DSL matching precision between gold and generated programs.
- 🔗 [execution_accuracy_eval.ipynb](../notebooks/execution_accuracy_eval.ipynb): Measures numeric correctness of the executed answers across dev samples.

Logs are stored in: `outputs/eval_dev_TIMESTAMP.jsonl` for traceable inspection.

---

## Performance Comparison

| Version | Retrieval Type | Exec Accuracy | Prog Match | Tool Usage | Recall@5 |
|---------|----------------|---------------|------------|------------|-----------|
| MVP1    | Page-selected  | 21.6%         | 45%        | 100%       | 85%       |
| MVP2    | Global + ReRank| 0%–5%         | 10–15%     | 100%       | 60–70%    |
| MVP3    | Pinecone       | < 10%         | 25%        | 100%       | ~55%      |

> Note: Pinecone indexing limited to 570 chunks due to service constraints.

---

## Error Analysis

### Typical Failure Cases

1. **Ambiguous Retrieval**
   - Global retriever often misses specific year-based table rows or includes irrelevant text.
2. **Incorrect Program Inference**
   - LLM occasionally reverses arguments or skips required DSL steps.
3. **Formatting Mismatches**
   - Answers match semantically but differ in decimal precision or formatting (e.g., "66.0" vs "66").
4. **Context Window Overflow**
   - Too many candidates in long documents leads to context truncation during planning.

---

## Examples

### Example 1 – Correct Execution

**Question:** _"What is the percentage increase in operating cash flow from 2008 to 2009?"_

**DSL:** `subtract(14.1, 12.3), divide(#0, 12.3)`

**Answer:** `14.6%` ✅

**Top Evidence:** Chunk with table for 2008–2009 cash flow rows.

---

### Example 2 – Incorrect Program

**Question:** _"What is the total of 2006 and 2007 deductions?"_

**Gold DSL:** `add(30, 36)`  
**Generated DSL:** `subtract(36, 30)` ❌

**Root Cause:** Model misinterpreted "total" as difference.

---

## Logs and Notebook Results

Evaluation logs are available in `outputs/` and plotted in:

- 📓 [Streamlit Logs Visualizer](../notebooks/streamlit_logs_demo.ipynb)
- 📓 [RAG Pipeline Evaluation](../notebooks/rag_eval_pipeline.ipynb)

---

For further model comparison or reproducibility, clone the repo and run:

```bash
poetry run python src/finrag/eval.py --split dev --sample 50 --verbose
```
