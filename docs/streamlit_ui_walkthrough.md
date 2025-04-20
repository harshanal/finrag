
# FinRAG – Streamlit UI Walkthrough

This section provides a visual and functional overview of the FinRAG Streamlit demo application. The interface is designed for exploring financial question-answering with transparency into how the system retrieves context, generates reasoning programs, and produces final answers.

---

## Purpose of the Demo

- Demonstrate a working LLM-driven QA system on financial reports
- Make internal steps (retrieval, reranking, DSL generation, execution) transparent

---

## UI Screenshots

### 1. Initial View – Landing Interface

- **Title**: "FinRAG: Financial Reasoning Assistant"
- **Subtitle**: Describes ConvFinQA dataset context
- **Input Field**: Allows user to enter or select a financial question
- **Page Selector / Global Search Toggle**

> 📷 _Example Screenshot: `landing_page.png`_

---

### 2. Retrieval Stage

- Shows:
  - Top-k retrieved chunks before reranking (BM25 + Embeddings)
  - Reranked chunks (via Cohere)
  - Document source and type (text/table/row)

> 📷 _Example Screenshot: `retrieval_chunks.png`_

---

### 3. Generated Program & Execution

- Displays the generated DSL program (e.g., `add(30, 36)`)
- Shows the final executed answer (e.g., `66`)
- **Optional**: Comparison with gold answer and program if available

> 📷 _Example Screenshot: `execution_and_result.png`_

---

### 4. Accuracy Indicators

- Visual status for:
  - ✅ DSL program matched gold
  - ✅ Execution result matches gold answer
  - 🔁 Tool was used correctly

---

## Interactive Flow

1. User selects or types a question
2. System retrieves and reranks relevant evidence
3. LLM generates a DSL program
4. Tool executor runs the program
5. UI presents:
   - Retrieved evidence
   - Generated reasoning chain
   - Final answer and gold comparison



## 📁 Screenshots Folder

All relevant screenshots are stored in `/docs/screenshots/` and referenced here.

To add new screenshots:

1. Run the Streamlit app
2. Capture screen regions using your OS tools
3. Save as `.png` in `/docs/screenshots/`
4. Update image references in this markdown

---

For a live walkthrough, launch the app locally:

```bash
poetry run streamlit run app.py
```

