import os
import sys

# Ensure src is on path for package imports
sys.path.insert(0, os.path.abspath("src"))

import json
import streamlit as st

from finrag.chunk_utils import build_candidate_chunks
from finrag.retriever import retrieve_evidence
from finrag.agent import plan_and_execute

def answers_match(pred, gold):
    try:
        pred_str = str(pred)
        gold_str = str(gold)
        pred_val = float(pred_str.replace('%',''))
        gold_val = float(gold_str.replace('%',''))
        # Determine decimal precision of gold answer
        if '.' in gold_str:
            decimals = len(gold_str.split('%')[0].split('.')[-1])
        else:
            decimals = 0
        # Round predicted to same precision and compare
        return round(pred_val, decimals) == gold_val
    except:
        return str(pred) == str(gold)

@st.cache_data
def load_data(split="dev"):
    path = os.path.join("data", f"{split}_turn.json")
    with open(path) as f:
        return json.load(f)

st.title("FinRAG: Financial Reasoning Assistant")
st.markdown("Ask a question based on financial reports about the ConvFinQA dataset")

# Sidebar options and sample questions
data = load_data()

# Initialize session state for selected sample and question
if "selected_sample_id" not in st.session_state and data:
    st.session_state["selected_sample_id"] = data[0].get("id")

# Quick-Pick Questions Sidebar
questions_file = os.path.join("scripts", "upsert_questions.jsonl")
if os.path.isfile(questions_file):
    with open(questions_file, "r", encoding="utf-8") as f:
        quick_qs = [json.loads(line) for line in f]
    st.sidebar.markdown("### Quick Questions")
    for i, item in enumerate(quick_qs):
        q = item.get("question", "")
        if st.sidebar.button(q, key=f"quick_{i}"):
            st.session_state["question_input"] = q

# Main Input Panel
question = st.text_area(
    "",
    key="question_input",
    height=100,
)

if st.button("Run"):
    if not question:
        st.error("Please enter a question.")
    else:
        # Retrieve and rerank
        sample = next((s for s in data if s.get("id") == st.session_state["selected_sample_id"]), None)
        if not sample:
            st.error("No sample selected for retrieval.")
        else:
            ret = retrieve_evidence(sample, question)
            raw_chunks = ret.get("raw_chunks", [])
            reranked_chunks = ret.get("reranked_chunks", [])

            with st.expander("🔎 Top-K Retrieved Chunks (Before Rerank)"):
                for i, ch in enumerate(raw_chunks):
                    st.markdown(f"**{i+1}.** `{ch['chunk_id']}` — {ch['text'][:300]}...")
            with st.expander("📊 Top-K Chunks After Rerank (Used by LLM)"):
                for i, ch in enumerate(reranked_chunks):
                    st.markdown(
                        f"**{i+1}.** `{ch['chunk_id']}` — {ch['text'][:300]}...  score: {ch['score']:.2f}"
                    )

            with st.spinner("Planning and executing..."):
                result = plan_and_execute(question, reranked_chunks)

            # Results
            st.subheader("Generated Program")
            st.code(result["program"])
            st.subheader("Intermediate Values")
            st.write(result["intermediates"])
            st.subheader("Final Answer")
            st.write(result["answer"])
            # Gold answer and comparison from selected sample
            gold_answer = sample.get("qa", {}).get("answer") or sample.get("answer")
            if gold_answer is not None:
                st.subheader("Gold Answer")
                st.write(gold_answer)
                if answers_match(result["answer"], gold_answer):
                    st.success("✅ Correct!")
                else:
                    st.error("❌ Incorrect.")
            gold_program = sample.get("qa", {}).get("program") or sample.get("program")
            if gold_program:
                st.subheader("DSL Program (Gold)")
                st.code(gold_program)
