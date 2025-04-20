import os
import sys
import math

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

st.title("FinRAG Interactive Demo")

# Sidebar options and sample questions
data = load_data()

# Initialize session state for selected sample and question
if "selected_sample_id" not in st.session_state and data:
    st.session_state["selected_sample_id"] = data[0].get("id")
if "question_input" not in st.session_state and data:
    st.session_state["question_input"] = data[0].get("qa", {}).get("question", "")

# Sample Questions Sidebar
raw_qas = [(s.get("id"), s.get("qa", {}).get("question")) for s in data if s.get("qa", {}).get("question")]
seen = set(); unique_qas = []
for sid, q in raw_qas:
    if q not in seen:
        seen.add(q); unique_qas.append((sid, q))
unique_qas = unique_qas[:20]
st.sidebar.markdown("### Sample Questions")
for i, (sid, q) in enumerate(unique_qas):
    col1, col2 = st.sidebar.columns([5,1])
    col1.markdown(f"- {q}")
    if col2.button("📋", key=f"copy_q_{i}"):
        st.session_state["selected_sample_id"] = sid
        st.session_state["question_input"] = q

# Main Input Panel
st.subheader("Your Question")
question = st.text_area(
    "Enter your financial question:",
    value=st.session_state["question_input"],
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
