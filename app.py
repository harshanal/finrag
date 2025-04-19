import os
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.header("Options")
split = st.sidebar.selectbox("Split", ["dev"])
use_retrieval = st.sidebar.checkbox("Use Retrieval", value=False)
data = load_data(split)

sample_ids = [s.get("id") for s in data]
selected = st.sidebar.selectbox("Select Example", ["None"] + sample_ids)
if selected and selected != "None":
    sample = next((s for s in data if s.get("id") == selected), {})
    qa = sample.get("qa", {})
    question = qa.get("question") or sample.get("question", "")
    gold_answer = qa.get("answer") or sample.get("answer", "[No gold answer available]")
    gold_program = qa.get("program") or sample.get("program")
else:
    sample = None
    question = st.text_area("Enter your financial question:")
    gold_answer = None
    gold_program = None

if st.sidebar.button("Clear Conversation"):
    st.session_state.chat_history = []

# Display past turns
if st.session_state.chat_history:
    st.subheader("Conversation History")
    for idx, (q, a) in enumerate(st.session_state.chat_history, start=1):
        st.markdown(f"**Q{idx}:** {q}")
        st.markdown(f"**A{idx}:** {a}")

# Run current turn
if st.button("Run"):
    if not question:
        st.error("Please enter a question or select an example.")
    else:
        if sample:
            all_chunks = build_candidate_chunks(sample)
            if use_retrieval:
                ids = retrieve_evidence(sample, question)
                evidence_chunks = [c for c in all_chunks if c["chunk_id"] in ids]
            else:
                gold_inds = sample.get("gold_inds", []) or []
                if gold_inds:
                    evidence_chunks = [all_chunks[i] for i in gold_inds if i < len(all_chunks)]
                else:
                    evidence_chunks = all_chunks
        else:
            st.error("Please select an example to use evidence retrieval.")
            evidence_chunks = []

        st.subheader("Question")
        st.write(question)

        with st.expander("Evidence"):
            for c in evidence_chunks:
                st.write(f"ID: {c['chunk_id']}")
                st.write(c["text"])

        with st.spinner("Planning and executing..."):
            result = plan_and_execute(question, evidence_chunks, st.session_state.chat_history)
            # Append this turn to history
            st.session_state.chat_history.append((question, result["answer"]))

        st.subheader("Generated Program")
        st.code(result["program"])

        st.subheader("Intermediate Values")
        st.write(result["intermediates"])

        st.subheader("Final Answer")
        st.write(result["answer"])

        st.subheader("Gold Answer")
        st.write(gold_answer if gold_answer is not None else "[No gold answer available]")
        if gold_answer is not None:
            if answers_match(result["answer"], gold_answer):
                st.success("✅ Correct!")
            else:
                st.error("❌ Incorrect.")
        if gold_program:
            st.subheader("DSL Program (Gold)")
            st.code(gold_program)
