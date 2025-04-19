import os
import os
import sys

# Ensure src is on path for package imports
sys.path.insert(0, os.path.abspath("src"))

import json
import streamlit as st

from finrag.chunk_utils import build_candidate_chunks
from finrag.retriever import retrieve_evidence
from finrag.agent import plan_and_execute

@st.cache_data
def load_data(split="dev"):
    path = os.path.join("data", f"{split}_turn.json")
    with open(path) as f:
        return json.load(f)

st.title("FinRAG Interactive Demo")

st.sidebar.header("Options")
split = st.sidebar.selectbox("Split", ["dev"])
use_retrieval = st.sidebar.checkbox("Use Retrieval", value=False)
data = load_data(split)

sample_ids = [s["id"] for s in data]
selected = st.sidebar.selectbox("Select Example", ["None"] + sample_ids)
if selected != "None":
    sample = next((s for s in data if s["id"] == selected), None)
    question = sample["qa"]["question"]
else:
    sample = None
    question = st.text_area("Enter your financial question:")

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
            result = plan_and_execute(question, evidence_chunks)

        st.subheader("Generated Program")
        st.code(result["program"])

        st.subheader("Intermediate Values")
        st.write(result["intermediates"])

        st.subheader("Final Answer")
        st.write(result["answer"])
