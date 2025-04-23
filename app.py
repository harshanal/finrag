import os
import sys

# Ensure src is on path for package imports
sys.path.insert(0, os.path.abspath("src"))

import json
import streamlit as st

# Assuming these imports are correct based on your project structure
try:
    # from finrag.chunk_utils import build_candidate_chunks # Not needed directly here
    from finrag.retriever import retrieve_evidence
    from finrag.agent import plan_and_execute
except ImportError as e:
    st.error(f"Failed to import FinRAG modules: {e}. Ensure src is in PYTHONPATH.")
    st.stop()

# --- Removed answers_match function (no gold comparison) ---
# --- Removed load_data function (no sample loading) ---

st.set_page_config(layout="wide") # Use wider layout
st.title("FinRAG: Global Search & Reasoning Assistant")
st.markdown("Ask a question to search across the indexed financial reports.")

# --- Sidebar Removed --- 

# --- Sample Selection Removed ---

# --- Quick Questions Removed ---

# --- Main Panel (No Columns) ---

# Initialize session state for chunks if they don't exist
if 'raw_chunks' not in st.session_state:
    st.session_state['raw_chunks'] = []
if 'reranked_chunks' not in st.session_state:
    st.session_state['reranked_chunks'] = []
if 'agent_result' not in st.session_state:
    st.session_state['agent_result'] = None

st.subheader("Ask a Question")
question = st.text_area(
        "Enter your question here:",
    key="question_input",
    height=100,
        label_visibility="collapsed" # Hide the label above
)

# Button to trigger retrieval
if st.button("Retrieve Evidence (Global Search)", use_container_width=True):
    st.session_state['raw_chunks'] = [] # Clear previous results
    st.session_state['reranked_chunks'] = []
    st.session_state['agent_result'] = None
    if not question:
        st.error("Please enter a question.")
    else:
        st.info(f"Performing global search for evidence...")
        with st.spinner("Retrieving evidence..."):
            # Pass empty dict {} for 'turn' to trigger global search
            ret = retrieve_evidence({}, question) 
        # Store results in session state
        st.session_state['raw_chunks'] = ret.get("raw_chunks", [])
        st.session_state['reranked_chunks'] = ret.get("reranked_chunks", [])
        
        if not st.session_state['reranked_chunks']:
             st.warning("No relevant evidence chunks found after reranking. The agent may not be able to answer.")
        else:
             st.success(f"{len(st.session_state['reranked_chunks'])} evidence chunks retrieved and reranked.")

# --- Display Retrieval Results (using session state) ---
if st.session_state['raw_chunks'] or st.session_state['reranked_chunks']:
    st.markdown("### Retrieved Evidence")
    with st.expander("🔎 Top Retrieved Chunks (Before Rerank)"):
        raw_chunks = st.session_state['raw_chunks']
        if raw_chunks:
            for i, ch in enumerate(raw_chunks):
                st.markdown(f"**{i+1}.** `{ch.get('chunk_id', 'N/A')}`")
                st.text(f"{ch.get('text', '')[:500]}...") # Show more text
        else:
            st.write("No raw chunks retrieved.")

    with st.expander("📊 Top Reranked Chunks (Used by LLM)"):
        reranked_chunks = st.session_state['reranked_chunks']
        if reranked_chunks:
            for i, ch in enumerate(reranked_chunks):
                st.markdown(f"**{i+1}.** `{ch.get('chunk_id', 'N/A')}` — Score: {ch.get('score', 0):.2f}")
                st.text(f"{ch.get('text', '')[:500]}...") # Show more text
        else:
            st.write("No chunks after reranking.")
    
    # --- START DEBUG: Show reranked chunks passed to agent ---
    with st.expander("🐛 DEBUG: Reranked Chunks Input to Agent"):
        st.json(st.session_state['reranked_chunks'])
    # --- END DEBUG ---

# --- Conditional Button and Logic for Agent Execution ---
if st.session_state['reranked_chunks']:
    if st.button("Run Agent with Evidence", use_container_width=True):
        st.session_state['agent_result'] = None # Clear previous agent result
        st.info("Running agent with retrieved evidence...")
        with st.spinner("Planning and executing..."):
            # Use question from input and chunks from session state
            current_question = st.session_state.get("question_input", "") 
            reranked_chunks = st.session_state['reranked_chunks']
            result = plan_and_execute(current_question, reranked_chunks)
            st.session_state['agent_result'] = result # Store agent result

# --- Display Agent Results (if available) ---
if st.session_state.get('agent_result'):
    st.markdown("### Agent Results")
    result = st.session_state['agent_result']
    # --- START DEBUG: Show raw result from plan_and_execute ---
    st.subheader("🐛 DEBUG: Raw Agent Result")
    st.json(result)
    # --- END DEBUG ---
    
    # Display results
    st.subheader("📝 Generated Program")
    st.code(result.get("program", "N/A"), language="python") # Assume DSL is python-like
    st.subheader("🔢 Intermediate Values")
    st.write(result.get("intermediates", []))
    st.subheader("💡 Final Answer")
    st.write(result.get("answer", "No answer generated."))

    # --- Gold Answer Comparison Removed ---

# --- Column 2 Removed ---
