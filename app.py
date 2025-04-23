import os
import sys

# Ensure src is on path for package imports
sys.path.insert(0, os.path.abspath("src"))

import json
import streamlit as st

try:
    from finrag.retriever import retrieve_evidence
    from finrag.agent import plan_and_execute
except ImportError as e:
    st.error(f"Failed to import FinRAG modules: {e}. Ensure src is in PYTHONPATH.")
    st.stop()

st.set_page_config(layout="wide", page_title="FinRAG Assistant") # Set page title
st.title("📈 FinRAG: Financial Reasoning Assistant")
st.markdown("Ask a question to search across the indexed financial reports from the ConvFinQA dataset.")

# --- Main Panel (No Columns) ---

# Initialize session state for chunks if they don't exist
if 'raw_chunks' not in st.session_state:
    st.session_state['raw_chunks'] = []
if 'reranked_chunks' not in st.session_state:
    st.session_state['reranked_chunks'] = []
if 'agent_result' not in st.session_state:
    st.session_state['agent_result'] = None

# --- Question Input --- 
st.subheader("❓ Ask a Question")
question = st.text_area(
    "Enter your question here:",
    key="question_input",
    height=100,
    label_visibility="collapsed" 
)

# --- Retrieve Button --- 
if st.button("🔍 Retrieve Evidence (Global Search)", use_container_width=True, type="primary"):
    st.session_state['raw_chunks'] = [] # Clear previous results
    st.session_state['reranked_chunks'] = []
    st.session_state['agent_result'] = None
    if not question:
        st.error("Please enter a question.")
    else:
        st.info(f"Performing global search for evidence...")
        with st.spinner("Retrieving evidence..."):
            ret = retrieve_evidence({}, question) 
        st.session_state['raw_chunks'] = ret.get("raw_chunks", [])
        st.session_state['reranked_chunks'] = ret.get("reranked_chunks", [])
        
        if not st.session_state['reranked_chunks']:
             st.warning("No relevant evidence chunks found after reranking.")
        else:
             st.success(f"✅ {len(st.session_state['reranked_chunks'])} evidence chunks retrieved and reranked.")

st.markdown("--- ") # Divider

# --- Display Retrieval Results --- 
if st.session_state['raw_chunks'] or st.session_state['reranked_chunks']:
    st.subheader("📄 Retrieved Evidence")
    with st.expander("Top Retrieved Chunks (Before Rerank)"):
        raw_chunks = st.session_state['raw_chunks']
        if raw_chunks:
            for i, ch in enumerate(raw_chunks):
                score_str = f" (Score: {ch.get('score', 0):.2f})" if ch.get('score') is not None else ""
                st.markdown(f"**{i+1}.** `{ch.get('chunk_id', 'N/A')}`{score_str}")
                st.text(f"{ch.get('text', '')[:500].strip()}...") 
        else:
            st.write("No raw chunks retrieved.")

    with st.expander("Top Reranked Chunks (Used by LLM)"):
        reranked_chunks = st.session_state['reranked_chunks']
        if reranked_chunks:
            for i, ch in enumerate(reranked_chunks):
                score_str = f" (Score: {ch.get('score', 0):.2f})" if ch.get('score') is not None else ""
                st.markdown(f"**{i+1}.** `{ch.get('chunk_id', 'N/A')}`{score_str}")
                st.text(f"{ch.get('text', '')[:500].strip()}...")
        else:
            st.write("No chunks after reranking.")
    
    # --- START DEBUG: Show reranked chunks passed to agent ---
    # with st.expander("🐛 DEBUG: Reranked Chunks Input to Agent"):
    #     st.json(st.session_state['reranked_chunks'])
    # --- END DEBUG ---

# --- Conditional Agent Button --- 
if st.session_state['reranked_chunks']:
    st.markdown("--- ") # Divider
    if st.button("🤖 Run Agent with Evidence", use_container_width=True):
        st.session_state['agent_result'] = None # Clear previous agent result
        st.info("Running agent with retrieved evidence...")
        with st.spinner("Planning and executing..."):
            current_question = st.session_state.get("question_input", "") 
            reranked_chunks = st.session_state['reranked_chunks']
            result = plan_and_execute(current_question, reranked_chunks)
            st.session_state['agent_result'] = result 

# --- Display Agent Results --- 
if st.session_state.get('agent_result'):
    st.markdown("--- ") # Divider
    st.subheader("💡 Agent Results")
    result = st.session_state['agent_result']
    
    # Display final answer more prominently
    final_answer = result.get("answer", "No answer generated.")
    if "Error" in final_answer or "failed" in result.get("program", ""): 
        st.error(f"**Final Answer:** {final_answer}")
    else:
        try:
            # Attempt to create a metric display if answer looks numeric/percentage
            label = "Final Answer" 
            st.metric(label=label, value=str(final_answer))
        except:
            st.success(f"**Final Answer:** {final_answer}") # Fallback for non-metric answers

    # Use columns for other results to save vertical space
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown("**📝 Generated Program (Template)**")
        st.code(result.get("program", "N/A"), language="python") 
        
        # Optionally show DEBUG info in an expander
        with st.expander("🐛 DEBUG: Raw Agent Result"):
            st.json(result)
    
    with res_col2:
        st.markdown("**🔢 Intermediate Values**")
        intermediates = result.get("intermediates", [])
        if intermediates:
            st.json(intermediates) # Display intermediates as JSON
        else:
            st.write("N/A")



