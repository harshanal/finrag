import os
import sys

# Ensure src is on path for package imports
sys.path.insert(0, os.path.abspath("src"))

import json
import streamlit as st

# Assuming these imports are correct based on your project structure
try:
    # from finrag.chunk_utils import build_candidate_chunks # Not directly used in app.py itself
from finrag.retriever import retrieve_evidence
from finrag.agent import plan_and_execute
except ImportError as e:
    st.error(f"Failed to import FinRAG modules: {e}. Ensure src is in PYTHONPATH.")
    st.stop()

def answers_match(pred, gold):
    try:
        pred_str = str(pred)
        gold_str = str(gold)
        pred_val = float(pred_str.replace('%',''))
        gold_val = float(gold_str.replace('%',''))
        # Determine decimal precision of gold answer
        if '.' in gold_str:
            # Handle cases like '14.1%' -> '14.1' -> ['14', '1'] -> len=1
            relevant_part = gold_str.split('%')[0]
            if '.' in relevant_part:
                 decimals = len(relevant_part.split('.')[-1])
            else:
                 decimals = 0
        else:
            decimals = 0
        # Round predicted to same precision and compare
        return round(pred_val, decimals) == gold_val
    except Exception: # Catch broader exceptions during conversion/comparison
        return str(pred) == str(gold) # Fallback to string comparison

@st.cache_data
def load_data(split="train", max_entries: int | None = None): # Modified function signature, default split to train
    """Loads data from the specified split, optionally limiting entries."""
    path = os.path.join("data", f"{split}.json") # Use the actual filename format
    try:
        with open(path, 'r', encoding='utf-8') as f: # Added encoding
            loaded_data = json.load(f)
            # Slice the data if max_entries is specified
            if max_entries is not None and max_entries > 0:
                st.sidebar.info(f"Loading first {min(max_entries, len(loaded_data))} entries.") # Inform user
                return loaded_data[:max_entries]
            else:
                return loaded_data
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {path}")
        return []
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {path}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred loading {path}: {e}")
        return []

st.set_page_config(layout="wide") # Use wider layout
st.title("FinRAG: Financial Reasoning Assistant")
st.markdown("Ask a question based on financial reports (ConvFinQA dataset)")

# --- Sidebar Configuration --- Added section ---
st.sidebar.title("⚙️ Configuration")

# --- START MODIFICATION: Remove Data Split Selection ---
# Remove the dropdown for selecting data split
# data_split = st.sidebar.selectbox(
#     "Select Data Split",
#     ("dev", "train", "test"), 
#     index=0, 
#     help="Choose the dataset split to load."
# )
data_split = "train" # Hardcode to always use train split
st.sidebar.info("Loading data from: data/train.json") # Inform user
# --- END MODIFICATION ---

# Add number input for max entries
max_data_entries = st.sidebar.number_input(
    f"Max Entries to Load from '{data_split}' (0=All)", # Label still reflects the hardcoded split
    min_value=0,  # 0 means load all
    value=0,      # Default to loading all
    step=50,
    help="Limit the number of data samples loaded for faster testing. Set to 0 to load all."
)

# Use the value from number input (treat 0 as None for load_data)
max_entries_to_load = max_data_entries if max_data_entries > 0 else None
# Pass the hardcoded split and limit to load_data
data = load_data(split=data_split, max_entries=max_entries_to_load)

if not data:
    # --- START MODIFICATION: Update Error Message Filename ---
    st.error(f"No data loaded from 'data/{data_split}.json'. Check configuration or data file.")
    # --- END MODIFICATION ---
    st.stop() # Stop execution if no data

# Sample Selection Dropdown
sample_options = {s.get("id", f"Entry_{i}"): i for i, s in enumerate(data)} # Use index as fallback ID
selected_sample_id = st.sidebar.selectbox(
    "Select Sample ID",
    options=list(sample_options.keys()),
    index=0, # Default to first sample in the (potentially limited) list
    help="Choose a specific data sample to focus the question on."
)
# Update session state if needed (though direct use might be simpler here)
# st.session_state["selected_sample_id"] = selected_sample_id
selected_sample_index = sample_options[selected_sample_id]
sample = data[selected_sample_index] # Get the currently selected sample

# --- End Sidebar Configuration ---


# Quick-Pick Questions Sidebar - Consider loading only if file exists
st.sidebar.markdown("---") # Separator
st.sidebar.markdown("### Quick Questions")
questions_file = os.path.join("scripts", "upsert_questions.jsonl")
quick_qs = []
if os.path.isfile(questions_file):
    try:
    with open(questions_file, "r", encoding="utf-8") as f:
        quick_qs = [json.loads(line) for line in f]
    except Exception as e:
        st.sidebar.warning(f"Could not load quick questions: {e}")

if quick_qs:
    for i, item in enumerate(quick_qs):
        q = item.get("question", "")
        # Use columns for better layout if many questions
        if st.sidebar.button(q, key=f"quick_{i}"):
            st.session_state["question_input"] = q
            # Automatically select the sample associated with the quick question, if available
            q_sample_id = item.get("id")
            if q_sample_id in sample_options:
                # This part might not work as expected if the sample ID isn't in the loaded 'data' subset
                # A better approach might be to ensure quick questions only appear if their sample ID is loaded
                # Or, trigger a reload if a quick question for an unloaded sample is clicked (more complex)
                st.sidebar.info(f"Selected sample ID: {q_sample_id}")
                # Force rerun doesn't seem necessary here as selectbox default will be updated by state?
                # Or maybe directly set selected_sample_id = q_sample_id here if selectbox update isn't reliable
else:
    st.sidebar.write("No quick questions found.")

# --- Main Panel ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Ask a Question")
question = st.text_area(
        "Enter your question here:",
    key="question_input",
    height=100,
        label_visibility="collapsed" # Hide the label above
)

    if st.button("Run Analysis", use_container_width=True):
    if not question:
        st.error("Please enter a question.")
        elif not sample:
             st.error("No sample selected for retrieval (this shouldn't happen).")
        else:
            st.info(f"Running analysis for Sample ID: {selected_sample_id}...")
            # Retrieve and rerank using the selected sample
            with st.spinner("Retrieving evidence..."):
                ret = retrieve_evidence(sample, question) # Pass the selected sample
            raw_chunks = ret.get("raw_chunks", [])
            reranked_chunks = ret.get("reranked_chunks", [])

            with st.expander("🔎 Top Retrieved Chunks (Before Rerank)"):
                 if raw_chunks:
                for i, ch in enumerate(raw_chunks):
                         st.markdown(f"**{i+1}.** `{ch.get('chunk_id', 'N/A')}`")
                         st.text(f"{ch.get('text', '')[:500]}...") # Show more text
                 else:
                    st.write("No raw chunks retrieved.")

            with st.expander("📊 Top Reranked Chunks (Used by LLM)"):
                 if reranked_chunks:
                for i, ch in enumerate(reranked_chunks):
                         st.markdown(f"**{i+1}.** `{ch.get('chunk_id', 'N/A')}` — Score: {ch.get('score', 0):.2f}")
                         st.text(f"{ch.get('text', '')[:500]}...") # Show more text
                 else:
                     st.write("No chunks after reranking.")

            if not reranked_chunks:
                st.warning("No relevant evidence chunks found after reranking. The agent may not be able to answer.")
                # Optionally stop here or let the agent try with no evidence
                # st.stop()
            
            # --- START DEBUG: Show reranked chunks passed to agent ---
            with st.expander("🐛 DEBUG: Reranked Chunks Input to Agent"):
                 st.json(reranked_chunks)
            # --- END DEBUG ---

            with st.spinner("Planning and executing..."):
                result = plan_and_execute(question, reranked_chunks)

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

            # Gold answer and comparison from selected sample
            gold_answer = sample.get("qa", {}).get("answer") or sample.get("answer")
            st.subheader("🏆 Gold Answer")
            if gold_answer is not None:
                st.write(gold_answer)
                if answers_match(result.get("answer"), gold_answer):
                    st.success("✅ Correct!")
                else:
                    st.error("❌ Incorrect.")
            else:
                st.write("N/A")

            gold_program = sample.get("qa", {}).get("program") or sample.get("program")
            st.subheader("🥇 DSL Program (Gold)")
            if gold_program:
                st.code(gold_program, language="python")
            else:
                 st.write("N/A")


with col2:
    st.subheader(f"Selected Sample Details (ID: {selected_sample_id})")
    if sample:
        st.json(sample, expanded=False) # Show the JSON of the selected sample collapsed by default
    else:
        st.write("No sample selected.")
