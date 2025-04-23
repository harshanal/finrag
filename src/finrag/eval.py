import argparse
import json
import logging
import os
import random
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import finrag.logging_conf  # configure global logging
import logging

# --- W&B Import ---
import wandb
# --- End W&B Import ---

from finrag.agent import plan_and_execute
from finrag.chunk_utils import build_candidate_chunks
from finrag.retriever import retrieve_evidence


def evaluate_agent(data: List[Dict[str, Any]], use_retrieval: bool = False, strict: bool = False) -> Dict[str, Any]:
    # Group samples into conversations by stripping turn suffix
    conv_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in data:
        sid = sample.get("id", "")
        conv_id = sid.rsplit("_", 1)[0]
        conv_groups[conv_id].append(sample)
    total = sum(len(v) for v in conv_groups.values())
    processed = 0
    exec_correct = 0
    prog_match = 0
    tool_used = 0
    failures: List[Any] = []
    examples: List[Dict[str, Any]] = []

    # Iterate through each conversation
    for conv_id, samples in conv_groups.items():
        # --- START MODIFICATION: Robust Turn Sorting ---
        def get_turn_index(sample_dict):
            sample_id = sample_dict.get("id", "")
            try:
                # Try splitting by hyphen first (e.g., ...pdf-3 -> 3)
                return int(sample_id.rsplit("-", 1)[-1])
            except (ValueError, IndexError):
                try:
                    # Fallback: Try splitting by underscore (e.g., ..._3 -> 3)
                    return int(sample_id.rsplit("_", 1)[-1])
                except (ValueError, IndexError):
                    # If both fail, log warning and return 0 for sorting
                    warnings.warn(f"Could not parse turn index from ID '{sample_id}' for conv '{conv_id}'. Using default sort order 0.")
                    return 0

        try:
            turns = sorted(samples, key=get_turn_index)
        except Exception as sort_err: # Catch any unexpected sorting errors
            warnings.warn(f"Sorting turns failed unexpectedly for conv '{conv_id}': {sort_err}. Processing in original order.")
            turns = samples
        # --- END MODIFICATION: Robust Turn Sorting ---

        chat_history: List[Tuple[str, str]] = []
        for sample in turns:
            sample_id = sample.get("id")
            qa = sample.get("qa", {})
            question = qa.get("question")
            gold_answer = qa.get("answer")
            gold_program = qa.get("program")
            # Skip samples with missing QA fields
            if question is None or gold_answer is None or gold_program is None:
                msg = f"Skipping sample {sample_id}: missing question/answer/program"
                if strict:
                    raise ValueError(msg)
                warnings.warn(msg)
                continue
            processed += 1
            # DEBUG: show sample keys
            print(f"DEBUG_SAMPLE[{sample_id}]: keys={list(sample.keys())}")
            # DEBUG: show QA keys and values
            print(f"DEBUG_SAMPLE[{sample_id}]: QA={sample.get('qa')}")
            # build candidate chunks and debug
            all_chunks = build_candidate_chunks(sample)
            logging.debug(f"Sample {sample_id} gold_inds: {sample.get('gold_inds')}")
            logging.debug(f"Candidate chunk_ids: {[c['chunk_id'] for c in all_chunks]}")
            print(f"DEBUG_SAMPLE[{sample_id}]: candidate chunk_ids={[c['chunk_id'] for c in all_chunks]}")
            if use_retrieval:
                # New retrieve_evidence returns dict with raw and reranked chunks
                ret = retrieve_evidence(sample, question=question, top_k=10, bm25_k=10)
                # Use reranked_chunks as evidence
                evidence_chunks = ret.get("reranked_chunks", [])
            else:
                # gold_inds are indices into candidate_chunks
                gold_inds = sample.get("gold_inds", []) or []
                print(f"DEBUG_SAMPLE[{sample_id}]: gold_inds={gold_inds}")
                if gold_inds:
                    evidence_chunks = []
                    for idx in gold_inds:
                        try:
                            evidence_chunks.append(all_chunks[int(idx)])
                        except Exception as e:
                            print(f"DEBUG_SAMPLE[{sample_id}]: invalid gold index {idx}, error={e}")
                else: # If no gold_inds, use all candidate chunks
                    evidence_chunks = all_chunks
            # Handle missing evidence
            if not evidence_chunks:
                warnings.warn(f"No evidence chunks available for sample {sample_id} after initial selection.")
                # Fallback to retrieval pipeline if not already using retrieval
                if not use_retrieval:
                    ret = retrieve_evidence(sample, question=question, top_k=10, bm25_k=10)
                    evidence_chunks = ret.get("reranked_chunks", [])
                    logging.debug(f"Sample {sample_id} retrieval fallback chunk_ids: {[c['chunk_id'] for c in evidence_chunks]}")
                # If still no evidence, skip this sample
                if not evidence_chunks:
                    warnings.warn(f"No evidence chunks available after retrieval for sample {sample_id}. Skipping sample.")
                    continue
            try:
                # Pass chat_history into planner for multi-turn context
                result = plan_and_execute(question, evidence_chunks, chat_history)
                tool = result.get("tool")
                # count only valid tool calls - update to check for the new tool name
                if tool == "python_eval":
                    tool_used += 1
                answer = result.get("answer")
                program = result.get("program")
                if answer == gold_answer:
                    exec_correct += 1
                else:
                    failures.append(sample_id)
                if program == gold_program:
                    prog_match += 1
                examples.append({
                    "id": sample_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "agent_answer": answer,
                    "gold_program": gold_program,
                    "program": program,
                    "tool": tool,
                    "evidence": [c["chunk_id"] for c in evidence_chunks],
                })
                # Update history for next turn
                chat_history.append((question, answer))
            except Exception as e:
                failures.append(sample_id)
                examples.append({"id": sample_id, "error": str(e)})

    return {
        "total": processed,
        "execution_accuracy": exec_correct / processed if processed else 0,
        "program_match_rate": prog_match / processed if processed else 0,
        "tool_usage_rate": tool_used / processed if processed else 0,
        "failures": failures,
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--strict", action="store_true", help="Raise on invalid samples")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--retrieval", action="store_true")
    # --- W&B: Add argument for run name/notes (optional) ---
    parser.add_argument("--wandb_notes", type=str, default=None, help="Optional notes for the W&B run.")
    parser.add_argument("--wandb_tags", nargs='+', default=[], help="Optional tags for the W&B run (e.g., --wandb_tags baseline markdown_chunking).")
    parser.add_argument("--wandb_project", type=str, default="finrag-eval", help="W&B project name.")
    # --- End W&B Arguments ---
    args = parser.parse_args()

    # --- W&B: Initialize Run --- 
    try:
        run = wandb.init(
            project=args.wandb_project,
            config=vars(args), # Log all argparse arguments
            notes=args.wandb_notes,
            tags=args.wandb_tags,
            reinit=True, # Allows running multiple evals in one script if needed later
            job_type="evaluation"
        )
        print(f"W&B Run initialized: {run.url}")
    except Exception as e:
        print(f"Error initializing W&B: {e}. Tracking disabled for this run.")
        run = None # Set run to None if init fails
    # --- End W&B Init ---

    # configure logging for verbose debug
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Construct data path using the split argument, removing the '_turn' suffix assumption
    data_path = os.path.join(os.getcwd(), "data", f"{args.split}.json")
    print(f"Attempting to load data from: {data_path}") # Add print statement for debugging
    if not os.path.isfile(data_path):
        # --- W&B: Log error and finish if file not found ---
        error_msg = f"Data file not found: {data_path}"
        print(f"Error: {error_msg}")
        if run:
             wandb.log({"error": error_msg})
             run.finish(exit_code=1)
        # --- End W&B Error Log ---
        raise FileNotFoundError(error_msg)

    # Ensure correct encoding is specified when opening
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
         # --- W&B: Log error and finish if file load fails ---
        error_msg = f"Failed to load or parse data file {data_path}: {e}"
        print(f"Error: {error_msg}")
        if run:
             wandb.log({"error": error_msg})
             run.finish(exit_code=1)
        # --- End W&B Error Log ---
        raise # Re-raise the exception

    if args.sample and args.sample > 0 and args.sample < len(data):
        print(f"Sampling {args.sample} random entries from the data.") # Clarify sampling
        data = random.sample(data, args.sample)
    elif args.sample:
        print(f"Warning: Sample size {args.sample} is invalid or >= total entries {len(data)}. Processing all {len(data)} entries.")

    print(f"Starting evaluation with {len(data)} samples...")
    try:
        report = evaluate_agent(data, use_retrieval=args.retrieval, strict=args.strict)
        total = report["total"]
        def pct(x: float) -> str:
            return f"{x*100:.2f}%"

        print(f"Total samples processed: {total}")
        print(f"Execution Accuracy: {pct(report['execution_accuracy'])}")
        print(f"Program Match Rate: {pct(report['program_match_rate'])}")
        print(f"Tool Usage Rate: {pct(report['tool_usage_rate'])}")
        print(f"Failures: {len(report['failures'])}")

        # --- W&B: Log Metrics ---
        if run:
            wandb.log({
                "total_samples_processed": total,
                "execution_accuracy": report['execution_accuracy'],
                "program_match_rate": report['program_match_rate'],
                "tool_usage_rate": report['tool_usage_rate'],
                "failure_count": len(report['failures']),
                "samples_in_split_used": len(data) # Log how many samples were actually used (after sampling)
            })
        # --- End W&B Log Metrics ---

        out_dir = "outputs"
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"eval_{args.split}{f'_sample{args.sample}' if args.sample else ''}_{ts}.jsonl"
        log_file_path = os.path.join(out_dir, log_file_name)
        with open(log_file_path, "w") as f:
            for ex in report["examples"]:
                f.write(json.dumps(ex))
                f.write("\n")
        print(f"Full log: {log_file_path}")

        # --- W&B: Log Output File as Artifact ---
        if run:
            try:
                artifact = wandb.Artifact(f'eval_results_{args.split}', type='evaluation-results')
                artifact.add_file(log_file_path)
                run.log_artifact(artifact)
                print(f"Evaluation results artifact uploaded to W&B.")
            except Exception as e:
                print(f"Warning: Failed to log artifact to W&B: {e}")
        # --- End W&B Log Artifact ---

    except Exception as e:
        print(f"\n--- Evaluation failed with error: {e} ---")
        logging.error("Evaluation failed", exc_info=True)
         # --- W&B: Log error and finish if evaluation fails ---
        if run:
            wandb.log({"error": str(e)})
            run.finish(exit_code=1)
        # --- End W&B Error Log ---
        # Optionally re-raise or handle differently
        # raise e 

    finally:
        # --- W&B: Finish Run ---
        if run:
            run.finish()
        # --- End W&B Finish ---

if __name__ == "__main__":
    main()
