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

from finrag.agent import plan_and_execute, FUNCTION_NAME
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
        # Sort turns by turn index from id suffix
        turns = sorted(samples, key=lambda s: int(s.get("id", "").rsplit("_",1)[1]))
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
                # count only valid tool calls
                if tool == FUNCTION_NAME:
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
    args = parser.parse_args()

    # configure logging for verbose debug
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    data_path = os.path.join(os.getcwd(), "data", f"{args.split}_turn.json")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    with open(data_path) as f:
        data = json.load(f)

    if args.sample and args.sample < len(data):
        data = random.sample(data, args.sample)

    report = evaluate_agent(data, use_retrieval=args.retrieval, strict=args.strict)
    total = report["total"]
    def pct(x: float) -> str:
        return f"{x*100:.2f}%"

    print(f"Total samples: {total}")
    print(f"Execution Accuracy: {pct(report['execution_accuracy'])}")
    print(f"Program Match Rate: {pct(report['program_match_rate'])}")
    print(f"Tool Usage Rate: {pct(report['tool_usage_rate'])}")
    print(f"Failures: {len(report['failures'])}")

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(out_dir, f"eval_{args.split}_{ts}.jsonl")
    with open(log_file, "w") as f:
        for ex in report["examples"]:
            f.write(json.dumps(ex))
            f.write("\n")
    print(f"Full log: {log_file}")


if __name__ == "__main__":
    main()
