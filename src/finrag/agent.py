"""
########################################################################
# FinRAG Agent Module (agent.py)
#
# This module implements the "planning and execution" stage of the RAG pipeline:
# 1. It accepts reranked evidence chunks and a user question.
# 2. It formats the evidence into prompts for the LLM's function-calling API.
# 3. generate_tool_call() constructs and validates a DSL program that computes
#    the answer by invoking the math tool.
# 4. plan_and_execute() orchestrates the tool call: it invokes generate_tool_call(),
#    executes the returned DSL via run_math_tool(), captures intermediate values,
#    formats the final answer, and returns a structured dict containing:
#    - program: the DSL string
#    - intermediates: the numeric steps
#    - answer: the final formatted result
#
# Main functions:
# - generate_tool_call(question, evidence_chunks, chat_history=None):
#     Builds the function-calling JSON payload for the LLM based on question & evidence.
# - plan_and_execute(question, evidence_chunks, chat_history=None):
#     Runs generate_tool_call, executes the returned program, handles errors & fallbacks,
#     and composes the final output.
########################################################################
"""

"""Agent module for FinRAG."""

import json
import openai
import warnings
import re
from typing import Any, Dict, List, Tuple, Optional
from finrag.tools import MathToolInput, run_math_tool
import logging

# logger for debugging
logger = logging.getLogger(__name__)

FUNCTION_NAME = "run_math_tool"
FUNCTION_DEFINITIONS = [
    {
        "name": FUNCTION_NAME,
        "description": "Run a math program using DSL",
        "parameters": {
            "type": "object",
            "properties": {
                "program": {"type": "string", "description": "DSL program string"},
            },
            "required": ["program"],
        },
    }
]

def is_valid_dsl(program: str) -> bool:
    """Validate DSL program string format."""
    pattern = r'^[a-zA-Z_]+\([^)]*\)(,\s*[a-zA-Z_]+\([^)]*\))*$'
    return bool(re.fullmatch(pattern, program.strip()))

def answer(question: str, conversation_history: List[Dict], doc_id: str) -> Dict:
    """Answer a question given history and document ID."""
    raise NotImplementedError

def generate_tool_call(
    question: str,
    evidence_chunks: List[Dict[str, str]],
    chat_history: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, str]:
    """Use LLM function calling to generate a math tool call from question and evidence."""
    # format evidence for prompt
    evidence_text = "\n".join(
        f"{i+1}. [{chunk['chunk_id']}] {chunk['text']}" for i, chunk in enumerate(evidence_chunks)
    )
    # system prompt
    system_prompt = """IMPORTANT: Reply ONLY with a JSON function_call using the provided schema; do NOT include any explanations or plain text.
You are an expert financial analyst AI. Your task is to answer questions based on provided evidence by generating a program for a simple math DSL. 
Follow these steps carefully:
1. Understand the Question: Identify the core calculation needed (e.g., sum, difference, percentage change).
2. Analyze Evidence: Scan the provided evidence text snippets (e.g., '[c1] Some text with numbers 123 and 456') to find the specific numerical values required by the question. Pay close attention to units (e.g., millions, thousands) and time periods (e.g., 2015, 2016).
   - Ensure numbers extracted account for units (e.g., if evidence says '2.2 billion' and other numbers are in millions, use 2200).
3. Select Arguments: Extract the *exact* numerical values from the evidence that correspond to the concepts in the question. DO NOT use years (like 2015) as calculation arguments unless the question specifically asks for calculations on the years themselves. Use the numbers *associated* with those years in the evidence.
4. Formulate DSL Program: Construct a program using ONLY the allowed DSL functions: `add`, `subtract`, `multiply`, `divide`. The program must be a flat sequence of comma-separated calls, like `func1(arg1, arg2), func2(#0, arg3)`. Use `#N` to reference the result of the N-th previous step (0-indexed). 
   - Example Percentage Change (New - Old) / Old: `subtract(NewValue, OldValue), divide(#0, OldValue)`
   - Example Sum: `add(Value1, Value2), add(#0, Value3)`
5. Output Format: Respond ONLY with a JSON object containing the function call, like `{"function_call": {"name": "run_math_tool", "arguments": "{\"program\": \"YOUR_DSL_PROGRAM_HERE\"}"}}`. Do not include any other text, explanations, or markdown formatting.
Constraints:
- DO NOT nest function calls (e.g., `divide(subtract(a,b), b)` is INVALID).
- Use only numbers found/derived from the evidence (after adjusting for units) or simple constants (like 100) as arguments.
- Ensure the program logically answers the question posed.
"""
    # few-shot demonstrations
    # Example 1: simple percentage change
    demo1_user = (
        "Question: What is the percentage change from 50 to 75?\n"
        "Relevant evidence:\n1. [c1] Value was 50 in 2020.\n2. [c2] Value was 75 in 2021.\n"
        "Emit ONLY a function call."
    )
    demo1_assist = {"role": "assistant", "function_call": {"name": FUNCTION_NAME, "arguments": json.dumps({"program": "subtract(75, 50), divide(#0, 50)"})}}
    # Example 2: basis point variation
    demo2_user = (
        "Question: What is the basis point variation in margin from 20% to 25%?\n"
        "Relevant evidence:\n1. [c1] Margin was 20% in 2020.\n2. [c2] Margin was 25% in 2021.\n"
        "Emit ONLY a function call."
    )
    demo2_assist = {"role": "assistant", "function_call": {"name": FUNCTION_NAME, "arguments": json.dumps({"program": "subtract(25, 20), multiply(#0, 100)"})}}
    # real user query
    real_user = {"role": "user", "content": f"Question: {question}\nRelevant evidence:\n{evidence_text}\nEmit ONLY a function call."}
    # Build chat messages: system, history (if any), few-shot demos, then current query
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        for idx, (q, a) in enumerate(chat_history):
            messages.append({"role": "user", "content": f"Q{idx+1}: {q}"})
            messages.append({"role": "assistant", "content": f"A{idx+1}: {a}"})
    messages += [
        {"role": "user", "content": demo1_user},
        demo1_assist,
        {"role": "user", "content": demo2_user},
        demo2_assist,
        real_user,
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=messages,
        functions=FUNCTION_DEFINITIONS,
        function_call={"name": FUNCTION_NAME},
        temperature=0,
    )
    message = response.choices[0].message
    # debug function_call extraction (printed)
    func_call = getattr(message, "function_call", None)
    print("DEBUG: RAW function_call:", func_call)
    # Case 1: No function_call returned, attempt fallback on message.content
    if not func_call:
        logger.warning("No function_call returned by LLM, attempting fallback on content.")
        raw_content = getattr(message, 'content', None)
        matches = re.findall(r"[a-zA-Z_]+\([^)]*\)", raw_content or "")
        if matches:
            program = ", ".join(matches)
            logger.warning(f"Fallback regex extracted program from content: {program}")
            return {"tool": FUNCTION_NAME, "program": program}
        logger.warning("Fallback on content failed.")
        # Heuristic numeric fallback based on evidence_text and question
        nums = re.findall(r"\d+\.?\d*", evidence_text)
        numbers = [float(x) if '.' in x else int(x) for x in nums]
        q_lower = question.lower()
        # Percentage change fallback
        if ("percentage" in q_lower or "growth rate" in q_lower) and len(numbers) >= 2:
            new, old = numbers[-1], numbers[-2]
            program = f"subtract({new}, {old}), divide(#0, {old})"
            logger.warning(f"Heuristic percentage fallback: {program}")
            return {"tool": FUNCTION_NAME, "program": program}
        # Average fallback
        if "average" in q_lower and len(numbers) > 0:
            count = len(numbers)
            parts = []
            # Sum sequence
            if count >= 2:
                parts.append(f"add({numbers[0]}, {numbers[1]})")
                for val in numbers[2:]:
                    parts.append(f"add(#0, {val})")
            parts.append(f"divide(#0, {count})")
            program = ", ".join(parts)
            logger.warning(f"Heuristic average fallback: {program}")
            return {"tool": FUNCTION_NAME, "program": program}
        # No heuristic match
        return {"tool": "none", "program": ""}
    # Case 2: function_call exists, proceed with argument parsing and validation
    args_str = func_call.get("arguments") or "{}"
    print("DEBUG: ARGS STRING:", args_str)
    try:
        params = json.loads(args_str)
    except Exception:
        logger.warning("Failed to parse arguments; attempting fallback.")
        program = ""
    else:
        program = params.get("program", "")
    # Validate program string and fallback if needed
    if not program or not is_valid_dsl(program):
        logger.warning(f"Program is invalid or empty: '{program}'. Applying regex fallback.")
        matches = re.findall(r"[a-zA-Z_]+\([^)]*\)", args_str)
        if matches:
            program = ", ".join(matches)
            logger.warning(f"Fallback regex extracted program from args_str: {program}")
        else:
            logger.warning("Fallback failed — no DSL extracted.")
            # Heuristic numeric fallback based on evidence_text and question
            nums = re.findall(r"\d+\.?\d*", evidence_text)
            numbers = [float(x) if '.' in x else int(x) for x in nums]
            q_lower = question.lower()
            # Percentage change fallback
            if ("percentage" in q_lower or "growth rate" in q_lower) and len(numbers) >= 2:
                new, old = numbers[-1], numbers[-2]
                program = f"subtract({new}, {old}), divide(#0, {old})"
                logger.warning(f"Heuristic percentage fallback: {program}")
                return {"tool": FUNCTION_NAME, "program": program}
            # Average fallback
            if "average" in q_lower and len(numbers) > 0:
                count = len(numbers)
                parts = []
                # Sum sequence
                if count >= 2:
                    parts.append(f"add({numbers[0]}, {numbers[1]})")
                    for val in numbers[2:]:
                        parts.append(f"add(#0, {val})")
                parts.append(f"divide(#0, {count})")
                program = ", ".join(parts)
                logger.warning(f"Heuristic average fallback: {program}")
                return {"tool": FUNCTION_NAME, "program": program}
            # No heuristic match
            return {"tool": "none", "program": ""}
    return {"tool": func_call.get("name", "none"), "program": program}

def plan_and_execute(
    question: str,
    evidence_chunks: List[Dict[str, str]],
    chat_history: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Plan with the LLM to generate a tool call and execute it."""
    tool_call = generate_tool_call(question, evidence_chunks, chat_history)
    tool = tool_call.get("tool", "none")
    program = tool_call.get("program", "")
    # If no valid tool or empty program, return no-op
    if tool != FUNCTION_NAME or not program:
        return {
            "answer": "",
            "program": program,
            "intermediates": [],
            "tool": tool,
            "evidence": [chunk["chunk_id"] for chunk in evidence_chunks],
        }
    # Valid math execution
    math_input = MathToolInput(program=program)
    result = run_math_tool(math_input)
    return {
        "answer": result["answer"],
        "program": result["program"],
        "intermediates": result["intermediates"],
        "tool": tool,
        "evidence": [chunk["chunk_id"] for chunk in evidence_chunks],
    }
