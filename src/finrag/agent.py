"""Agent module for FinRAG."""

import json
import openai
from typing import Any, Dict, List
from finrag.tools import MathToolInput, run_math_tool

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

def answer(question: str, conversation_history: List[Dict], doc_id: str) -> Dict:
    """Answer a question given history and document ID."""
    raise NotImplementedError

def generate_tool_call(question: str, evidence_chunks: List[Dict[str, str]]) -> Dict[str, str]:
    """Use LLM function calling to generate a math tool call from question and evidence."""
    evidence_text = "\n".join(
        f"{i+1}. [{chunk['chunk_id']}] {chunk['text']}" for i, chunk in enumerate(evidence_chunks)
    )
    messages = [
        {"role": "system", "content": "You are an agent that decides which tool to call."},
        {
            "role": "user",
            "content": (
                f"Question: {question}\nRelevant evidence:\n{evidence_text}\n"
                "Emit a function call."
            ),
        },
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=messages,
        functions=FUNCTION_DEFINITIONS,
        function_call={"name": FUNCTION_NAME},
    )
    message = response.choices[0].message
    if not getattr(message, "function_call", None):
        raise ValueError("No function_call returned by LLM")
    func_call = message.function_call
    args_str = func_call.get("arguments") or "{}"
    params = json.loads(args_str)
    program = params.get("program")
    return {"tool": func_call["name"], "program": program}

def plan_and_execute(question: str, evidence_chunks: List[Dict[str, str]]) -> Dict[str, Any]:
    """Plan with the LLM to generate a tool call and execute it."""
    tool_call = generate_tool_call(question, evidence_chunks)
    if tool_call["tool"] != FUNCTION_NAME:
        raise ValueError(f"Unsupported tool: {tool_call['tool']}")
    math_input = MathToolInput(program=tool_call["program"])
    result = run_math_tool(math_input)
    return {
        "answer": result["answer"],
        "program": result["program"],
        "intermediates": result["intermediates"],
        "tool": tool_call["tool"],
        "evidence": [chunk["chunk_id"] for chunk in evidence_chunks],
    }
