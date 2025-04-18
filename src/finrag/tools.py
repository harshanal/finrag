"""LLM and external tool wrappers for FinRAG."""

import os
from typing import Any, Dict

import cohere
import openai
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

openai.api_key = OPENAI_API_KEY
co = cohere.Client(COHERE_API_KEY)


def llm_completion(prompt: str, **kwargs: Any) -> str:
    """Call OpenAI completion API."""
    response = openai.ChatCompletion.create(
        model=kwargs.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=kwargs.get("temperature", 0),
    )
    return response.choices[0].message.content


class MathToolInput(BaseModel):
    program: str


def run_math_tool(input: MathToolInput) -> Dict[str, Any]:
    from finrag.dsl import execute_program
    out = execute_program(input.program)
    return {"answer": out["formatted"], "intermediates": out["intermediates"], "program": input.program}
