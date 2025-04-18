import json
import pytest
import openai
import finrag.agent as agent
from finrag.agent import FUNCTION_NAME

class DummyMessage:
    def __init__(self):
        # Simulate LLM function_call attribute
        self.function_call = {
            "name": FUNCTION_NAME,
            "arguments": json.dumps({"program": "subtract(10,5), divide(#0,5)"}),
        }

class DummyChoice:
    def __init__(self):
        self.message = DummyMessage()

class DummyResponse:
    def __init__(self):
        self.choices = [DummyChoice()]

@ pytest.fixture(autouse=True)
def patch_openai(monkeypatch):
    def fake_create(model, messages, functions, function_call):
        return DummyResponse()
    monkeypatch.setattr(openai.ChatCompletion, "create", fake_create)


def test_generate_tool_call():
    question = "What is 10 minus 5, then divide by 5?"
    evidence = [{"chunk_id": "chunk1", "text": "irrelevant"}]
    tool_call = agent.generate_tool_call(question, evidence)
    assert tool_call["tool"] == FUNCTION_NAME
    assert tool_call["program"] == "subtract(10,5), divide(#0,5)"


def test_plan_and_execute(monkeypatch):
    # Stub generate_tool_call
    monkeypatch.setattr(agent, "generate_tool_call", lambda q, e: {"tool": FUNCTION_NAME, "program": "subtract(10,5), divide(#0,5)"})
    # Stub run_math_tool
    monkeypatch.setattr(agent, "run_math_tool", lambda inp: {"answer": "100.0%", "program": inp.program, "intermediates": [5, 1.0]})
    question = "Dummy?"
    evidence = [{"chunk_id": "chunk1", "text": "data"}]
    result = agent.plan_and_execute(question, evidence)
    assert result["answer"] == "100.0%"
    assert result["program"] == "subtract(10,5), divide(#0,5)"
    assert result["intermediates"] == [5, 1.0]
    assert result["tool"] == FUNCTION_NAME
    assert result["evidence"] == ["chunk1"]
