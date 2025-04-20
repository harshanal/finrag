import json
import pytest
import openai
import finrag.agent as agent
from finrag.agent import FUNCTION_NAME
import logging

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

@pytest.fixture(autouse=True)
def patch_openai(monkeypatch):
    # stub ChatCompletion.create to accept any kwargs
    def fake_create(*args, **kwargs):
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


def test_generate_tool_call_regex_fallback(monkeypatch, caplog):
    import openai
    import finrag.agent as agent
    caplog.set_level(logging.WARNING)
    class DummyMessageNoFunc:
        def __init__(self):
            self.content = "Result: subtract(20,4) divide(#0,2)"
    class DummyChoiceNoFunc:
        def __init__(self):
            self.message = DummyMessageNoFunc()
    class DummyResponseNoFunc:
        def __init__(self):
            self.choices = [DummyChoiceNoFunc()]
    # Monkeypatch create to return no function_call
    monkeypatch.setattr(openai.ChatCompletion, "create", lambda *args, **kwargs: DummyResponseNoFunc())
    question = "Test fallback"
    evidence = [{"chunk_id": "c1", "text": "dummy"}]
    tool_call = agent.generate_tool_call(question, evidence)
    assert tool_call["tool"] == agent.FUNCTION_NAME
    assert tool_call["program"] == "subtract(20,4), divide(#0,2)"
    assert "Fallback regex extracted program" in caplog.text


def test_generate_tool_call_invalid_dsl_fallback(monkeypatch, caplog):
    import openai
    import finrag.agent as agent
    import json
    caplog.set_level(logging.WARNING)
    class DummyMessageInvalidDSL:
        def __init__(self):
            self.function_call = {
                "name": agent.FUNCTION_NAME,
                "arguments": json.dumps({"program": "compute change from X to Y"})
            }
    class DummyChoiceInvalidDSL:
        def __init__(self):
            self.message = DummyMessageInvalidDSL()
    class DummyResponseInvalidDSL:
        def __init__(self):
            self.choices = [DummyChoiceInvalidDSL()]
    # Monkeypatch create to return invalid DSL in function_call
    monkeypatch.setattr(openai.ChatCompletion, "create", lambda *args, **kwargs: DummyResponseInvalidDSL())
    question = "Test invalid DSL fallback"
    evidence = [{"chunk_id": "c1", "text": "dummy"}]
    tool_call = agent.generate_tool_call(question, evidence)
    assert tool_call["tool"] == "none" # Fallback should fail if no DSL in args
    assert tool_call["program"] == ""
    assert "Program is invalid or empty" in caplog.text # Check substring
    assert "Fallback failed — no DSL extracted." in caplog.text
