import pytest
from finrag.eval import evaluate_agent


def make_sample(sample_id, question, answer, program):
    return {
        "id": sample_id,
        "qa": {"question": question, "answer": answer, "program": program},
        # dummy turn fields for gold evidence (won't be used by stub)
        "pre_text": [],
        "table": [],
        "post_text": [],
        "ann_text_rows": [],
        "ann_table_rows": [],
        "ann_post_rows": [],
    }

@pytest.fixture(autouse=True)
def patch_plan_and_execute(monkeypatch):
    # Stub plan_and_execute to return answer/program based on program
    def fake_plan_and_execute(question, evidence_chunks):
        # extract id from question for variability
        if question.endswith("_ok"):
            return {"answer": "GOOD", "program": "GOOD", "tool": "run_math_tool", "evidence": []}
        else:
            return {"answer": "BAD", "program": "BAD", "tool": "run_math_tool", "evidence": []}
    import finrag.eval as eval_mod
    monkeypatch.setattr(eval_mod, "plan_and_execute", fake_plan_and_execute)


def test_evaluate_agent_metrics():
    # Two samples: one correct, one incorrect
    s1 = make_sample("1", "q1_ok", "GOOD", "GOOD")
    s2 = make_sample("2", "q2_fail", "GOOD", "GOOD")
    report = evaluate_agent([s1, s2], use_retrieval=False)
    assert report["total"] == 2
    assert report["execution_accuracy"] == pytest.approx(0.5)
    assert report["program_match_rate"] == pytest.approx(0.5)
    assert report["tool_usage_rate"] == pytest.approx(1.0)
    # Failures: sample 2 (answer BAD vs gold GOOD)
    assert report["failures"] == ["2"]
    # Examples length
    assert len(report["examples"]) == 2
