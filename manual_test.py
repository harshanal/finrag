"""
Manual test for plan_and_execute with a hand-picked QA pair and evidence.
Run with:
    poetry run python manual_test.py
"""
import json
from finrag.agent import plan_and_execute

if __name__ == "__main__":
    question = (
        "What is the percentage change in net cash from 2008 to 2009?"
    )
    evidence = [
        {
            "chunk_id": "text:pre:0",
            "text": (
                "Net cash from operations in 2008 was 180000 and in 2009 was 205000."
            ),
        }
    ]
    result = plan_and_execute(question, evidence)
    print(json.dumps(result, indent=2))
