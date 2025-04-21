#!/usr/bin/env python
import json
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    # Load dev turns
    dev_path = os.path.join(os.getcwd(), "data", "dev_turn.json")
    if not os.path.isfile(dev_path):
        print(f"Dev file not found: {dev_path}")
        return
    with open(dev_path, 'r', encoding='utf-8') as f:
        turns = json.load(f)

    seen = set()
    out = []
    for t in turns:
        qa = t.get("qa", {})
        q = qa.get("question")
        if q and q not in seen:
            seen.add(q)
            out.append({"sample_id": t.get("id"), "question": q})
        if len(out) >= 20:
            break

    output_path = os.path.join(os.getcwd(), "scripts", "upsert_questions.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out)} questions to {output_path}")

if __name__ == "__main__":
    main()
