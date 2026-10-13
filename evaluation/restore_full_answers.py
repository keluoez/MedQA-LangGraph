# -*- coding: utf-8 -*-
"""
Re-extract the 100 evaluation samples with FULL (un-truncated) answers
from the original Gold JSONL files.

The previous our_data_100_evaluated_deepseek.json had answers truncated to
300 chars, which caused Qwen to miss the entire 【Evidence】section with PMIDs,
resulting in artificially low Grounding scores.
"""

import json, os, glob

# The truncated file — we need the questions to match
TRUNCATED = "evaluation/our_data_100_evaluated_deepseek.json"

# Original Gold data directories
GOLD_DIRS = [
    "basic_drug_interaction/output",
    "basic_experimental_conditions/output",
    "basic_pathological_environment/output",
    "basic_physiological_conditions/output",
]

def load_gold_index():
    """Build question -> full_answer index from ALL Gold JSONL files."""
    index = {}
    for d in GOLD_DIRS:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith((".json", ".jsonl")):
                continue
            fpath = os.path.join(d, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    content = f.read().strip()
                    if content.startswith("["):
                        items = json.loads(content)
                    else:
                        items = [json.loads(l) for l in content.splitlines() if l.strip()]
                for it in items:
                    q = it.get("question", "").strip()
                    a = it.get("answer", "").strip()
                    if q and a and len(a) > 300:
                        # Prefer longer (more complete) answers
                        if q not in index or len(a) > len(index[q]):
                            index[q] = a
            except:
                pass
    return index

def main():
    with open(TRUNCATED, encoding="utf-8") as f:
        truncated_data = json.load(f)

    gold_index = load_gold_index()
    print(f"Gold index: {len(gold_index)} questions with full answers")

    restored = []
    matched = 0
    for item in truncated_data:
        q = item["question"].strip()
        full_answer = gold_index.get(q)
        if full_answer:
            matched += 1
            restored.append({
                "question": q,
                "answer": full_answer,
                "source": "OurData",
                "variant": "proposed",
            })
        else:
            # Fallback: use the truncated answer (shouldn't happen often)
            restored.append({
                "question": q,
                "answer": item.get("answer", ""),
                "source": "OurData",
                "variant": "proposed",
            })

    out = "evaluation/our_data_100_full_answers.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(restored, f, ensure_ascii=False, indent=2)

    # Show stats
    lens = [len(r["answer"]) for r in restored]
    print(f"Matched {matched}/{len(truncated_data)} questions to full Gold answers")
    print(f"Answer lengths: min={min(lens)}, max={max(lens)}, avg={sum(lens)//len(lens)}")
    has_evidence = sum(1 for r in restored if "【Evidence】" in r["answer"] or "PMID" in r["answer"])
    print(f"Answers with Evidence/PMID: {has_evidence}/{len(restored)}")
    print(f"Saved → {out}")

if __name__ == "__main__":
    main()
