# -*- coding: utf-8 -*-
"""
Ablation Study — Step 0: Sample 200 pairs from existing Gold data.
Run this ONCE. All 5 ablation variants use the SAME 200 samples.
"""

import json, os, random, argparse

# Compute project root (run-bdi/) from this file's location
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT  = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))

GOLD_FILES = [
    os.path.join(_PROJ_ROOT, "basic_drug_interaction",         "output"),
    os.path.join(_PROJ_ROOT, "basic_experimental_conditions",  "output"),
    os.path.join(_PROJ_ROOT, "basic_pathological_environment", "output"),
    os.path.join(_PROJ_ROOT, "basic_physiological_conditions", "output"),
]

def collect_gold(directories: list, min_score: float = 8.0) -> list:
    gold = []
    for d in directories:
        if not os.path.isdir(d):
            print(f"  [WARN] Directory not found: {d}")
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
                    score = it.get("quality_score", it.get("total", 0))
                    if score >= min_score and it.get("question") and it.get("answer"):
                        gold.append({
                            "question": it["question"],
                            "answer":   it["answer"],
                            "source":   fname,
                        })
            except Exception as e:
                print(f"  [WARN] Could not parse {fpath}: {e}")
    return gold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count",  type=int, default=200)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output", default="ablation_200_samples.json")
    args = parser.parse_args()

    print("Collecting Gold samples …")
    gold = collect_gold(GOLD_FILES)
    print(f"  Found {len(gold)} Gold samples (score ≥ 8.0)")

    if len(gold) < args.count:
        print(f"  [WARN] Only {len(gold)} samples available; using all.")
        sampled = gold
    else:
        random.seed(args.seed)
        sampled = random.sample(gold, args.count)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(sampled)} samples → {args.output}")


if __name__ == "__main__":
    main()
