# -*- coding: utf-8 -*-
"""Cross-validate generated answers with Qwen-2.5-72B and GPT-4o."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.evaluator import evaluate_batch, summarize

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
CONDITIONS = ["zero_shot", "fewshot_reasonmed", "fewshot_ours"]


def load_generated(condition):
    path = os.path.join(OUTPUT_DIR, f"gen_{condition}.jsonl")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def main():
    evaluators = [
        ("qwen",  "eval_{cond}_qwen.jsonl"),
        ("gpt4o", "eval_{cond}_gpt4o.jsonl"),
    ]

    for evaluator, tmpl in evaluators:
        print(f"\n{'='*70}")
        print(f"Evaluator: {evaluator}")
        print(f"{'='*70}")

        for condition in CONDITIONS:
            items = load_generated(condition)
            ckpt = os.path.join(OUTPUT_DIR, tmpl.format(cond=condition))
            print(f"\n--- {condition} ({len(items)} items) ---")
            results = evaluate_batch(items, model=evaluator,
                                     max_workers=8, checkpoint_file=ckpt)
            s = summarize(results)
            for variant, m in s.items():
                print(f"  {variant}: Total={m['mean_total']:.3f}  "
                      f"F={m['mean_factuality']:.3f} G={m['mean_granularity']:.3f} "
                      f"E={m['mean_grounding']:.3f}  n={m['n']}")

    # Print consolidated table
    print("\n" + "="*70)
    print("CONSOLIDATED RESULTS (All 3 Evaluators)")
    print("="*70)

    all_evals = ["deepseek", "qwen", "gpt4o"]
    templates = {
        "deepseek": "eval_{cond}_deepseek.jsonl",
        "qwen":     "eval_{cond}_qwen.jsonl",
        "gpt4o":    "eval_{cond}_gpt4o.jsonl",
    }

    for eval_name in all_evals:
        print(f"\n--- {eval_name} ---")
        print(f"{'Condition':<25} {'Total':>8} {'Fact':>6} {'Gran':>6} {'Grnd':>6}")
        print("-" * 58)
        for condition in CONDITIONS:
            ckpt = os.path.join(OUTPUT_DIR, templates[eval_name].format(cond=condition))
            if not os.path.exists(ckpt):
                print(f"  {condition:<23} [not yet evaluated]")
                continue
            items = []
            with open(ckpt, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
            s = summarize(items)
            label = condition.replace("_", " ").title()
            for variant, m in s.items():
                print(f"  {label:<23} {m['mean_total']:>8.3f}  "
                      f"{m['mean_factuality']:>6.3f} {m['mean_granularity']:>6.3f} "
                      f"{m['mean_grounding']:>6.3f}")


if __name__ == "__main__":
    main()
