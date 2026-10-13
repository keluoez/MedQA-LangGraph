# -*- coding: utf-8 -*-
"""
Baseline Comparison — Step 4: Evaluate all 4 methods with 3 evaluators.

Reads outputs from 01/02/03 and our_data_100_evaluated_*.json (Proposed).
Produces a unified results CSV + summary table ready to paste into the paper.

Usage:
    python evaluate.py --evaluators deepseek qwen gpt4o
"""

import json, os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from experiments.shared.evaluator import evaluate_batch, summarize

VARIANTS = {
    "vanilla_llm":    "../01_vanilla_llm/output/vanilla_llm_results.json",
    "standard_rag":   "../02_standard_rag/output/standard_rag_results.json",
    "iterative_rag":  "../03_iterative_rag/output/iterative_rag_results.json",
    # "proposed" — use restored full answers (not the truncated deepseek eval file)
    "proposed":       "../../../evaluation/our_data_100_full_answers.json",
}


def load_variant(name: str, path: str) -> list:
    if not os.path.exists(path):
        print(f"  [WARN] {name}: file not found → {path}")
        return []
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    for it in items:
        it["variant"] = name
    return items


def run(evaluators: list, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)
    all_items = []
    for name, path in VARIANTS.items():
        items = load_variant(name, path)
        print(f"  Loaded {len(items):3d} items for '{name}'")
        all_items.extend(items)

    if not all_items:
        print("[ERROR] No items loaded. Run the generate scripts first.")
        return

    for evaluator in evaluators:
        print(f"\n=== Evaluating with {evaluator.upper()} ===")
        items_to_eval = [it for it in all_items if it.get("variant") != "proposed"]
        proposed_items = [it for it in all_items if it.get("variant") == "proposed"]

        results = evaluate_batch(items_to_eval, model=evaluator, max_workers=8)

        # ALL evaluators independently evaluate proposed answers via API
        # using the full un-truncated answers from our_data_100_full_answers.json
        print(f"  [proposed] Calling {evaluator.upper()} API to independently "
              f"evaluate {len(proposed_items)} proposed answers …")
        proposed_results = evaluate_batch(proposed_items, model=evaluator,
                                          max_workers=8)
        results.extend(proposed_results)

        out_path = os.path.join(output_dir, f"baseline_results_{evaluator}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        summary = summarize(results)
        print(f"\n{'Variant':<22} {'N':>4} {'Factuality':>12} {'Granularity':>12} {'Grounding':>12} {'Total':>8}")
        print("-" * 76)
        order = ["vanilla_llm", "standard_rag", "iterative_rag", "proposed"]
        for v in order:
            if v in summary:
                s = summary[v]
                print(f"{v:<22} {s['n']:>4} {s['mean_factuality']:>12.3f} "
                      f"{s['mean_granularity']:>12.3f} {s['mean_grounding']:>12.3f} "
                      f"{s['mean_total']:>8.3f}")

        summary_path = os.path.join(output_dir, f"baseline_summary_{evaluator}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved → {out_path}")
        print(f"Summary → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluators", nargs="+",
                        default=["deepseek"],
                        choices=["deepseek", "qwen", "gpt4o"],
                        help="Which evaluator models to use")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    run(args.evaluators, args.output)
