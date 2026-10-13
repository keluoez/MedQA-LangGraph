# -*- coding: utf-8 -*-
"""
Ablation Study — Step 6: Evaluate all 5 variants with DeepSeek evaluator.
Produces summary table matching Table IV in the paper.

Usage:
    python evaluate.py [--evaluators deepseek qwen]
"""

import json, os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from experiments.shared.evaluator import evaluate_batch, summarize

# Variant → output file mapping
VARIANTS = {
    "full_system":         "../01_full_system/output/full_system_results.json",
    "wo_pmid_whitelist":   "../05_wo_pmid_whitelist/output/wo_pmid_whitelist_results.json",
    "wo_correction_loop":  "../03_wo_correction_loop/output/wo_correction_loop_results.json",
    "wo_struct_constraint":"../04_wo_struct_constraint/output/wo_struct_constraint_results.json",
    "wo_retrieval":        "../02_wo_retrieval/output/wo_retrieval_results.json",
}

DISPLAY_ORDER = [
    "full_system",
    "wo_pmid_whitelist",
    "wo_correction_loop",
    "wo_struct_constraint",
    "wo_retrieval",
]


def load_variant(name, path):
    if not os.path.exists(path):
        print(f"  [WARN] {name}: not found → {path}")
        return []
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    for it in items:
        it["variant"] = name
    return items


def compute_halluc_rate(items):
    """Compute citation hallucination rate from raw generation metadata."""
    rates = {}
    from collections import defaultdict
    groups = defaultdict(list)
    for it in items:
        groups[it.get("variant", "unknown")].append(it)
    for v, rows in groups.items():
        n = len(rows)
        halluc = sum(1 for r in rows if r.get("citation_hallucination", False))
        rates[v] = round(halluc / n * 100, 1) if n else 0.0
    return rates


def run(evaluators, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    all_items = []
    for name, path in VARIANTS.items():
        items = load_variant(name, path)
        print(f"  Loaded {len(items):3d} items for '{name}'")
        all_items.extend(items)

    halluc_rates = compute_halluc_rate(all_items)

    for evaluator in evaluators:
        print(f"\n=== Evaluating with {evaluator.upper()} (n={len(all_items)}) ===")
        ckpt_path = os.path.join(output_dir, f"ablation_results_{evaluator}.jsonl")
        results = evaluate_batch(all_items, model=evaluator, max_workers=8, checkpoint_file=ckpt_path)

        out_path = os.path.join(output_dir, f"ablation_results_{evaluator}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        summary = summarize(results)

        print(f"\n{'Variant':<26} {'N':>4} {'Mean Score':>12} {'Halluc%':>9}")
        print("-" * 58)
        for v in DISPLAY_ORDER:
            if v in summary:
                s = summary[v]
                h = halluc_rates.get(v, "N/A")
                print(f"{v:<26} {s['n']:>4} {s['mean_total']:>12.3f} {str(h)+' %':>9}")

        summary_path = os.path.join(output_dir, f"ablation_summary_{evaluator}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"scores": summary, "hallucination_rates": halluc_rates},
                      f, ensure_ascii=False, indent=2)
        print(f"\nSaved → {out_path}\nSummary → {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluators", nargs="+", default=["deepseek"],
                    choices=["deepseek", "qwen", "gpt4o"])
    ap.add_argument("--output", default="output")
    args = ap.parse_args()
    run(args.evaluators, args.output)
