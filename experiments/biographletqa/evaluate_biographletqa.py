#!/usr/bin/env python3
"""
Evaluate BioGraphletQA against our MedQA-LangGraph.
- Samples 100 random QA pairs from filtered_ds.jsonl
- Evaluates with the same 3-evaluator rubric (Factuality, Granularity, Grounding)
- Compares Grounding scores against ours (2.80) and ReasonMed (1.57)

Usage:
    python evaluate_biographletqa.py [--evaluators deepseek qwen gpt4o]
"""
import json, os, sys, random, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.evaluator import evaluate_batch, summarize

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FILTERED_DS = os.path.join(DATA_DIR, "filtered_ds.jsonl")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")


def load_and_sample(n=100, seed=42):
    """Load filtered_ds.jsonl and sample n entries."""
    if not os.path.exists(FILTERED_DS):
        print(f"ERROR: {FILTERED_DS} not found. Download it first.")
        sys.exit(1)

    print(f"Loading {FILTERED_DS} ...")
    all_entries = []
    with open(FILTERED_DS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_entries.append(json.loads(line))

    print(f"Loaded {len(all_entries)} entries. Sampling {n} ...")
    rng = random.Random(seed)
    sample = rng.sample(all_entries, min(n, len(all_entries)))

    # Convert to our evaluation format
    eval_items = []
    for entry in sample:
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        # Build a combined display text
        full_answer = answer
        # Include graphlet info as evidence context for evaluators
        node_names = entry.get("node_names", [])
        if node_names:
            full_answer += f"\n\n[KG Context: {', '.join(node_names[:5])}]"

        eval_items.append({
            "question": question,
            "answer": full_answer,
            "variant": "biographletqa",
            "source": f"biographletqa_id_{entry.get('id', '?')}",
            # Preserve original for reference
            "_original_answer": answer,
            "_node_names": node_names,
            "_graphlet_id": entry.get("graphlet_id"),
        })

    return eval_items


def run(evaluators):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    items = load_and_sample(n=100)

    for evaluator in evaluators:
        print(f"\n=== Evaluating BioGraphletQA with {evaluator.upper()} (n={len(items)}) ===")
        ckpt = os.path.join(OUTPUT_DIR, f"biographletqa_{evaluator}.jsonl")
        results = evaluate_batch(items, model=evaluator, max_workers=8, checkpoint_file=ckpt)

        out_path = os.path.join(OUTPUT_DIR, f"biographletqa_results_{evaluator}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        summary = summarize(results)
        print(f"\n{'='*60}")
        print(f"BioGraphletQA — {evaluator.upper()} RESULTS")
        print(f"{'='*60}")
        if "biographletqa" in summary:
            s = summary["biographletqa"]
            print(f"  N = {s['n']}")
            print(f"  Factuality:   {s['mean_factuality']:.3f} / 4.0")
            print(f"  Granularity:  {s['mean_granularity']:.3f} / 3.0")
            print(f"  Grounding:    {s['mean_grounding']:.3f} / 3.0")
            print(f"  TOTAL:        {s['mean_total']:.3f} / 10.0")

        summary_path = os.path.join(OUTPUT_DIR, f"biographletqa_summary_{evaluator}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved -> {out_path}")
        print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluators", nargs="+", default=["deepseek"],
                    choices=["deepseek", "qwen", "gpt4o"])
    args = ap.parse_args()
    run(args.evaluators)
