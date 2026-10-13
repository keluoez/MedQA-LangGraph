#!/usr/bin/env python3
"""
Master script: Run all comparison evaluations.
Runs each dataset through both DeepSeek-V3 and Qwen2.5-72B evaluators.
Supports resume - safe to re-run if interrupted.

Prerequisites:
  Place downloaded datasets in evaluation/comparison_data/:
    - medical_o1_100_for_eval.json
    - chatdoctor_100_for_eval.json
    - medquad_100_for_eval.json

Usage:
  python run_all_comparisons.py --dataset all --model all     # run everything
  python run_all_comparisons.py --dataset medical_o1 --model deepseek  # single
  python run_all_comparisons.py --stats     # print stats for completed evals
"""
import subprocess, sys, json, os
from pathlib import Path

DATASETS = ["medical_o1", "chatdoctor", "medquad"]
MODELS = ["deepseek", "qwen"]


def run_eval(dataset, model):
    """Run a single evaluation; returns True on success."""
    cmd = [
        sys.executable, "run_comparison_eval.py",
        "--dataset", dataset,
        "--model", model,
    ]
    print(f"\n{'#'*70}")
    print(f"# {dataset} @ {model}")
    print(f"{'#'*70}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def print_all_stats():
    """Print stats for all completed evaluations."""
    print("\n" + "="*80)
    print("EVALUATION STATISTICS SUMMARY")
    print("="*80)

    all_stats = {}
    for dataset in DATASETS:
        for model in MODELS:
            path = Path(f"comparison_data/{dataset}_evaluated_{model}.json")
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                n = len(results)
                if n > 0:
                    s = {
                        "count": n,
                        "factuality": round(sum(r["factuality"] for r in results)/n, 3),
                        "granularity": round(sum(r["granularity"] for r in results)/n, 3),
                        "grounding": round(sum(r["grounding"] for r in results)/n, 3),
                        "total": round(sum(r["total"] for r in results)/n, 3),
                    }
                    all_stats[f"{dataset}_{model}"] = s

    # Print per-dataset
    for dataset in DATASETS:
        print(f"\n{'='*40}")
        print(f"  {dataset}")
        print(f"{'='*40}")
        for model in MODELS:
            key = f"{dataset}_{model}"
            if key in all_stats:
                s = all_stats[key]
                print(f"  {model:>10s}: F={s['factuality']:.2f} G={s['granularity']:.2f} Gr={s['grounding']:.2f} Total={s['total']:.2f}")

    # Save stats
    with open("comparison_data/all_stats.json", 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\nStats saved to comparison_data/all_stats.json")

    # Load ReasonMed + OurData stats for comparison
    print(f"\n{'='*80}")
    print("FULL COMPARISON (including existing ReasonMed + OurData)")
    print(f"{'='*80}")

    # ReasonMed (already evaluated)
    for model, fname in [("deepseek", "reasonmed_100_evaluated_deepseek.json"),
                          ("qwen", "reasonmed_100_evaluated_final.json")]:
        path = Path(fname)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            n = len(data)
            s = {
                "factuality": round(sum(r["factuality"] for r in data)/n, 3),
                "granularity": round(sum(r["granularity"] for r in data)/n, 3),
                "grounding": round(sum(r["grounding"] for r in data)/n, 3),
                "total": round(sum(r["total"] for r in data)/n, 3),
            }
            print(f"  ReasonMed ({model:>10s}): F={s['factuality']:.2f} G={s['granularity']:.2f} Gr={s['grounding']:.2f} Total={s['total']:.2f}")

    # OurData (already evaluated)
    for model, fname in [("deepseek", "our_data_100_evaluated_deepseek.json"),
                          ("qwen", "our_data_100_evaluated_qwen.json")]:
        path = Path(fname)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            n = len(data)
            s = {
                "factuality": round(sum(r["factuality"] for r in data)/n, 3),
                "granularity": round(sum(r["granularity"] for r in data)/n, 3),
                "grounding": round(sum(r["grounding"] for r in data)/n, 3),
                "total": round(sum(r["total"] for r in data)/n, 3),
            }
            print(f"  OurData   ({model:>10s}): F={s['factuality']:.2f} G={s['granularity']:.2f} Gr={s['grounding']:.2f} Total={s['total']:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--model", default="all")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        print_all_stats()
        sys.exit(0)

    ds_list = DATASETS if args.dataset == "all" else [args.dataset]
    md_list = MODELS if args.model == "all" else [args.model]

    print(f"Running evaluations: {ds_list} x {md_list}")
    for ds in ds_list:
        for md in md_list:
            run_eval(ds, md)

    print_all_stats()
