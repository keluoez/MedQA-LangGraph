# -*- coding: utf-8 -*-
"""
One-click runner for the BASELINE COMPARISON experiment (Table II).

Steps:
  1. Vanilla LLM        → 01_vanilla_llm/
  2. Standard RAG       → 02_standard_rag/
  3. Iterative RAG      → 03_iterative_rag/
  4. Evaluate all 4     → 04_evaluate_all/

Usage:
    cd baseline_comparison
    python run_all_baselines.py --count 100 --evaluators deepseek qwen gpt4o
"""

import argparse, subprocess, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE, "..", "..", "evaluation", "our_data_100_for_eval.json")


def run(cmd, cwd=None):
    print(f"\n{'='*60}\n  {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run([sys.executable] + cmd, cwd=cwd or BASE)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count",      type=int,   default=100)
    ap.add_argument("--workers",    type=int,   default=5)
    ap.add_argument("--evaluators", nargs="+",  default=["deepseek"],
                    choices=["deepseek", "qwen", "gpt4o"])
    ap.add_argument("--skip-gen",   action="store_true",
                    help="Skip generation (re-evaluate existing outputs only)")
    args = ap.parse_args()

    if not args.skip_gen:
        run(["01_vanilla_llm/generate.py",
             "--input", INPUT,
             "--output", "01_vanilla_llm/output/vanilla_llm_results.json",
             "--count", str(args.count),
             "--workers", str(args.workers)])

        run(["02_standard_rag/generate.py",
             "--input", INPUT,
             "--output", "02_standard_rag/output/standard_rag_results.json",
             "--count", str(args.count),
             "--workers", str(args.workers)])

        run(["03_iterative_rag/generate.py",
             "--input", INPUT,
             "--output", "03_iterative_rag/output/iterative_rag_results.json",
             "--count", str(args.count),
             "--workers", str(args.workers)])

    for ev in args.evaluators:
        run(["04_evaluate_all/evaluate.py",
             "--evaluators", ev,
             "--output", "04_evaluate_all/output"])

    print("\n✅  Baseline comparison complete. Results in 04_evaluate_all/output/")


if __name__ == "__main__":
    main()
