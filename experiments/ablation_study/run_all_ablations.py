# -*- coding: utf-8 -*-
"""
One-click runner for the ABLATION STUDY experiment (Table IV).

Steps:
  0. Sample 200 shared pairs         → 00_sample_200/
  1. Full System (reference)         → 01_full_system/   (reuses existing Gold data)
  2. w/o Retrieval                   → 02_wo_retrieval/
  3. w/o Correction Loop             → 03_wo_correction_loop/
  4. w/o Structural Constraint       → 04_wo_struct_constraint/
  5. w/o PMID Whitelist (new)        → 05_wo_pmid_whitelist/
  6. Evaluate all 5                  → 06_evaluate_all/

Usage:
    cd ablation_study
    python run_all_ablations.py --count 200 --evaluators deepseek
"""

import argparse, subprocess, sys, os

BASE   = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(BASE, "00_sample_200", "ablation_200_samples.json")


def run(cmd, cwd=None):
    print(f"\n{'='*60}\n  {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run([sys.executable] + cmd, cwd=cwd or BASE)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count",      type=int,  default=200)
    ap.add_argument("--workers",    type=int,  default=5)
    ap.add_argument("--evaluators", nargs="+", default=["deepseek"],
                    choices=["deepseek", "qwen", "gpt4o"])
    ap.add_argument("--skip-gen",   action="store_true",
                    help="Skip generation (re-evaluate existing outputs only)")
    args = ap.parse_args()

    if not args.skip_gen:
        # Step 0: sample
        run(["00_sample_200/sample.py",
             "--count", str(args.count),
             "--output", SAMPLE])

        # Step 2-5: ablation variants
        for variant_dir, script in [
            ("02_wo_retrieval",        "generate.py"),
            ("03_wo_correction_loop",  "generate.py"),
            ("04_wo_struct_constraint","generate.py"),
            ("05_wo_pmid_whitelist",   "generate.py"),
        ]:
            output = os.path.join(BASE, variant_dir, "output",
                                  f"{variant_dir.split('_',1)[1]}_results.json")
            run([f"{variant_dir}/{script}",
                 "--input", SAMPLE,
                 "--output", output,
                 "--workers", str(args.workers)])

    # Step 6: evaluate
    for ev in args.evaluators:
        run(["06_evaluate_all/evaluate.py",
             "--evaluators", ev,
             "--output", "06_evaluate_all/output"])

    print("\n✅  Ablation study complete. Results in 06_evaluate_all/output/")


if __name__ == "__main__":
    main()
