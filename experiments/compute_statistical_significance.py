#!/usr/bin/env python3
"""
Compute statistical significance tests for MedQA-LangGraph paper.
- Paired t-test (Welch's t-test for unequal variance)
- Wilcoxon signed-rank test
- Bootstrap 95% confidence intervals

Covers: Baseline Comparison (Table II), Ablation Study (Table IV), Downstream Utility
"""
import json
import os
import sys
import numpy as np
from scipy import stats
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(BASE_DIR, "baseline_comparison", "04_evaluate_all", "output")
RAG_QR_DIR   = os.path.join(BASE_DIR, "baseline_comparison", "05_rag_query_rewrite", "output")
ABLATION_DIR = os.path.join(BASE_DIR, "ablation_study", "06_evaluate_all", "output")

EVALUATORS = ["deepseek", "qwen", "gpt4o"]

BASELINE_VARIANTS = {
    "vanilla_llm": "Vanilla LLM",
    "standard_rag": "Standard RAG",
    "iterative_rag": "Iterative RAG",
    "rag_query_rewrite": "RAG + Query Rewrite",
    "proposed": "MedQA-LangGraph (Ours)",
}

ABLATION_VARIANTS = {
    "full_system": "Full System",
    "wo_pmid_whitelist": "w/o PMID Whitelist",
    "wo_correction_loop": "w/o Correction Loop",
    "wo_struct_constraint": "w/o Struct. Constraint",
    "wo_retrieval": "w/o Retrieval",
}

ABLATION_ORDER = [
    "full_system",
    "wo_struct_constraint",
    "wo_pmid_whitelist",
    "wo_correction_loop",
    "wo_retrieval",
]


def _read_json(path: str):
    """Read JSON file with encoding fallback."""
    for enc in ["utf-8", "gbk", "gb2312", "latin-1"]:
        try:
            with open(path, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    with open(path, encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def load_baseline(evaluator: str) -> dict:
    """Load baseline results for one evaluator. Returns {variant: [total_scores]}."""
    path = os.path.join(BASELINE_DIR, f"baseline_results_{evaluator}.json")
    items = _read_json(path)
    groups = defaultdict(list)
    for it in items:
        groups[it.get("variant", "unknown")].append(it.get("total", 0))
    return dict(groups)


def load_rag_qr(evaluator: str) -> list:
    """Load RAG+QueryRewrite results. Returns [total_scores]."""
    path = os.path.join(RAG_QR_DIR, f"eval_results_{evaluator}.json")
    items = _read_json(path)
    return [it.get("total", 0) for it in items]


def load_ablation(evaluator: str) -> dict:
    """Load ablation results for one evaluator."""
    path = os.path.join(ABLATION_DIR, f"ablation_results_{evaluator}.json")
    items = _read_json(path)
    groups = defaultdict(list)
    for it in items:
        groups[it.get("variant", "unknown")].append(it.get("total", 0))
    return dict(groups)


def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(42)
    means = []
    data = np.array(data)
    n = len(data)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return np.mean(data), lower, upper


def compute_paired_tests(group_a: list, group_b: list, name_a: str, name_b: str) -> dict:
    """Compute paired t-test and Wilcoxon signed-rank between two groups."""
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)
    # Ensure equal length by truncation
    min_n = min(len(a), len(b))
    a = a[:min_n]
    b = b[:min_n]

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(a, b)

    # Wilcoxon signed-rank
    w_stat, w_p = stats.wilcoxon(a, b) if min_n > 0 else (np.nan, np.nan)

    # Mean diff
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    diff = mean_a - mean_b

    # Bootstrap CI for the difference
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(10000):
        idx = rng.choice(min_n, size=min_n, replace=True)
        diffs.append(np.mean(a[idx]) - np.mean(b[idx]))
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)

    return {
        "pair": f"{name_a} vs {name_b}",
        "mean_a": round(mean_a, 3),
        "mean_b": round(mean_b, 3),
        "diff": round(diff, 3),
        "bootstrap_95ci": (round(ci_low, 3), round(ci_high, 3)),
        "t_test": (round(t_stat, 4), round(t_p, 6)),
        "wilcoxon": (round(w_stat, 4), round(w_p, 6)),
        "n": min_n,
    }


def run():
    print("=" * 90)
    print("STATISTICAL SIGNIFICANCE ANALYSIS FOR MedQA-LangGraph PAPER")
    print("=" * 90)

    # ═══════════════════════════════════════════════════════════════════
    # 1. BASELINE COMPARISON (Ours vs each baseline, per evaluator)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 90)
    print("1. BASELINE COMPARISON — Paired Tests (Ours vs each baseline)")
    print("─" * 90)

    for evaluator in EVALUATORS:
        data = load_baseline(evaluator)
        # Load RAG+QR separately
        rag_qr_scores = load_rag_qr(evaluator)
        data["rag_query_rewrite"] = rag_qr_scores

        proposed = data.get("proposed", [])
        if not proposed:
            print(f"  [{evaluator}] No proposed data, skipping")
            continue

        print(f"\n  [{evaluator.upper()}]")
        baselines_to_test = ["vanilla_llm", "standard_rag", "rag_query_rewrite", "iterative_rag"]
        for base in baselines_to_test:
            base_scores = data.get(base, [])
            if not base_scores:
                continue
            r = compute_paired_tests(proposed, base_scores, "Ours", BASELINE_VARIANTS[base])
            sig = "***" if r["t_test"][1] < 0.001 else ("**" if r["t_test"][1] < 0.01 else ("*" if r["t_test"][1] < 0.05 else "ns"))
            print(f"    Ours ({r['mean_a']:.2f}) vs {BASELINE_VARIANTS[base]:<18} ({r['mean_b']:.2f})  "
                  f"Δ={r['diff']:+.2f}  95%CI=[{r['bootstrap_95ci'][0]:.2f}, {r['bootstrap_95ci'][1]:.2f}]  "
                  f"t={r['t_test'][0]:.3f} p={r['t_test'][1]:.6f} {sig}  "
                  f"Wilcoxon p={r['wilcoxon'][1]:.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # 2. ABLATION STUDY (Full System vs each variant)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 90)
    print("2. ABLATION STUDY — Paired Tests (Full System vs each variant)")
    print("─" * 90)

    for evaluator in EVALUATORS:
        data = load_ablation(evaluator)
        full = data.get("full_system", [])
        if not full:
            print(f"  [{evaluator}] No full_system data, skipping")
            continue

        print(f"\n  [{evaluator.upper()}]")
        for variant in ABLATION_ORDER[1:]:  # skip full_system
            var_scores = data.get(variant, [])
            if not var_scores:
                continue
            r = compute_paired_tests(full, var_scores, "Full System", ABLATION_VARIANTS[variant])
            sig = "***" if r["t_test"][1] < 0.001 else ("**" if r["t_test"][1] < 0.01 else ("*" if r["t_test"][1] < 0.05 else "ns"))
            print(f"    Full ({r['mean_a']:.2f}) vs {ABLATION_VARIANTS[variant]:<22} ({r['mean_b']:.2f})  "
                  f"Δ={r['diff']:+.2f}  95%CI=[{r['bootstrap_95ci'][0]:.2f}, {r['bootstrap_95ci'][1]:.2f}]  "
                  f"t={r['t_test'][0]:.3f} p={r['t_test'][1]:.6f} {sig}  "
                  f"Wilcoxon p={r['wilcoxon'][1]:.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # 3. BOOTSTRAP CI FOR ALL MEANS (baseline + ablation, all evaluators)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 90)
    print("3. BOOTSTRAP 95% CONFIDENCE INTERVALS FOR ALL MEANS")
    print("─" * 90)

    print("\n  Baseline Comparison:")
    for evaluator in EVALUATORS:
        data = load_baseline(evaluator)
        rag_qr_scores = load_rag_qr(evaluator)
        data["rag_query_rewrite"] = rag_qr_scores
        print(f"\n    [{evaluator.upper()}]")
        for v in ["vanilla_llm", "standard_rag", "rag_query_rewrite", "iterative_rag", "proposed"]:
            scores = data.get(v, [])
            if scores:
                m, lo, hi = bootstrap_ci(scores)
                print(f"      {BASELINE_VARIANTS[v]:<22} mean={m:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]")

    print("\n  Ablation Study:")
    for evaluator in EVALUATORS:
        data = load_ablation(evaluator)
        print(f"\n    [{evaluator.upper()}]")
        for v in ABLATION_ORDER:
            scores = data.get(v, [])
            if scores:
                m, lo, hi = bootstrap_ci(scores)
                print(f"      {ABLATION_VARIANTS[v]:<25} mean={m:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]")

    # ═══════════════════════════════════════════════════════════════════
    # 4. SUMMARY — Paper-ready statements
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("4. PAPER-READY SIGNIFICANCE STATEMENTS")
    print("=" * 90)

    # Aggregate across evaluators for baseline: Ours vs RAG+QR
    all_proposed = []
    all_ragqr = []
    for evaluator in EVALUATORS:
        data = load_baseline(evaluator)
        proposed = data.get("proposed", [])
        rag_qr = load_rag_qr(evaluator)
        if proposed:
            all_proposed.extend(proposed)
            all_ragqr.extend(rag_qr[:len(proposed)])
    if all_proposed:
        r = compute_paired_tests(all_proposed, all_ragqr, "Ours", "RAG+QR")
        print(f"\n  Combined (3 evaluators × n=100): Ours vs RAG+Query Rewrite")
        print(f"    Ours = {r['mean_a']:.3f}, RAG+Query Rewrite = {r['mean_b']:.3f}")
        print(f"    Δ = {r['diff']:+.3f}, 95% Bootstrap CI = [{r['bootstrap_95ci'][0]:.3f}, {r['bootstrap_95ci'][1]:.3f}]")
        print(f"    Paired t-test: t = {r['t_test'][0]:.4f}, p = {r['t_test'][1]:.6f}")
        print(f"    Wilcoxon signed-rank: p = {r['wilcoxon'][1]:.6f}")

    # Aggregate across evaluators for ablation: Full vs each variant
    all_full = []
    all_no_corr = []
    all_no_pmid = []
    all_no_struct = []
    all_no_retrieval = []
    for evaluator in EVALUATORS:
        data = load_ablation(evaluator)
        full = data.get("full_system", [])
        no_corr = data.get("wo_correction_loop", [])
        no_pmid = data.get("wo_pmid_whitelist", [])
        no_struct = data.get("wo_struct_constraint", [])
        no_retrieval = data.get("wo_retrieval", [])
        if full:
            all_full.extend(full)
            all_no_corr.extend(no_corr[:len(full)])
            all_no_pmid.extend(no_pmid[:len(full)])
            all_no_struct.extend(no_struct[:len(full)])
            all_no_retrieval.extend(no_retrieval[:len(full)])

    if all_full:
        print(f"\n  Combined (3 evaluators × n=200): Ablation Significance")
        for name, scores in [
            ("w/o Correction Loop", all_no_corr),
            ("w/o PMID Whitelist", all_no_pmid),
            ("w/o Struct. Constraint", all_no_struct),
            ("w/o Retrieval", all_no_retrieval),
        ]:
            r = compute_paired_tests(all_full, scores, "Full System", name)
            print(f"    Full System vs {name}: Δ = {r['diff']:+.2f}, "
                  f"95%CI = [{r['bootstrap_95ci'][0]:.2f}, {r['bootstrap_95ci'][1]:.2f}], "
                  f"p_t = {r['t_test'][1]:.6f}, p_w = {r['wilcoxon'][1]:.6f}")

    print("\n" + "=" * 90)
    print("DONE.")
    print("=" * 90)


if __name__ == "__main__":
    run()
