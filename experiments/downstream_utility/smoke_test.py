"""Smoke test: run 5 items across all 3 conditions."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_fewshot_eval import (
    load_pubmedqa, load_our_gold, load_reasonmed,
    select_fewshot_examples, predict_one,
    compute_metrics,
)
from shared.api_config import NVIDIA_API_KEY, NVIDIA_BASE_URL, GENERATION_MODEL
from openai import OpenAI

client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

pubmedqa = load_pubmedqa()
our_gold = load_our_gold()
reasonmed = load_reasonmed()

our_shots = select_fewshot_examples(our_gold, 3)
rm_shots = select_fewshot_examples(reasonmed, 3)

test_items = pubmedqa[:5]

for condition, shots, label in [
    ("zero_shot", None, "Zero-Shot"),
    ("fewshot_reasonmed", rm_shots, "Few-Shot ReasonMed"),
    ("fewshot_ours", our_shots, "Few-Shot MedQA-LG"),
]:
    print(f"\n=== {label} ===")
    results = []
    for item in test_items:
        r = predict_one(client, GENERATION_MODEL, condition, item, shots)
        icon = "OK" if r["correct"] else "X"
        print(f"  [{icon}] GT={r['ground_truth']:<6} Pred={r['prediction']:<8} raw={r['raw_output'][:60]}")
        results.append(r)
    m = compute_metrics(results)
    print(f"  Accuracy: {m['accuracy']:.3f}")
