# -*- coding: utf-8 -*-
"""
Ablation Variant 2 — w/o Retrieval
No evidence retrieval. Direct LLM generation, no structure constraint, no verification.
This is identical to "Vanilla LLM" in Table II — provides the ablation lower bound.
"""

import json, os, sys, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from experiments.shared.api_config import NVIDIA_API_KEY, NVIDIA_BASE_URL, GENERATION_MODEL

client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

SYSTEM = "You are a biomedical expert. Answer the question about drug-protein mechanism."


def generate(item: dict) -> dict:
    try:
        resp = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user",   "content": item["question"]}],
            temperature=0.7, max_tokens=1500, timeout=90)
        return {"question": item["question"],
                "answer":   resp.choices[0].message.content or "",
                "variant":  "wo_retrieval"}
    except Exception as e:
        return {"question": item["question"], "answer": "", "variant": "wo_retrieval", "error": str(e)}


def run(input_file, output_file, workers=8):
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[w/o Retrieval] {len(data)} samples …")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(generate, it): it for it in data}
        done = 0
        for f in as_completed(futures):
            done += 1; results.append(f.result())
            print(f"  [{done}/{len(data)}]", flush=True)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved → {output_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   default="../00_sample_200/ablation_200_samples.json")
    ap.add_argument("--output",  default="output/wo_retrieval_results.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run(args.input, args.output, args.workers)
