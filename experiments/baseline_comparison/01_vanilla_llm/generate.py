# -*- coding: utf-8 -*-
"""
Baseline 1 — Vanilla LLM
No retrieval. No structure constraint. No verification. No correction.
Corresponds to "Vanilla LLM" row in Table II.
"""

import json, os, sys, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.api_config import NVIDIA_API_KEY, NVIDIA_BASE_URL, GENERATION_MODEL

client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

SYSTEM_PROMPT = """You are a biomedical expert. Answer the following question about
drug-protein interaction mechanisms as accurately as possible based on your knowledge."""

def generate_vanilla(item: dict) -> dict:
    question = item["question"]
    try:
        resp = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            temperature=0.7,
            max_tokens=1500,
            timeout=90,
        )
        answer = resp.choices[0].message.content or ""
        return {
            "question": question,
            "answer":   answer,
            "variant":  "vanilla_llm",
            "source":   item.get("source", ""),
        }
    except Exception as e:
        return {"question": question, "answer": "", "variant": "vanilla_llm",
                "error": str(e)}


def run(input_file: str, output_file: str, count: int = 100, workers: int = 8):
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)
    data = data[:count]
    print(f"[Vanilla LLM] Running on {len(data)} samples …")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(generate_vanilla, it): it for it in data}
        done = 0
        for f in as_completed(futures):
            done += 1
            r = f.result()
            results.append(r)
            status = "OK" if "error" not in r else f"ERR"
            print(f"  [{done}/{len(data)}] {status}", flush=True)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Vanilla LLM] Saved {len(results)} results → {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="../../evaluation/our_data_100_for_eval.json")
    parser.add_argument("--output",  default="output/vanilla_llm_results.json")
    parser.add_argument("--count",   type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.input, args.output, args.count, args.workers)
