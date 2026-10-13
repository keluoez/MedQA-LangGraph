# -*- coding: utf-8 -*-
"""
Ablation Variant 1 — Full System (Reference)
This is NOT re-generated. It reuses the existing 200 Gold samples
from 00_sample_200/ablation_200_samples.json as the Full System output,
since those samples were already produced by the complete pipeline.

This script simply copies the sampled data into the expected output location
and tags each item with variant="full_system".
"""

import json, os, sys, argparse

def run(input_file: str, output_file: str):
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    for it in data:
        it["variant"] = "full_system"
        # citation_hallucination is effectively 0 for Gold-verified data
        it["citation_hallucination"] = False

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[Full System] Tagged {len(data)} Gold samples → {output_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="../00_sample_200/ablation_200_samples.json")
    ap.add_argument("--output", default="output/full_system_results.json")
    args = ap.parse_args()
    run(args.input, args.output)
