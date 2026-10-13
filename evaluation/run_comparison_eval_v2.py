#!/usr/bin/env python3
"""
Unified comparison evaluation — extended to support GPT-4o + ReasonMed.
Supports real-time checkpoint save (every single item).

Usage:
  python run_comparison_eval_v2.py --dataset medical_o1 --model gpt4o
  python run_comparison_eval_v2.py --dataset reasonmed --model gpt4o
  python run_comparison_eval_v2.py --dataset all --model gpt4o
"""
import json, os, time, re, hashlib, argparse
from pathlib import Path
from openai import OpenAI

# ===== CONFIGURATION =====
# 加载项目根目录 .env（密钥配置）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY_SF = os.getenv("SILICONFLOW_API_KEY", "")
API_KEY_OAI = os.getenv("OPENAI_API_KEY", "")
BASE_URL_SF = "https://api.siliconflow.cn/v1"
BASE_URL_OAI = "https://proaiapi.tech/v1"

MODEL_CONFIGS = {
    "deepseek": ("deepseek-ai/DeepSeek-V3",          API_KEY_SF,  BASE_URL_SF),
    "qwen":     ("Qwen/Qwen2.5-72B-Instruct",         API_KEY_SF,  BASE_URL_SF),
    "gpt4o":    ("gpt-4o-2024-11-20",                 API_KEY_OAI, BASE_URL_OAI),
}

BASE_DIR = Path(__file__).resolve().parent  # evaluation/

DATASET_CONFIGS = {
    "medical_o1":  BASE_DIR / "comparison_data/medical_o1_100_for_eval.json",
    "chatdoctor":  BASE_DIR / "comparison_data/chatdoctor_100_for_eval.json",
    "medquad":     BASE_DIR / "comparison_data/medquad_100_for_eval.json",
    "reasonmed":   BASE_DIR / "reasonmed_for_eval.json",  # full file, we sample 100
}

EVALUATOR_PROMPT = """You are a biomedical QA scoring expert. Score strictly on 3 dimensions only.

【Scoring Dimensions】
1. Factuality (0-4, step 0.5)
   4.0 = core mechanism 100% correct, all claims evidenced
   3.5 = mostly correct, minor imprecision
   3.0 = correct overall, a few details imprecise
   2.5 = direction correct, some key mechanisms unsupported
   2.0 = direction correct, key mechanism lacks evidence
   1.0 = major factual errors
   0.0 = completely wrong or contradicts evidence

2. Mechanism Granularity (0-3, step 0.5)
   3.0 = full molecular mechanism (action type + functional change + downstream effect)
   2.5 = nearly complete, minor details missing
   2.0 = clear molecular functional change described
   1.0 = only inhibit/activate stated, no molecular detail
   0.0 = no mechanistic information

3. Evidence Grounding (0-3, step 0.5)
   3.0 = every claim linked to PMID / UniProt / DrugBank
   2.5 = evidence sufficient but not all claims bound
   2.0 = citations present but weakly linked
   1.0 = citations listed but unlinked
   0.0 = no verifiable evidence

【Output format — strictly follow this】
1. Factuality: X.X
2. Mechanism Granularity: X.X
3. Evidence Grounding: X.X
4. Total = sum of the three
5. Brief rationale (2-3 sentences)

Score objectively. Do NOT comment on format, length, or writing style."""


def parse_scores(text):
    patterns = {
        "factuality":  r"Factuality[：:]\s*(\d+\.?\d*)",
        "granularity": r"(?:Mechanism )?Granularity[：:]\s*(\d+\.?\d*)",
        "grounding":   r"(?:Evidence )?Grounding[：:]\s*(\d+\.?\d*)",
    }
    scores = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        scores[key] = float(match.group(1)) if match else 0.0
    scores["total"] = round(sum(scores.values()), 1)
    return scores


def compute_stats(results):
    if not results:
        return {}
    n = len(results)
    return {
        "count": n,
        "factuality": round(sum(r["factuality"] for r in results) / n, 3),
        "granularity": round(sum(r["granularity"] for r in results) / n, 3),
        "grounding": round(sum(r["grounding"] for r in results) / n, 3),
        "total": round(sum(r["total"] for r in results) / n, 3),
    }


def load_dataset(dataset_name):
    """Load dataset, handling sampling for large files."""
    path = Path(DATASET_CONFIGS[dataset_name])
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle reasonmed: it's a large list — sample 100
    if dataset_name == "reasonmed" and len(data) > 100:
        import random
        rng = random.Random(42)
        data = rng.sample(data, 100)
        print(f"  Sampled 100 from {len(data)} reasonmed entries")

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASET_CONFIGS.keys()) + ["all"])
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    model_id, api_key, base_url = MODEL_CONFIGS[args.model]
    client = OpenAI(api_key=api_key, base_url=base_url)

    datasets_to_run = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]

    for ds_name in datasets_to_run:
        output_path = BASE_DIR / f"comparison_data/{ds_name}_evaluated_{args.model}.json"
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} | Evaluator: {args.model} ({model_id})")
        print(f"Output: {output_path}")
        print(f"{'='*60}")

        data = load_dataset(ds_name)
        print(f"Loaded {len(data)} samples")

        # Resume from checkpoint
        results = []
        existing_keys = set()
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            existing_keys = {r.get("q_hash", "") for r in results}
            print(f"Resumed {len(results)} from checkpoint")

        for idx, item in enumerate(data, 1):
            question = item.get("question", "")
            answer = item.get("answer", "")
            q_key = hashlib.md5((question + answer[:100]).encode('utf-8')).hexdigest()

            if q_key in existing_keys:
                continue

            print(f"[{idx}/{len(data)}] ", end="", flush=True)

            user_msg = (f"Please score the following biomedical QA pair:\n\n"
                        f"Question: {question}\n\nAnswer: {answer}\n\n"
                        f"Strictly follow the output format.")

            raw = None
            for attempt in range(5):
                try:
                    resp = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": EVALUATOR_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.0,
                        max_tokens=600,
                        timeout=90,
                    )
                    raw = resp.choices[0].message.content or ""
                    break
                except Exception as e:
                    err = str(e)
                    wait = 30 if ("429" in err or "rate" in err.lower()) else 10
                    print(f"(retry {attempt+1}: {err[:50]}) ", end="", flush=True)
                    time.sleep(wait)

            if raw is None:
                print("FAILED", flush=True)
                continue

            scores = parse_scores(raw)
            result = {
                "q_hash": q_key,
                "question": question[:300],
                "answer": answer[:300],
                "factuality": scores["factuality"],
                "granularity": scores["granularity"],
                "grounding": scores["grounding"],
                "total": scores["total"],
                "raw_eval": raw[:300],
                "source": ds_name,
            }
            results.append(result)
            existing_keys.add(q_key)

            print(f"F:{scores['factuality']:.1f} Gr:{scores['granularity']:.1f} Gnd:{scores['grounding']:.1f} T:{scores['total']:.1f}", flush=True)

            # REAL-TIME SAVE every single item
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            time.sleep(0.5)  # gentle rate limit

        # Final stats
        stats = compute_stats(results)
        print(f"\n{'='*60}")
        print(f"DONE: {ds_name} @ {args.model}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}")

        # Save stats
        stats_path = BASE_DIR / f"comparison_data/{ds_name}_stats_{args.model}.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
