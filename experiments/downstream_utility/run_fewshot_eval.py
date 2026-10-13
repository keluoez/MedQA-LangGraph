# -*- coding: utf-8 -*-
"""
Downstream Utility Validation — Few-Shot PubMedQA Generation Evaluation.

Instead of yes/no classification (where PubMedQA is inherently a reading
comprehension task), this script evaluates whether MedQA-LangGraph few-shot
examples improve the QUALITY of free-text biomedical answers.

Three conditions:
  1. Zero-shot: Generate answer from PubMedQA context only
  2. Few-shot ReasonMed: 3 ReasonMed examples + generate answer
  3. Few-shot MedQA-LangGraph: 3 Gold examples + generate answer

Evaluation: DeepSeek-V3 scores all generated answers on Factuality (0–4),
Granularity (0–3), Grounding (0–3). Total = 0–10.

Hypothesis: MedQA-LangGraph few-shot produces higher Grounding and Factuality
scores because its structured evidence-grounded format teaches the model to
anchor claims in the provided context.
"""

import json
import os
import re
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.api_config import NVIDIA_API_KEY, NVIDIA_BASE_URL, GENERATION_MODEL
from shared.evaluator import evaluate_batch, summarize
from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────────────
PUBMEDQA_PATH    = os.path.join(os.path.dirname(__file__), "pubmedqa.json")
OUR_DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation",
                                "our_data_100_full_answers.json")
OUR_EVAL_PATH    = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation",
                                "our_data_100_evaluated_deepseek.json")
REASONMED_FULL   = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation",
                                "reasonmed_for_eval.json")
REASONMED_EVAL   = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation",
                                "reasonmed_100_evaluated_final.json")
OUTPUT_DIR       = os.path.join(os.path.dirname(__file__), "output")

N_TEST           = 100
N_SHOTS          = 3
MAX_WORKERS_GEN  = 6

# ── Prompts ───────────────────────────────────────────────────────────────────

ZERO_SHOT_SYSTEM = """You are a biomedical research assistant. Answer the question based on the provided context.
Be specific about molecular mechanisms. Cite evidence from the context.
Keep your answer concise (3-5 sentences)."""

FEWSHOT_SYSTEM = """You are a biomedical research assistant. Study the example QA pairs below to understand
how to write evidence-grounded biomedical answers, then answer a new question.
Be specific about molecular mechanisms. Keep your answer concise (3-5 sentences)."""


def load_pubmedqa():
    with open(PUBMEDQA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for pmid, v in data.items():
        items.append({
            "pmid": pmid,
            "question": v["QUESTION"],
            "contexts": v.get("CONTEXTS", []),
            "label": v.get("final_decision", "unknown").strip().lower(),
            "long_answer": v.get("LONG_ANSWER", ""),
        })
    return items


def load_our_gold():
    with open(OUR_DATA_PATH, "r", encoding="utf-8") as f:
        full = json.load(f)
    with open(OUR_EVAL_PATH, "r", encoding="utf-8") as f:
        evals = json.load(f)
    eval_by_q = {e["question"]: e for e in evals}
    gold = []
    for item in full:
        q = item.get("question", "")
        e = eval_by_q.get(q, {})
        gold.append({
            "question": q,
            "answer": item.get("answer", ""),
            "total": e.get("total", 0),
        })
    gold.sort(key=lambda x: x["total"], reverse=True)
    return gold


def load_reasonmed():
    with open(REASONMED_FULL, "r", encoding="utf-8") as f:
        full = json.load(f)
    with open(REASONMED_EVAL, "r", encoding="utf-8") as f:
        evals = json.load(f)
    eval_by_q = {e["question"]: e for e in evals if "question" in e}
    items = []
    for item in full:
        q = item.get("question", "")
        e = eval_by_q.get(q, {})
        items.append({
            "question": q,
            "answer": item.get("answer", ""),
            "total": e.get("total", 0),
        })
    items.sort(key=lambda x: x["total"], reverse=True)
    return items


def format_our_example(item):
    body = item["answer"]
    if len(body) > 1000:
        body = body[:1000] + "\n[...]"
    return f"Q: {item['question']}\nA: {body}"


def format_reasonmed_example(item):
    body = item["answer"]
    if len(body) > 1000:
        body = body[:1000] + "\n[...]"
    return f"Q: {item['question']}\nA: {body}"


def format_pubmedqa_prompt(pubmedqa_item):
    contexts = pubmedqa_item.get("contexts", [])
    ctx_text = " ".join(contexts)
    if len(ctx_text) > 2500:
        ctx_text = ctx_text[:2500]
    return (f"Context: {ctx_text}\n\n"
            f"Question: {pubmedqa_item['question']}")


def generate_one(client, model, condition, test_item, fewshot_examples=None,
                 max_retry=3):
    """Generate a free-text answer for one PubMedQA item."""
    if condition == "zero_shot":
        messages = [
            {"role": "system", "content": ZERO_SHOT_SYSTEM},
            {"role": "user", "content": format_pubmedqa_prompt(test_item)},
        ]
    else:
        if condition == "fewshot_ours":
            fmt_func = format_our_example
        else:
            fmt_func = format_reasonmed_example

        examples_block = []
        for i, ex in enumerate(fewshot_examples, 1):
            examples_block.append(f"--- Example {i} ---")
            examples_block.append(fmt_func(ex))
            examples_block.append("")
        examples_text = "\n".join(examples_block)

        user_msg = (f"Here are some examples of evidence-grounded biomedical QA:\n\n"
                    f"{examples_text}\n"
                    f"Now, following the same rigorous style, answer this question:\n\n"
                    f"{format_pubmedqa_prompt(test_item)}")

        messages = [
            {"role": "system", "content": FEWSHOT_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=500,
                timeout=90,
            )
            raw = resp.choices[0].message.content or ""
            return {
                "question": test_item["question"][:300],
                "answer": raw.strip(),
                "variant": condition,
                "source": f"pubmedqa_{test_item['pmid']}",
            }
        except Exception as exc:
            err = str(exc)
            wait = 15 * (attempt + 1) if ("429" in err or "rate" in err.lower()) else 5
            if attempt < max_retry - 1:
                time.sleep(wait)
            else:
                return {
                    "question": test_item["question"][:300],
                    "answer": f"[ERROR: {err}]",
                    "variant": condition,
                    "source": f"pubmedqa_{test_item['pmid']}",
                }
    return None


def generate_batch(test_items, condition, fewshot_examples, label, checkpoint_file):
    """Generate answers for all test items, with checkpointing."""
    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    model = GENERATION_MODEL

    results = []
    completed = set()

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    results.append(r)
                    completed.add(r.get("source", ""))
                except:
                    pass
        print(f"  [Resumed] {len(results)} items from checkpoint")

    to_run = [it for it in test_items
              if f"pubmedqa_{it['pmid']}" not in completed]
    if not to_run:
        print(f"  All {len(results)} items already generated.")
        return results

    print(f"  Generating {len(to_run)} answers ({label})...")
    f_out = open(checkpoint_file, "a", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_GEN) as ex:
        futures = {
            ex.submit(generate_one, client, model, condition, it, fewshot_examples): it
            for it in to_run
        }
        done = 0
        for f in as_completed(futures):
            done += 1
            r = f.result()
            results.append(r)
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
            f_out.flush()
            if done % 20 == 0 or done == len(to_run):
                print(f"  [{done}/{len(to_run)}] {label}", flush=True)

    f_out.close()
    return results


def main():
    print("=" * 70)
    print("Downstream Utility Validation — Free-Text Generation Quality")
    print("=" * 70)

    # 1. Load data
    print("\n[1/5] Loading data...")
    pubmedqa = load_pubmedqa()
    our_gold = load_our_gold()
    reasonmed = load_reasonmed()
    print(f"  PubMedQA: {len(pubmedqa)} items")
    print(f"  MedQA-LangGraph Gold: {len(our_gold)} (top={our_gold[0]['total']:.1f})")
    print(f"  ReasonMed: {len(reasonmed)} (top={reasonmed[0]['total']:.1f})")

    # 2. Select few-shot examples
    print("\n[2/5] Selecting few-shot examples...")
    our_shots = our_gold[:N_SHOTS]
    rm_shots  = reasonmed[:N_SHOTS]
    for i, ex in enumerate(our_shots):
        print(f"  Our Shot {i+1}: {ex['question'][:80]} (score={ex['total']:.1f})")
    for i, ex in enumerate(rm_shots):
        print(f"  RM Shot  {i+1}: {ex['question'][:80]} (score={ex['total']:.1f})")

    # 3. Sample test items (balanced across labels)
    print(f"\n[3/5] Sampling {N_TEST} PubMedQA test items...")
    random.seed(42)
    by_label = {"yes": [], "no": [], "maybe": []}
    for item in pubmedqa:
        lbl = item["label"]
        if lbl in by_label:
            by_label[lbl].append(item)

    test_items = []
    for lbl, items in by_label.items():
        n = N_TEST * len(items) // len(pubmedqa)
        sampled = random.sample(items, min(n, len(items)))
        test_items.extend(sampled)
    random.shuffle(test_items)
    print(f"  Test set: {len(test_items)} items")
    for lbl in ["yes", "no", "maybe"]:
        c = sum(1 for it in test_items if it["label"] == lbl)
        print(f"    {lbl}: {c}")

    # 4. Generate answers for all 3 conditions
    print("\n[4/5] Generating answers (3 conditions)...")

    conditions = [
        ("zero_shot", None, "zero_shot",
         os.path.join(OUTPUT_DIR, "gen_zero_shot.jsonl")),
        ("fewshot_reasonmed", rm_shots, "fewshot_reasonmed",
         os.path.join(OUTPUT_DIR, "gen_fewshot_reasonmed.jsonl")),
        ("fewshot_ours", our_shots, "fewshot_ours",
         os.path.join(OUTPUT_DIR, "gen_fewshot_ours.jsonl")),
    ]

    all_generated = {}
    for cond_key, shots, label, ckpt in conditions:
        print(f"\n--- Generating: {label} ---")
        results = generate_batch(test_items, cond_key, shots, label, ckpt)
        all_generated[label] = results
        print(f"  Generated {len(results)} answers for {label}")

    # 5. Evaluate all generated answers with DeepSeek-V3
    print("\n[5/5] Evaluating all generated answers with DeepSeek-V3...")
    eval_results = {}

    for label, generated in all_generated.items():
        print(f"\n--- Evaluating: {label} ({len(generated)} items) ---")
        ckpt_file = os.path.join(OUTPUT_DIR, f"eval_{label}_deepseek.jsonl")
        results = evaluate_batch(generated, model="deepseek",
                                 max_workers=8, checkpoint_file=ckpt_file)
        eval_results[label] = results
        summary = summarize(results)
        for variant, m in summary.items():
            print(f"  {variant}: {m['mean_total']:.3f}  "
                  f"(F={m['mean_factuality']:.3f} G={m['mean_granularity']:.3f} "
                  f"E={m['mean_grounding']:.3f})  n={m['n']}")

    # 6. Save final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    final = {}
    for label, results in eval_results.items():
        s = summarize(results)
        final[label] = s

    summary_path = os.path.join(OUTPUT_DIR, "downstream_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\n{'Condition':<30} {'Total':>8}  {'Fact':>6} {'Gran':>6} {'Grnd':>6}")
    print("-" * 65)
    for label, s in final.items():
        for variant, m in s.items():
            print(f"{label:<30} {m['mean_total']:>8.3f}  "
                  f"{m['mean_factuality']:>6.3f} {m['mean_granularity']:>6.3f} "
                  f"{m['mean_grounding']:>6.3f}  (n={m['n']})")

    print("\nDone. Summary:", summary_path)
    return final


if __name__ == "__main__":
    main()
