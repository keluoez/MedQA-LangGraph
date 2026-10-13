# -*- coding: utf-8 -*-
"""
Shared evaluator used by BOTH baseline_comparison and ablation_study.

Supports three independent evaluator models:
  - DeepSeek-V3      (SiliconFlow)
  - Qwen2.5-72B      (SiliconFlow)
  - GPT-4o           (OpenAI)

Usage:
    from shared.evaluator import evaluate_batch
    results = evaluate_batch(items, model="deepseek")
"""

import re
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.api_config import (
    SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL,
    OPENAI_API_KEY, OPENAI_BASE_URL,
    DEEPSEEK_MODEL, QWEN_MODEL, GPT4O_MODEL,
)

# ── Evaluator prompt (model-agnostic) ─────────────────────────────────────────
EVALUATOR_SYSTEM = """You are a biomedical QA scoring expert. Score strictly on 3 dimensions only.

【Scoring Dimensions】
1. Factuality (0–4, step 0.5)
   4.0 = core mechanism 100% correct, all claims evidenced
   3.5 = mostly correct, minor imprecision
   3.0 = correct overall, a few details imprecise
   2.5 = direction correct, some key mechanisms unsupported
   2.0 = direction correct, key mechanism lacks evidence
   1.0 = major factual errors
   0.0 = completely wrong or contradicts evidence

2. Mechanism Granularity (0–3, step 0.5)
   3.0 = full molecular mechanism (action type + functional change + downstream effect)
   2.5 = nearly complete, minor details missing
   2.0 = clear molecular functional change described
   1.0 = only inhibit/activate stated, no molecular detail
   0.0 = no mechanistic information

3. Evidence Grounding (0–3, step 0.5)
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
5. Brief rationale (2–3 sentences)

Score objectively. Do NOT comment on format, length, or writing style."""


def _make_client(model_key: str):
    if model_key == "gpt4o":
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL), GPT4O_MODEL
    else:
        return OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL), \
               DEEPSEEK_MODEL if model_key == "deepseek" else QWEN_MODEL


def _parse(text: str) -> dict:
    patterns = {
        "factuality":  r"Factuality[：:]\s*(\d+\.?\d*)",
        "granularity": r"Mechanism Granularity[：:]\s*(\d+\.?\d*)",
        "grounding":   r"Evidence Grounding[：:]\s*(\d+\.?\d*)",
    }
    scores = {}
    for k, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        scores[k] = float(m.group(1)) if m else 0.0
    scores["total"] = sum(scores.values())
    return scores


def _evaluate_one(item: dict, model_key: str, max_retry: int = 3) -> dict:
    client, model = _make_client(model_key)
    question = item.get("question", "")
    answer   = item.get("answer", "")
    user_msg = (f"Please score the following biomedical QA pair:\n\n"
                f"Question: {question}\n\nAnswer: {answer}\n\n"
                f"Strictly follow the output format.")

    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EVALUATOR_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=600,
                timeout=90,
            )
            raw = resp.choices[0].message.content or ""
            scores = _parse(raw)
            return {
                "question":      question[:200],
                "answer":        answer[:300],
                "evaluator":     model_key,
                "factuality":    scores["factuality"],
                "granularity":   scores["granularity"],
                "grounding":     scores["grounding"],
                "total":         scores["total"],
                "raw_eval":      raw[:400],
                "variant":       item.get("variant", "unknown"),
                "source":        item.get("source", "unknown"),
            }
        except Exception as e:
            err = str(e)
            wait = 15 * (attempt + 1) if ("429" in err or "rate" in err.lower()) else 5
            if attempt < max_retry - 1:
                time.sleep(wait)
            else:
                return {"error": err, "question": question[:100],
                        "variant": item.get("variant", ""), "evaluator": model_key}
    return {"error": "max_retry", "evaluator": model_key}


def evaluate_batch(items: list, model: str = "deepseek",
                   max_workers: int = 8, checkpoint_file: str = None) -> list:
    """
    Evaluate a list of QA dicts in parallel.
    model: "deepseek" | "qwen" | "gpt4o"
    checkpoint_file: If provided, saves progress real-time as JSONL and resumes from it.
    """
    results = []
    completed_keys = set()
    
    # 1. Load existing checkpoint if available
    if checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    r = json.loads(line)
                    results.append(r)
                    # Use question+variant as unique key
                    completed_keys.add(r.get("question", "") + "||" + r.get("variant", ""))
                except:
                    pass
        print(f"  [Checkpoint] Resumed {len(results)} items from {checkpoint_file}")

    # 2. Filter out already completed items
    items_to_run = []
    for it in items:
        k = it.get("question", "") + "||" + it.get("variant", "")
        if k not in completed_keys:
            items_to_run.append(it)
            
    if not items_to_run:
        print("  [Checkpoint] All items already evaluated.")
        return results

    # 3. Open checkpoint file in append mode
    f_out = open(checkpoint_file, "a", encoding="utf-8") if checkpoint_file else None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_evaluate_one, it, model): it for it in items_to_run}
        done = 0
        total = len(items_to_run)
        for f in as_completed(futures):
            done += 1
            r = f.result()
            results.append(r)
            
            # Real-time save
            if f_out:
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                f_out.flush()
                
            status = "OK" if "error" not in r else f"ERR:{r['error'][:40]}"
            print(f"  [{done}/{total}] evaluator={model} | {status}", flush=True)
            
    if f_out:
        f_out.close()
        
    return results


def summarize(results: list) -> dict:
    """Return mean scores grouped by 'variant' field."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if "error" not in r:
            groups[r.get("variant", "unknown")].append(r)

    summary = {}
    for variant, rows in groups.items():
        n = len(rows)
        summary[variant] = {
            "n":           n,
            "mean_factuality":  round(sum(r["factuality"] for r in rows) / n, 3),
            "mean_granularity": round(sum(r["granularity"] for r in rows) / n, 3),
            "mean_grounding":   round(sum(r["grounding"]   for r in rows) / n, 3),
            "mean_total":       round(sum(r["total"]       for r in rows) / n, 3),
        }
    return summary
