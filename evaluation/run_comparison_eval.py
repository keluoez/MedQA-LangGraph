#!/usr/bin/env python3
"""
Unified comparison evaluation script.
Evaluates multiple datasets with multiple evaluators.
Supports checkpoint/resume to handle API failures.

Usage:
  python run_comparison_eval.py --dataset medical_o1 --model deepseek
  python run_comparison_eval.py --dataset chatdoctor --model qwen
  python run_comparison_eval.py --dataset medquad --model deepseek
"""

import json, os, time, re, argparse
from pathlib import Path
from openai import OpenAI

# ===== CONFIGURATION =====
# 加载项目根目录 .env（密钥配置）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
BASE_URL = "https://api.siliconflow.cn/v1"

MODEL_CONFIGS = {
    "deepseek": "deepseek-ai/DeepSeek-V3",
    "qwen": "Qwen/Qwen2.5-72B-Instruct",
}

DATASET_CONFIGS = {
    "medical_o1": "comparison_data/medical_o1_100_for_eval.json",
    "chatdoctor": "comparison_data/chatdoctor_100_for_eval.json",
    "medquad": "comparison_data/medquad_100_for_eval.json",
}

EVALUATOR_PROMPT = """你是生物医药QA数据集自动评分专家，只做严格、可复现的量化评分。

任务：对药物-靶点作用机制QA数据进行自动评分。
规则：绝对不评价格式、结构、长度、文风，只评价以下3个维度，严格按量化标准打分。

====================
【固定评分维度 · 严格量化标准】
====================

1. 事实正确性 Factuality（0～4 分，精确到0.5）
4.0：核心机制100%正确，所有陈述均有依据
3.5：基本正确，仅有轻微不精确
3.0：整体正确，少量细节不够精准
2.5：方向正确，但部分关键机制无依据
2.0：方向正确，但关键机制缺少证据支撑
1.5：存在明显事实偏差
1.0：机制大部分错误或无依据
0.5：核心机制错误
0.0：完全错误，与证据矛盾

2. 机制精细度 Mechanism Granularity（0～3 分，精确到0.5）
3.0：完整分子机制（作用类型+功能变化+下游效应）
2.5：接近完整，缺少量细节
2.0：有明确分子功能变化
1.5：能说明作用，但不够精细
1.0：仅说明抑制/激活，无分子细节
0.5：仅描述药理效应，无机制
0.0：无任何机制信息

3. 证据可追溯性 Grounding（0～3 分，精确到0.5）
3.0：每条机制都对应 PMID / UniProt / DrugBank
2.5：证据充分，但未逐条绑定
2.0：有引用，但关联较弱
1.5：有引用，但不明确
1.0：仅列出引用，无对应关系
0.5：仅有数据库名，无具体ID
0.0：无任何可验证证据

====================
【输出要求】
====================
对每条QA输出：
1. 事实正确性：X.X
2. 机制精细度：X.X
3. 证据可追溯性：X.X
4. 总分 = 三项相加
5. 简要理由（2～3句话）

只输出客观评分，不修改原文，不补充知识。"""


def parse_scores(text):
    patterns = [
        ("factuality", r"事实正确性[：:]\s*(\d+\.?\d*)"),
        ("granularity", r"机制精细度[：:]\s*(\d+\.?\d*)"),
        ("grounding", r"证据可追溯性[：:]\s*(\d+\.?\d*)"),
    ]
    scores = {}
    for key, pattern in patterns:
        match = re.search(pattern, text)
        scores[key] = float(match.group(1)) if match else 0.0
    scores["total"] = sum(scores.values())
    return scores


def evaluate_single(client, question, answer, model_id, retry_count=5):
    user_prompt = f"""请严格按照评分标准对以下QA进行评分：

问题：{question}

答案：{answer}

请按要求的格式输出评分结果。"""

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": EVALUATOR_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=800,
                timeout=90,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate" in error_msg.lower():
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            elif "503" in error_msg or "502" in error_msg:
                wait = 10 * (attempt + 1)
                print(f"  Server error, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"  Error: {error_msg[:100]}", flush=True)
                time.sleep(5)
    return "Error: Max retries exceeded"


def compute_stats(results):
    if not results:
        return {}
    n = len(results)
    return {
        "count": n,
        "factuality_mean": round(sum(r["factuality"] for r in results) / n, 3),
        "granularity_mean": round(sum(r["granularity"] for r in results) / n, 3),
        "grounding_mean": round(sum(r["grounding"] for r in results) / n, 3),
        "total_mean": round(sum(r["total"] for r in results) / n, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    model_id = MODEL_CONFIGS[args.model]
    input_path = Path(DATASET_CONFIGS[args.dataset])
    output_path = Path(f"comparison_data/{args.dataset}_evaluated_{args.model}.json")

    print(f"{'='*60}")
    print(f"Dataset: {args.dataset} | Evaluator: {args.model} ({model_id})")
    print(f"Input: {input_path} | Output: {output_path}")
    print(f"{'='*60}")

    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")

    # Resume from checkpoint (use hash of full question for unique key)
    import hashlib
    results = []
    existing_keys = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        existing_keys = {r.get("q_hash", "") for r in results}
        print(f"Resuming from checkpoint: {len(results)} already evaluated")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    for idx, item in enumerate(data, 1):
        question = item.get("question", "")
        answer = item.get("answer", "")
        q_key = hashlib.md5((question + answer[:100]).encode('utf-8')).hexdigest()

        if q_key in existing_keys:
            continue

        print(f"\n[{idx}/{len(data)}] ", end="", flush=True)
        raw = evaluate_single(client, question, answer, model_id)
        scores = parse_scores(raw)

        result = {
            "q_hash": q_key,
            "question": question[:200] + "..." if len(question) > 200 else question,
            "answer": answer[:300] + "..." if len(answer) > 300 else answer,
            "factuality": scores["factuality"],
            "granularity": scores["granularity"],
            "grounding": scores["grounding"],
            "total": scores["total"],
            "raw_evaluation": raw[:500] if len(raw) > 500 else raw,
            "source": item.get("source", args.dataset),
        }
        results.append(result)
        existing_keys.add(q_key)

        print(f"F:{scores['factuality']:.1f} G:{scores['granularity']:.1f} Gr:{scores['grounding']:.1f} T:{scores['total']:.1f}", flush=True)

        # Save checkpoint every 10
        if idx % 10 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(5)  # rate limiting

    # Final save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Stats
    stats = compute_stats(results)
    print(f"\n{'='*60}")
    print(f"COMPLETE: {args.dataset} @ {args.model}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
