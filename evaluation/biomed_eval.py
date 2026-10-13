import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent

# 加载项目根目录 .env（密钥配置）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("BIOREASON_API_KEY") or os.getenv("SILICONFLOW_API_KEY") or ""
BASE_URL = os.getenv("BIOREASON_BASE_URL", "https://api.siliconflow.cn/v1")
MODEL = os.getenv("BIOREASON_MODEL", "Qwen/Qwen2.5-72B-Instruct")

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


def get_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def llm_evaluate(client: OpenAI, question: str, answer: str, evidence: str = "") -> Dict[str, Any]:
    user_prompt = f"""请严格按照评分标准对以下QA进行评分：

问题：{question}

答案：{answer}
"""
    if evidence:
        user_prompt += f"\n参考证据：\n{evidence}\n"
    user_prompt += "\n请按要求的格式输出评分结果。"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EVALUATOR_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=800,
        timeout=60,
    )

    content = response.choices[0].message.content or ""
    return content


def parse_score(text: str, score_name: str) -> float:
    import re
    patterns = [
        rf"{score_name}[：:]\s*(\d+\.?\d*)",
        rf"{score_name}\s*=\s*(\d+\.?\d*)",
        rf"(\d+\.?\d*)\s*分",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            score = float(match.group(1))
            return round(score * 2) / 2
    return 0.0


def parse_total_score(text: str) -> float:
    import re
    patterns = [
        r"总分\s*[=：:]\s*(\d+\.?\d*)",
        r"总分\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*分\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return 0.0


def extract_reason(text: str) -> str:
    lines = text.strip().split("\n")
    reason_lines = []
    in_reason = False
    for line in lines:
        if any(keyword in line.lower() for keyword in ["理由", "reason", "说明", "因为"]):
            in_reason = True
        if in_reason and line.strip():
            reason_lines.append(line.strip())
    if reason_lines:
        return " ".join(reason_lines[:3])
    return text.strip().split("\n")[-1] if lines else ""


def evaluate_single(client: OpenAI, item: Dict[str, Any]) -> Dict[str, Any]:
    question = item.get("question", "")
    answer = item.get("answer", "")
    evidence = item.get("evidence", "")

    raw_output = llm_evaluate(client, question, answer, evidence)

    factuality = parse_score(raw_output, "事实正确性")
    granularity = parse_score(raw_output, "机制精细度")
    grounding = parse_score(raw_output, "证据可追溯性")
    total = factuality + granularity + grounding

    return {
        "question": question,
        "answer": answer[:200] + "..." if len(answer) > 200 else answer,
        "factuality": factuality,
        "granularity": granularity,
        "grounding": grounding,
        "total": round(total, 1),
        "reason": extract_reason(raw_output),
        "raw_output": raw_output,
    }


def evaluate_batch(input_file: Path, output_file: Path = None, max_items: int = None):
    client = get_client()

    if input_file.suffix == ".json":
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("results", [data])
        else:
            raise ValueError("Unsupported JSON format")
    else:
        raise ValueError("Only JSON files supported")

    if max_items:
        items = items[:max_items]

    results = []
    for idx, item in enumerate(items, 1):
        print(f"[{idx}/{len(items)}] Evaluating...", flush=True)
        try:
            result = evaluate_single(client, item)
            results.append(result)
            print(f"  Factuality: {result['factuality']}, Granularity: {result['granularity']}, Grounding: {result['grounding']}, Total: {result['total']}", flush=True)
        except Exception as e:
            print(f"  Error: {e}", flush=True)
            results.append({
                "question": item.get("question", ""),
                "error": str(e),
                "factuality": 0.0,
                "granularity": 0.0,
                "grounding": 0.0,
                "total": 0.0,
                "reason": "",
            })

    if not output_file:
        output_file = input_file.parent / f"{input_file.stem}_evaluated.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, ensure_ascii=False, indent=2, fp=f)

    print(f"\nEvaluation completed!")
    print(f"Results saved to: {output_file}")

    stats = {
        "count": len(results),
        "avg_factuality": round(sum(r["factuality"] for r in results) / len(results), 2) if results else 0,
        "avg_granularity": round(sum(r["granularity"] for r in results) / len(results), 2) if results else 0,
        "avg_grounding": round(sum(r["grounding"] for r in results) / len(results), 2) if results else 0,
        "avg_total": round(sum(r["total"] for r in results) / len(results), 2) if results else 0,
    }

    print(f"\n=== Summary Statistics ===")
    print(f"Total items: {stats['count']}")
    print(f"Average Factuality: {stats['avg_factuality']}")
    print(f"Average Granularity: {stats['avg_granularity']}")
    print(f"Average Grounding: {stats['avg_grounding']}")
    print(f"Average Total: {stats['avg_total']}")

    return results, stats


def main():
    parser = argparse.ArgumentParser(description="BioMed QA 自动评分工具")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入JSON文件路径")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出JSON文件路径")
    parser.add_argument("--count", "-c", type=int, default=None, help="最多评估多少条")
    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return

    output_file = Path(args.output) if args.output else None
    evaluate_batch(input_file, output_file, args.count)


if __name__ == "__main__":
    main()
