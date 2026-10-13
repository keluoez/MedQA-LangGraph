import json
import os
import time
from pathlib import Path
from openai import OpenAI

# 配置
# 加载项目根目录 .env（密钥配置）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL = "deepseek-ai/DeepSeek-V3"

OUTPUT_FILE = Path("reasonmed_100_evaluated_deepseek.json")
INPUT_FILE = Path("reasonmed_for_eval.json")

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
    """从评测结果中解析分数"""
    import re
    scores = {}

    patterns = [
        ("factuality", r"事实正确性[：:]\s*(\d+\.?\d*)"),
        ("granularity", r"机制精细度[：:]\s*(\d+\.?\d*)"),
        ("grounding", r"证据可追溯性[：:]\s*(\d+\.?\d*)"),
    ]

    for key, pattern in patterns:
        match = re.search(pattern, text)
        if match:
            scores[key] = float(match.group(1))
        else:
            scores[key] = 0.0

    scores["total"] = scores.get("factuality", 0) + scores.get("granularity", 0) + scores.get("grounding", 0)
    return scores

def evaluate_single(client, question, answer, retry_count=3):
    """评测单个样本，带重试机制"""
    user_prompt = f"""请严格按照评分标准对以下QA进行评分：

问题：{question}

答案：{answer}

请按要求的格式输出评分结果。"""

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=MODEL,
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
            if "429" in error_msg or "rate limit" in error_msg.lower():
                print(f"    API限流，等待{30*(attempt+1)}秒后重试...")
                time.sleep(30 * (attempt + 1))
            else:
                return f"Error: {error_msg}"
    return "Error: Max retries exceeded"

def main():
    print("="*70)
    print("ReasonMed 100条数据评测")
    print("="*70)
    print(f"评测模型: {MODEL}")
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出文件: {OUTPUT_FILE}")
    print("="*70)

    # 加载数据
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    print(f"总样本数: {total}")

    # 检查是否已有部分结果
    existing_results = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing = json.load(f)
            for item in existing:
                existing_results[item.get("question", "")[:50]] = item
        print(f"已有评测结果: {len(existing_results)}条")

    results = []
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    for idx, item in enumerate(data, 1):
        question = item.get("question", "")
        answer = item.get("answer", "")
        q_key = question[:50]

        print(f"\n[{idx}/{total}] 评测中...", end=" ", flush=True)

        # 检查是否已评测
        if q_key in existing_results:
            result = existing_results[q_key]
            print(f"已存在 (F:{result['factuality']}, G:{result['granularity']}, Gr:{result['grounding']}, Total:{result['total']})")
            results.append(result)
            continue

        # 评测
        raw_output = evaluate_single(client, question, answer)
        scores = parse_scores(raw_output)

        result = {
            "question": question[:200] + "..." if len(question) > 200 else question,
            "answer": answer[:300] + "..." if len(answer) > 300 else answer,
            "factuality": scores.get("factuality", 0),
            "granularity": scores.get("granularity", 0),
            "grounding": scores.get("grounding", 0),
            "total": scores.get("total", 0),
            "raw_evaluation": raw_output[:500] if len(raw_output) > 500 else raw_output,
            "source": "ReasonMed"
        }

        results.append(result)

        print(f"完成 (F:{result['factuality']}, G:{result['granularity']}, Gr:{result['grounding']}, Total:{result['total']})")

        # 保存当前结果
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 避免API限流 - 每次请求后等待
        time.sleep(8)

    print(f"\n{'='*70}")
    print(f"评测完成！共{len(results)}条数据")
    print(f"结果已保存到: {OUTPUT_FILE}")
    print("="*70)

    # 计算统计
    if results:
        avg_f = sum(r["factuality"] for r in results) / len(results)
        avg_g = sum(r["granularity"] for r in results) / len(results)
        avg_gr = sum(r["grounding"] for r in results) / len(results)
        avg_t = sum(r["total"] for r in results) / len(results)

        print(f"\n平均Factuality: {avg_f:.2f}")
        print(f"平均Granularity: {avg_g:.2f}")
        print(f"平均Grounding: {avg_gr:.2f}")
        print(f"平均总分: {avg_t:.2f}")

if __name__ == "__main__":
    main()
