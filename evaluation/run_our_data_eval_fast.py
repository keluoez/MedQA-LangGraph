import json
import os
import time
import sys
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置
# 加载项目根目录 .env（密钥配置）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
BASE_URL = "https://api.siliconflow.cn/v1"

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

def evaluate_single(item, model, retry_count=3):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    question = item.get("question", "")
    answer = item.get("answer", "")
    user_prompt = f"""请严格按照评分标准对以下QA进行评分：\n\n问题：{question}\n\n答案：{answer}\n\n请按要求的格式输出评分结果。"""
    
    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EVALUATOR_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=800,
                timeout=90,
            )
            raw_output = response.choices[0].message.content
            scores = parse_scores(raw_output)
            return {
                "question": question[:200] + "..." if len(question) > 200 else question,
                "answer": answer[:300] + "..." if len(answer) > 300 else answer,
                "factuality": scores.get("factuality", 0),
                "granularity": scores.get("granularity", 0),
                "grounding": scores.get("grounding", 0),
                "total": scores.get("total", 0),
                "raw_evaluation": raw_output[:500] if len(raw_output) > 500 else raw_output,
                "source": "OurData"
            }
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate limit" in error_msg.lower():
                time.sleep(15 * (attempt + 1))
            else:
                if attempt == retry_count - 1:
                    return {"error": error_msg}
                time.sleep(5)
    return {"error": "Max retries"}

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_our_data_eval_fast.py [qwen|deepseek]")
        return
        
    model_arg = sys.argv[1].lower()
    if model_arg == "qwen":
        MODEL = "Qwen/Qwen2.5-72B-Instruct"
        OUTPUT_FILE = Path("our_data_100_evaluated_qwen.json")
    else:
        MODEL = "deepseek-ai/DeepSeek-V3"
        OUTPUT_FILE = Path("our_data_100_evaluated_deepseek.json")
        
    INPUT_FILE = Path("our_data_100_for_eval.json")
    
    print(f"模型: {MODEL}")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    existing_results = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing = json.load(f)
            for item in existing:
                existing_results[item.get("question", "")[:50]] = item

    results = []
    items_to_process = []
    
    for item in data:
        q_key = item.get("question", "")[:50]
        if q_key in existing_results:
            results.append(existing_results[q_key])
        else:
            items_to_process.append(item)

    print(f"Total to process: {len(items_to_process)}")

    if items_to_process:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_item = {executor.submit(evaluate_single, item, MODEL): item for item in items_to_process}
            completed = 0
            for future in as_completed(future_to_item):
                res = future.result()
                if "error" not in res:
                    results.append(res)
                completed += 1
                print(f"Progress: {completed}/{len(items_to_process)}", flush=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if results:
        avg_f = sum(r.get("factuality", 0) for r in results) / len(results)
        avg_g = sum(r.get("granularity", 0) for r in results) / len(results)
        avg_gr = sum(r.get("grounding", 0) for r in results) / len(results)
        avg_t = sum(r.get("total", 0) for r in results) / len(results)

        print(f"平均Factuality: {avg_f:.2f}")
        print(f"平均Granularity: {avg_g:.2f}")
        print(f"平均Grounding: {avg_gr:.2f}")
        print(f"平均总分: {avg_t:.2f}")

if __name__ == "__main__":
    main()
