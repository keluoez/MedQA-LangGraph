# -*- coding: utf-8 -*-
"""
BioReasonPipeline - 联合治疗模块
Combination Therapy: 联合治疗的协同机制
"""

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PAIRS_FILE = os.path.join(BASE_DIR, "combination_therapy_pairs.json")

import requests
import json
import time
import re
from typing import TypedDict, List, Dict
from openai import OpenAI

# ==================== 配置 ====================
# 加载项目根目录 .env（密钥配置）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("BIOREASON_API_KEY") or os.getenv("NVIDIA_API_KEY") or ""
BASE_URL = os.getenv("BIOREASON_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.getenv("BIOREASON_MODEL", "meta/llama-4-maverick-17b-128e-instruct")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==================== 工具函数 ====================
def search_chembl(chembl_id: str) -> str:
    """从ChEMBL获取药物信息"""
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
        r = requests.get(url, timeout=10)
        time.sleep(0.5)
        if r.status_code != 200:
            return "Not found"
        data = r.json()
        drug_name = data.get('pref_name', '')
        drug_type = data.get('molecule_type', '')
        return f"{drug_name} ({drug_type})"
    except:
        return "ChEMBL search failed"

# ==================== LLM调用 ====================
def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.5) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=3000,
            timeout=90
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"LLM Error: {e}")
        return ""

# ==================== 提示词 ====================
RESPONDER_PROMPT = """You are a biomedical expert. Generate an answer about synergistic mechanism of combination therapy following this EXACT format:

【Core Conclusion】[Drug A] + [Drug B] exhibit synergistic effects in [Therapeutic scene] through complementary mechanisms:

[Detailed mechanism including:
1. How drug A and drug B work individually on their targets
2. Synergistic or additive effects when combined
3. DOWNSTREAM EFFECTS - combined impact on disease pathway
4. Clinical benefits of the combination]

【Target Proteins】
1. Primary Target A: [Target] (UniProt: [ID])
   Mechanism: [Brief mechanism]
   Effect: [Therapeutic effect]

2. Primary Target B: [Target] (UniProt: [ID])
   Mechanism: [Brief mechanism]
   Effect: [Therapeutic effect]

【Clinical Significance】[Clinical relevance and outcomes]
【Evidence】UniProt: [IDs]; [Journal Year; PMID]

CRITICAL:
- Include DOWNSTREAM EFFECTS of the combination
- Explain the synergy mechanism
- Use real references
- Write in ENGLISH
- If research data is limited, state the limitations
- Write COMPREHENSIVE paragraphs, NOT brief bullet points
- ENGLISH only"""

VERIFIER_PROMPT = """You are a biomedical fact-checker. Score and verify the answer for scientific accuracy.

CHECKLIST:
1. FORMAT: Does it have all required 【】section headers: 【Core Conclusion】【Target Proteins】【Clinical Significance】【Evidence】?
2. CONSISTENCY: Are drug names, target proteins, UniProt IDs consistent throughout?
3. EVIDENCE: Are the UniProt ID and PMID numbers cited (not fabricated)?
4. MECHANISM: Is the synergistic mechanism scientifically plausible?
5. LOGIC: Is the reasoning coherent?

Return ONLY valid JSON:
{
  "verdict": "Correct" or "Error",
  "factuality_score": 0-4,
  "grounding_score": 0-3,
  "format_score": 0-2,
  "consistency_score": 0-1,
  "errors": ["list specific errors if any"]
}

Scoring:
- factuality_score 4 = mechanism is correct and specific; 3 = mostly correct; 2 = partially correct; 1 = major error; 0 = wrong
- grounding_score 3 = valid PMIDs cited; 2 = PMIDs present but not all verifiable; 1 = no PMIDs but content plausible; 0 = contradicts evidence
- format_score 2 = all 4 sections present; 1 = one missing; 0 = two or more missing
- consistency_score 1 = all entities consistent; 0 = any inconsistency"""

REFINER_PROMPT = """You are a biomedical text editor. Fix ONLY the listed scientific errors in the answer.

CRITICAL RULES:
- Output ONLY the corrected answer in full. Do NOT add any editorial notes.
- Do NOT write 'Changes made:', 'Note:', 'The original response', or any meta-commentary.
- Do NOT add new invented mechanisms.
- Keep all four 【】section headers intact.
- Maintain scientific accuracy.
- English only."""

# ==================== 智能体 ====================
class PipelineState:
    def __init__(self, pair: Dict):
        self.pair = pair
        self.research_data = ""
        self.question = ""
        self.answer = ""
        self.error_steps = []
        self.correction_count = 0
        self.final_sample = {}
        self.status = "init"

def collector(state: PipelineState) -> PipelineState:
    pair = state.pair
    
    # Adapt keys to match the new dataset format
    drugs = pair.get('drug_combination', pair.get('drugs', 'Unknown Drugs'))
    scene = pair.get('therapeutic_area', pair.get('scene', 'Unknown Context'))
    chembl_ids = pair.get('chembl_ids', [])
    targets = [pair.get('target_protein')] if pair.get('target_protein') else pair.get('targets', [])
    uniprot_ids = [pair.get('uniprot_id')] if pair.get('uniprot_id') else pair.get('uniprot_ids', [])
    
    # Store adapted keys back for further steps
    state.pair['drugs'] = drugs
    state.pair['scene'] = scene
    state.pair['targets'] = targets
    state.pair['uniprot_ids'] = uniprot_ids

    print(f"  [Collector] 收集: {drugs} -> {scene}")
    
    drug_infos = []
    for cid in chembl_ids:
        drug_infos.append(search_chembl(cid))
    drug_info = f"Drug combination: {' + '.join(drug_infos)}" if drug_infos else f"Drug combination: {drugs}"
    
    targets_info = f"Target proteins: {', '.join(targets)}"
    
    state.research_data = f"""【Drug Information from ChEMBL】
{drug_info}

【Target Proteins】
{targets_info}

【Therapeutic Context】
Therapeutic scene: {pair['scene']}"""
    
    print(f"  [Collector] 完成")
    return state

def responder(state: PipelineState) -> PipelineState:
    pair = state.pair
    print(f"  [Responder] 生成回答...")
    
    question = f"What is the synergistic mechanism of {pair['drugs']} in {pair['scene']}?"
    state.question = question
    
    user_prompt = f"""Drug Combination: {pair['drugs']}
Therapeutic Scene: {pair['scene']}
Target Proteins: {', '.join(pair['targets'])} (UniProt: {', '.join(pair['uniprot_ids'])})

Research Data:
{state.research_data}

Generate the answer using the provided research data."""

    answer = call_llm(RESPONDER_PROMPT, user_prompt, temperature=0.5)
    
    if "【Core Conclusion】" not in answer:
        print(f"  [Responder] 格式重试...")
        answer = call_llm(RESPONDER_PROMPT, user_prompt, temperature=0.7)
    
    state.answer = answer
    print(f"  [Responder] 完成 ({len(answer)} chars)")
    return state

def verifier(state: PipelineState) -> PipelineState:
    pair = state.pair
    print(f"  [Verifier] 验证...")
    
    user_prompt = f"""Drugs: {pair['drugs']}, Scene: {pair['scene']}
Target Protein: {', '.join(pair.get('targets', []))} (UniProt: {pair.get('uniprot_ids', [''])[0]})

Answer to verify (full):
{state.answer}

Verify and score the answer."""

    result = call_llm(VERIFIER_PROMPT, user_prompt, temperature=0.1)
    
    try:
        json_match = re.search(r'\{[\s\S]+\}', result)
        verification = json.loads(json_match.group()) if json_match else {}
    except:
        verification = {}
    
    state.error_steps = verification.get('errors', [])
    # 计算综合质量分 (满分10)
    state.quality_score = (
        float(verification.get('factuality_score', 0))
        + float(verification.get('grounding_score', 0))
        + float(verification.get('format_score', 0))
        + float(verification.get('consistency_score', 0))
    )
    state.verification = verification
    print(f"  [Verifier] {verification.get('verdict','?')} | score={state.quality_score:.1f} | errors={len(state.error_steps)}")
    return state

def refiner(state: PipelineState) -> PipelineState:
    print(f"  [Refiner] 修正 (第{state.correction_count+1}次)...")
    
    user_prompt = f"""Fix these errors:
{chr(10).join(f'- {e}' for e in state.error_steps)}

Original Answer:
{state.answer}

Fix the errors."""

    corrected = call_llm(REFINER_PROMPT, user_prompt, temperature=0.3)
    
    if "【Core Conclusion】" in corrected:
        state.answer = corrected
    
    state.correction_count += 1
    print(f"  [Refiner] 完成")
    return state

def output_agent(state: PipelineState) -> PipelineState:
    pair = state.pair
    answer = state.answer
    
    # 清理验证残留 & Refiner 元评论残迹
    answer = re.sub(r'\[(CHEMISTRY|RESIDUE|NOMENCLATURE|MECHANISM)\].*', '', answer, flags=re.IGNORECASE | re.DOTALL)
    # 删除 Refiner 自我评审遗留内容（从这些关键词到末尾全部删除）
    answer = re.sub(
        r'\n+(?:Note:|Changes made:|The original response|Corrections made:|The revised response).*$',
        '', answer, flags=re.DOTALL | re.IGNORECASE
    )
    
    # 重新排序各部分，确保顺序正确
    sections = {}
    for tag in ['【Core Conclusion】', '【Target Proteins】', '【Clinical Significance】', '【Evidence】']:
        pos = answer.find(tag)
        if pos != -1:
            sections[tag] = answer[pos:]
    
    if sections:
        ordered = []
        for tag in ['【Core Conclusion】', '【Target Proteins】', '【Clinical Significance】', '【Evidence】']:
            if tag in sections:
                content = sections[tag]
                next_tags = ['【Core Conclusion】', '【Target Proteins】', '【Clinical Significance】', '【Evidence】']
                for next_tag in next_tags:
                    if next_tag != tag:
                        next_pos = content.find(next_tag)
                        if next_pos != -1:
                            content = content[:next_pos]
                ordered.append(content.strip())
        answer = '\n\n'.join(ordered)
    
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    answer = answer.strip()
    
    # UniProt ID: 只取第一个有效 ID，不拼接
    raw_uids = pair.get('uniprot_ids', [])
    primary_uid = raw_uids[0] if raw_uids else pair.get('uniprot_id', '')
    
    state.final_sample = {
        "question": state.question,
        "answer": answer,
        "category": "Complex-Combination Therapy",
        "drugs": pair['drugs'],
        "scene": pair['scene'],
        "targets": pair['targets'],
        "uniprot_id": primary_uid
    }
    
    print(f"  [Output] 完成")
    return state

# ==================== 主流程 ====================
def process_single(pair: Dict) -> Dict:
    start_time = time.time()
    # Adapt keys upfront
    drugs = pair.get('drug_combination', pair.get('drugs', 'Unknown'))
    scene = pair.get('therapeutic_area', pair.get('scene', 'Unknown'))
    pair['drugs'] = drugs
    pair['scene'] = scene
    
    print(f"\n处理: {drugs} -> {scene}")
    
    state = PipelineState(pair)
    state.quality_score = 0.0
    state.verification = {}
    
    state = collector(state)
    state = responder(state)
    
    max_corrections = 2
    for i in range(max_corrections + 1):
        state = verifier(state)
        if not state.error_steps or state.quality_score >= 8.0:
            break
        if state.correction_count >= max_corrections:
            break
        state = refiner(state)
    
    state = output_agent(state)
    elapsed = round(time.time() - start_time, 2)
    state.final_sample['elapsed_seconds'] = elapsed
    state.final_sample['quality_score'] = round(state.quality_score, 2)
    state.final_sample['quality_log'] = {
        'correction_count': state.correction_count,
        'errors': state.error_steps,
        'subscores': {
            'factuality': state.verification.get('factuality_score', 0),
            'grounding': state.verification.get('grounding_score', 0),
            'format': state.verification.get('format_score', 0),
            'consistency': state.verification.get('consistency_score', 0),
        }
    }
    print(f"总耗时: {elapsed:.1f}秒 | 质量分: {state.quality_score:.1f}")
    return state.final_sample

def batch_process(pairs: List[Dict], output_file: str = None, max_workers: int = 10):
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "batch_output.jsonl")
    
    print(f"\n批量处理联合治疗样本: {len(pairs)} 条", flush=True)
    out_path = output_file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    completed_pairs = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    s = json.loads(line)
                    completed_pairs.add(f"{s.get('drugs')}::{s.get('uniprot_id')}")
                except Exception:
                    pass

    filtered_pairs = []
    for p in pairs:
        c_drugs = p.get('drug_combination', p.get('drugs', 'Unknown'))
        c_uid = p.get('uniprot_id', '') or (p.get('uniprot_ids', [''])[0] if p.get('uniprot_ids') else '')
        pid = f"{c_drugs}::{c_uid}"
        if pid not in completed_pairs:
            filtered_pairs.append(p)
            
    if not filtered_pairs:
        print("所有样本均已处理完毕！", flush=True)
        return []

    results = []
    flock = threading.Lock()
    
    def process_pair(pair):
        try:
            sample = process_single(pair)
            if sample:
                with flock:
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            return sample
        except Exception as e:
            return {"error": str(e), "drugs": pair.get('drug_combination')}

    print(f"启动 {max_workers} 个线程处理 {len(filtered_pairs)} 条数据...", flush=True)
    completed_count = 0
    total_count = len(filtered_pairs)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        f2p = {executor.submit(process_pair, p): p for p in filtered_pairs}
        for future in as_completed(f2p):
            completed_count += 1
            res = future.result()
            drugs_name = res.get('drugs') or res.get('drug_combination', 'Unknown')
            if "error" in res:
                print(f"[{completed_count}/{total_count}] FAILED: {drugs_name} | Error: {res['error']}", flush=True)
            else:
                print(f"[{completed_count}/{total_count}] SUCCESS {drugs_name}", flush=True)
                results.append(res)
                
    return results

# ==================== 主程序 ====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("BioReasonPipeline - 联合治疗模块")
    print("=" * 60)
    
    if os.path.exists(PAIRS_FILE):
        with open(PAIRS_FILE, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
    else:
        pairs = []
    
    if args.count:
        pairs = pairs[:args.count]
        
    print(f"\n加载 {len(pairs)} 个联合治疗对")
    
    batch_process(pairs, max_workers=args.workers)
