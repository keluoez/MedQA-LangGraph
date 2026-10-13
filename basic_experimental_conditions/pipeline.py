# -*- coding: utf-8 -*-
"""
BioReasonPipeline - 实验条件模块
Experimental Conditions: 实验条件对蛋白活性的影响
"""

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PAIRS_FILE = os.path.join(BASE_DIR, "experiment_pairs.json")

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

API_KEY = os.getenv("BIOREASON_API_KEY") or os.getenv("SILICONFLOW_API_KEY") or ""
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==================== 工具函数 ====================
def search_uniprot_detailed(uniprot_id: str) -> str:
    """从UniProt获取详细的蛋白功能信息，包括催化位点、结合位点等"""
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        r = requests.get(url, timeout=10)
        time.sleep(0.5)
        if r.status_code != 200:
            return "Not found"
        
        data = r.json()
        
        # 基本信息
        protein_name = data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
        
        # 获取Catalytic activity - 可能在texts或reaction字段
        catalytic_info = ""
        for comment in data.get('comments', []):
            if comment.get('commentType') == 'CATALYTIC ACTIVITY':
                # 尝试从reaction获取
                reaction = comment.get('reaction')
                if reaction and isinstance(reaction, dict):
                    name = reaction.get('name', '')
                    equation = reaction.get('equation', '')
                    if name:
                        catalytic_info = name
                    if equation:
                        catalytic_info += f" | Equation: {equation}"
                # 如果没有reaction，尝试从texts获取
                if not catalytic_info:
                    texts = comment.get('texts', [])
                    if texts and isinstance(texts, list):
                        text_obj = texts[0]
                        if isinstance(text_obj, dict):
                            catalytic_info = text_obj.get('value', '')
                break
        
        # 获取pH依赖性信息
        ph_info = ""
        for comment in data.get('comments', []):
            if comment.get('commentType') == 'BIOPHYSICOCHEMICAL PROPERTIES':
                texts = comment.get('texts', [])
                for text_obj in texts:
                    text_val = text_obj.get('value', '')
                    if 'pH' in text_val or 'optimal' in text_val.lower():
                        ph_info = text_val
                        break
        
        # 获取Cofactor
        cofactor_info = ""
        for comment in data.get('comments', []):
            if comment.get('commentType') == 'COFACTOR':
                cofactors = comment.get('cofactors')
                if cofactors and isinstance(cofactors, list):
                    cofactor_info = "; ".join([c.get('name', {}).get('value', '') if isinstance(c.get('name'), dict) else '' for c in cofactors])
                if not cofactor_info:
                    texts = comment.get('texts', [])
                    if texts and isinstance(texts, list):
                        text_obj = texts[0]
                        if isinstance(text_obj, dict):
                            cofactor_info = text_obj.get('value', '')
                break
        
        # 获取Function信息
        function_info = ""
        for comment in data.get('comments', []):
            if comment.get('commentType') == 'FUNCTION':
                texts = comment.get('texts', [])
                if texts and isinstance(texts, list):
                    text_obj = texts[0]
                    if isinstance(text_obj, dict):
                        function_info = text_obj.get('value', '')
                break
        
        # 获取Active site和Binding site信息（关键！）
        feature_info = []
        for feature in data.get('features', []):
            ftype = feature.get('type', '')
            if ftype in ['ACTIVE_SITE', 'BINDING_SITE', 'SITE', 'METAL_BINDING']:
                desc = feature.get('description', '') or ''
                pos_start = feature.get('location', {}).get('start', {}).get('value', '')
                pos_end = feature.get('location', {}).get('end', {}).get('value', '')
                if pos_start:
                    if pos_end and pos_end != pos_start:
                        feature_info.append(f"{ftype} {pos_start}-{pos_end}: {desc}")
                    else:
                        feature_info.append(f"{ftype} {pos_start}: {desc}")
        
        result = f"Protein Name: {protein_name}"
        if function_info:
            result += f"\n\n=== FUNCTION ===\n{function_info[:500]}"
        if catalytic_info:
            result += f"\n\n=== CATALYTIC ACTIVITY ===\n{catalytic_info}"
        if ph_info:
            result += f"\n\n=== pH OPTIMUM ===\n{ph_info}"
        if cofactor_info:
            result += f"\n\n=== COFACTOR ===\n{cofactor_info}"
        if feature_info:
            result += f"\n\n=== ACTIVE SITES / BINDING SITES ===\n" + "\n".join(feature_info)
        
        return result
    except Exception as e:
        return f"Error: {str(e)[:100]}"

def search_pubmed_enzyme_detailed(query: str) -> str:
    """从PubMed获取酶学机制详细信息 - 简化版"""
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        
        params = {"db": "pubmed", "term": query, "retmax": 3, "retmode": "json", "sort": "relevance"}
        r = requests.get(search_url, params=params, timeout=15)
        time.sleep(0.5)
        if r.status_code != 200:
            return "No results"
        
        pmids = r.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return "No results"
        
        # 获取摘要
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {"db": "pubmed", "id": ",".join(pmids[:2]), "retmode": "xml", "rettype": "abstract"}
        r2 = requests.get(fetch_url, params=fetch_params, timeout=20)
        time.sleep(0.5)
        
        if r2.status_code != 200:
            return f"PMIDs: {', '.join(pmids[:2])}"
        
        from xml.etree import ElementTree as ET
        try:
            root = ET.fromstring(r2.content)
            abstracts = []
            for article in root.findall(".//PubmedArticle"):
                abstract_parts = article.findall(".//AbstractText")
                abstract = " ".join([a.text or "" for a in abstract_parts])
                if abstract:
                    abstracts.append(abstract[:500])
            if abstracts:
                return "\n".join(abstracts)
        except:
            pass
        
        return f"PMIDs: {', '.join(pmids[:2])}"
    except Exception as e:
        return f"Search failed: {str(e)[:50]}"

def search_kegg_enzyme(enzyme_name: str) -> str:
    """从KEGG获取酶的详细信息"""
    try:
        # 先搜索酶的EC编号
        search_url = "https://rest.kegg.jp/list/enzyme"
        r = requests.get(search_url, timeout=10)
        time.sleep(0.3)
        
        if r.status_code != 200:
            return "No KEGG data"
        
        # 搜索匹配的酶
        lines = r.text.split('\n')
        for line in lines:
            if enzyme_name.lower() in line.lower():
                return f"KEGG Enzyme: {line}"
        
        return "No KEGG match"
    except:
        return "KEGG search failed"

def search_pubmed(query: str) -> str:
    """从PubMed获取相关文献"""
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmax": 2, "retmode": "json"}
        r = requests.get(search_url, params=params, timeout=10)
        time.sleep(0.3)
        if r.status_code != 200:
            return "No results"
        pmids = r.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return "No results"
        return f"PMIDs: {', '.join(pmids[:2])}"
    except:
        return "Search failed"

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
            max_tokens=2000,
            timeout=60
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"LLM Error: {e}")
        return ""

# ==================== 提示词 ====================
RESPONDER_PROMPT = """You are a biomedical expert. Generate a CONCISE answer about how experimental conditions affect protein activity.

Use EXACTLY this format with 【】 brackets:

【Core Conclusion】
[One paragraph summarizing the mechanism - keep it brief]

【Protein Information】
Target Protein: [Name] (UniProt: [ID])

【Condition】
Experimental Condition: [Description]

【Clinical Relevance】
[Brief sentence - 1-2 sentences max]

【Evidence】
[Source references]

CRITICAL: Use 【】 brackets for ALL sections. Keep answers CONCISE."""

VERIFIER_PROMPT = """You are a biomedical fact-checker. Verify the answer for scientific accuracy.

CHECKLIST (all must pass):
1. FORMAT: Does it have all required 【】section headers: 【Core Conclusion】【Protein Information】【Condition】【Experimental/Clinical Relevance】【Evidence】?
2. CONSISTENCY: Are experimental condition, protein name, UniProt ID consistent throughout?
3. EVIDENCE: Are the UniProt ID and PMID numbers valid? (PMID must be 8 digits, check for fake/invented PMIDs like 41863798 which is from 2024 and may not exist)
4. CATALYTIC TRIAD: For enzymes, check if the catalytic residues/triad are CORRECT:
   - Pepsin: Asp-Asp-Thr (Asp32, Asp215, Thr218), NOT His-Ser-Asp
   - Trypsin: His-Asp-Ser (His57, Asp102, Ser195)
   - Chymotrypsin: His-Asp-Ser (His57, Asp102, Ser195)
   - If wrong catalytic triad is mentioned, mark as Error
5. MECHANISM: Is the mechanism description scientifically plausible?
6. LOGIC: Is the reasoning coherent and logical?

Output JSON format:
{"verdict":"Correct" or "Error","errors":["list specific errors if any"]}"""

REFINER_PROMPT = """You are a biomedical text corrector. Fix the scientific errors in the answer.

IMPORTANT:
- Do NOT add invented mechanisms
- Keep descriptions accurate
- Maintain 【】format"""

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
    print(f"  [Collector] 收集: {pair['condition']} -> {pair['protein']}", flush=True)
    
    try:
        # 1. UniProt详细信息
        protein_info = search_uniprot_detailed(pair['uniprot_id'])
        print(f"  [Collector] UniProt: {len(protein_info)} chars", flush=True)
        
        # 2. 主查询 - 简化查询词
        main_query = f"{pair['protein']} {pair['condition']}"
        literature = search_pubmed_enzyme_detailed(main_query)
        print(f"  [Collector] PubMed: {len(literature)} chars", flush=True)
        
        # 3. 额外查询 - 下游通路
        downstream_query = f"{pair['protein']} enzyme"
        extra_lit = search_pubmed(downstream_query)
        print(f"  [Collector] Extra: {len(extra_lit)} chars", flush=True)
        
        state.research_data = f"""【Protein Information from UniProt】
{protein_info}

【Mechanism Literature from PubMed】
{literature}

【Additional Literature】
{extra_lit}"""
        
        print(f"  [Collector] 完成 ({len(state.research_data)} chars)", flush=True)
    except Exception as e:
        print(f"  [Collector] 错误: {e}", flush=True)
        import traceback
        traceback.print_exc()
        state.research_data = "Error"
    
    return state

def responder(state: PipelineState) -> PipelineState:
    pair = state.pair
    print(f"  [Responder] 生成回答...", flush=True)
    
    question = f"How does {pair['condition']} affect {pair['protein']} activity and function?"
    state.question = question
    
    # 提取PMID
    lit_lines = [line for line in state.research_data.split('\n') if 'PMID' in line]
    pmid_refs = []
    for line in lit_lines[:3]:
        import re
        pmids = re.findall(r'\b\d{7,8}\b', line)
        for p in pmids:
            pmid_refs.append(f"PMID:{p}")
    evidence_refs = "; ".join(set(pmid_refs)) if pmid_refs else "References from PubMed"
    
    user_prompt = f"""Condition: {pair['condition']}
Target Protein: {pair['protein']} (UniProt: {pair['uniprot_id']})

Research Data:
{state.research_data}

Generate the answer using the provided research data."""

    answer = call_llm(RESPONDER_PROMPT, user_prompt, temperature=0.5)
    
    if "【Core Conclusion】" not in answer:
        print(f"  [Responder] 格式重试...", flush=True)
        answer = call_llm(RESPONDER_PROMPT, user_prompt, temperature=0.7)
    
    state.answer = answer
    print(f"  [Responder] 完成 ({len(answer)} chars)", flush=True)
    return state

def verifier(state: PipelineState) -> PipelineState:
    pair = state.pair
    print(f"  [Verifier] 验证...")
    
    user_prompt = f"""Condition: {pair['condition']}, Target: {pair['protein']}

Answer to verify:
{state.answer[:800]}

Verify the answer."""

    result = call_llm(VERIFIER_PROMPT, user_prompt, temperature=0.2)
    
    try:
        json_match = re.search(r'\{[^}]+\}', result, re.DOTALL)
        verification = json.loads(json_match.group()) if json_match else {"verdict": "Correct", "errors": []}
    except:
        verification = {"verdict": "Correct", "errors": []}
    
    state.error_steps = verification.get('errors', [])
    print(f"  [Verifier] {verification.get('verdict')} ({len(state.error_steps)} errors)")
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
    
    # 清理验证残留
    answer = re.sub(r'\[(CHEMISTRY|RESIDUE|NOMENCLATURE|MECHANISM)\].*', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\n?Note:.*', '', answer)
    
    # 重新排序各部分，确保顺序正确
    sections = {}
    for tag in ['【Core Conclusion】', '【Protein Information】', '【Condition】', '【Clinical Relevance】', '【Evidence】']:
        pos = answer.find(tag)
        if pos != -1:
            sections[tag] = answer[pos:]
    
    if sections:
        ordered = []
        for tag in ['【Core Conclusion】', '【Protein Information】', '【Condition】', '【Clinical Relevance】', '【Evidence】']:
            if tag in sections:
                content = sections[tag]
                next_tags = ['【Core Conclusion】', '【Protein Information】', '【Condition】', '【Clinical Relevance】', '【Evidence】']
                for next_tag in next_tags:
                    if next_tag != tag:
                        next_pos = content.find(next_tag)
                        if next_pos != -1:
                            content = content[:next_pos]
                ordered.append(content.strip())
        answer = '\n\n'.join(ordered)
    
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    answer = answer.strip()
    
    state.final_sample = {
        "question": state.question,
        "answer": answer,
        "category": "Basic-Experimental Conditions",
        "condition": pair['condition'],
        "protein": pair['protein'],
        "uniprot_id": pair['uniprot_id']
    }
    
    print(f"  [Output] 完成")
    return state

# ==================== 主流程 ====================
def process_single(pair: Dict) -> Dict:
    start_time = time.time()
    print(f"\n处理: {pair['condition']} -> {pair['protein']}")
    
    state = PipelineState(pair)
    
    state = collector(state)
    state = responder(state)
    
    for i in range(2):
        state = verifier(state)
        if not state.error_steps:
            break
        if state.correction_count >= 2:
            break
        state = refiner(state)
    
    state = output_agent(state)
    
    print(f"总耗时: {time.time() - start_time:.1f}秒")
    return state.final_sample

def batch_process(pairs: List[Dict], output_file: str = None):
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "batch_output.json")
    
    print(f"\n批量处理: {len(pairs)} 条")
    results = []
    
    for i, pair in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] " + "="*50)
        try:
            sample = process_single(pair)
            if sample:
                results.append(sample)
                print(f"成功!")
        except Exception as e:
            import traceback
            print(f"失败: {e}")
            print(traceback.format_exc())
        time.sleep(1)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成! {len(results)} 条 -> {output_file}")
    return results

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("BioReasonPipeline - 实验条件模块")
    print("=" * 60)
    
    if os.path.exists(PAIRS_FILE):
        with open(PAIRS_FILE, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
    else:
        pairs = []
    
    print(f"\n加载 {len(pairs)} 个实验条件-蛋白对")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        try:
            count = int(sys.argv[2])
            pairs = pairs[:count]
            print(f"处理 {len(pairs)} 条数据")
        except:
            pairs = pairs[:1]
    
    batch_process(pairs)
