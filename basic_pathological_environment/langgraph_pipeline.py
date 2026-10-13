import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal

import requests
from openai import OpenAI
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

# 项目根加入 sys.path，保证 `python langgraph_pipeline.py` 直跑时可导入 core/ 共享库（M4-B2 P4）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core import batch as core_batch
from core import llm as core_llm
from core import text_utils as core_text
from core.http import cached_get
from core.json_utils import parse_json_object
from core.schemas import StrictPathologyVerifierJudge, validate_verifier_output
from core.pubmed import (
    extract_pubmed_title_abstract_from_xml,
    fetch_pubmed_article_xml,
    fetch_pubmed_pmids,
    fetch_pubmed_titles,
)

REQUESTS_HEADERS = {"User-Agent": "BioReasonPipeline/0.1 (mailto:local)"}

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
PAIRS_FILE = BASE_DIR / "disease_pairs.json"
DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "langgraph_batch_output.json"
LEGACY_PIPELINE_FILE = BASE_DIR / "pipeline.py"

# 加载项目根目录 .env（密钥配置）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# M4-B2：原骨干 llama-4-maverick 已于 2026-07-27 EOL，模型改为经 .env 可配置（与 exp/phys 模块对齐）
BASE_URL = os.getenv("BIOREASON_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.getenv("BIOREASON_MODEL", "meta/llama-3.2-90b-vision-instruct")
API_KEY = os.getenv("BIOREASON_API_KEY") or os.getenv("NVIDIA_API_KEY") or ""


def resolve_api_key() -> str:
    return core_llm.resolve_api_key(API_KEY, LEGACY_PIPELINE_FILE)


class PathologyState(TypedDict, total=False):
    pair: Dict[str, Any]
    question: str
    evidence: Dict[str, Any]
    answer: str
    best_answer: str
    best_score: float
    verification: Dict[str, Any]
    route: Literal["pass", "refine", "regenerate", "stop"]
    correction_count: int
    max_corrections: int
    final_sample: Dict[str, Any]
    elapsed_seconds: float


def get_client() -> OpenAI:
    return core_llm.get_client(resolve_api_key(), BASE_URL)


def llm_call(client: OpenAI, system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 1800) -> str:
    """LLM 调用（M4-B2 P4：重试/退避实现收敛至 core.llm，此处仅绑定模块 MODEL 配置）。"""
    return core_llm.llm_call(client, system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens, model=MODEL)


# M4-B2 P4：parse_json_object 已收敛至 core.json_utils（顶部 import，实现见 core/json_utils.py）


def fetch_uniprot_pathology(uniprot_id: str) -> Dict[str, Any]:
    """从UniProt获取病理环境相关的蛋白信息"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        r = cached_get(url, timeout=20, headers=REQUESTS_HEADERS)  # M4-B2 B2：经 SQLite 缓存
        if r.status_code != 200:
            return {"status": "not_found"}
        data = r.json()
        
        protein_name = (
            data.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value", "")
        )
        gene = ""
        genes = data.get("genes", [])
        if genes:
            gene = genes[0].get("geneName", {}).get("value", "")

        def extract_comment(comment_type: str) -> str:
            for comment in data.get("comments", []):
                if comment.get("commentType") != comment_type:
                    continue
                texts = comment.get("texts", [])
                if texts:
                    return texts[0].get("value", "")
            return ""

        function_text = extract_comment("FUNCTION")
        pathway_text = extract_comment("PATHWAY")
        pathology_text = extract_comment("PATHOLOGY")
        disease_text = extract_comment("DISEASE")
        
        # 获取疾病关联信息（增强版）
        disease_associations = []
        for comment in data.get("comments", []):
            if comment.get("commentType") in ["PATHOLOGY", "DISEASE"]:
                texts = comment.get("texts", [])
                if texts:
                    for text in texts:
                        disease_associations.append(text.get("value", ""))
        
        # 从特征部分获取更多疾病关联
        for feature in data.get("features", []):
            if feature.get("type") in ["VARIANT", "MUTAGENESIS"]:
                description = feature.get("description", "")
                if "disease" in description.lower() or "cancer" in description.lower():
                    disease_associations.append(description)

        keywords = []
        for kw in data.get("keywords", [])[:12]:
            v = kw.get("value")
            if v:
                keywords.append(v)

        return {
            "status": "ok",
            "uniprot_id": uniprot_id,
            "protein_name": protein_name,
            "gene": gene,
            "function": function_text,
            "pathway": pathway_text,
            "pathology": pathology_text,
            "disease": disease_text,
            "disease_associations": disease_associations[:5],  # 增加到5个关联
            "keywords": keywords,
        }
    except Exception as e:
        return {"status": f"error:{type(e).__name__}"}


# M4-B2 P4：validate_pmid（原定义未被任何调用点使用）已删除；
# PMID 有效性由 deterministic_checks 的白名单机制承担。

def fetch_pubmed_pathology_articles(disease: str, protein: str, gene: str = "", retmax: int = 6) -> List[Dict[str, Any]]:
    """从PubMed获取病理机制相关文献（基于蛋白-药物模块的智能策略）"""
    
    # 智能查询策略（基于蛋白-药物模块优化）
    pubmed_query = (
        f"({disease}[Title/Abstract]) AND "
        f"(({protein}[Title/Abstract]) OR ({gene}[Title/Abstract])) AND "
        f"(mechanism[Title/Abstract] OR \"mode of action\"[Title/Abstract] OR "
        f"pathophysiology[Title/Abstract] OR pathogenesis[Title/Abstract] OR "
        f"phosphorylation[Title/Abstract] OR mutation[Title/Abstract] OR "
        f"expression[Title/Abstract] OR activation[Title/Abstract] OR "
        f"inhibition[Title/Abstract])"
    )
    
    # 获取候选PMID
    candidate_pmids = fetch_pubmed_pmids(pubmed_query, retmax=30)
    if not candidate_pmids:
        # 备用查询策略
        fallback_query = f"{disease} {protein} mechanism"
        candidate_pmids = fetch_pubmed_pmids(fallback_query, retmax=30)
    
    # 获取标题用于智能排序
    titles = fetch_pubmed_titles(candidate_pmids)
    
    # PMID智能排序算法（基于蛋白-药物模块）
    token_pool = {disease.lower(), protein.lower(), str(gene).lower()}
    token_pool = {x for x in token_pool if x}
    
    def rank_pmid(pmid: str) -> float:
        title = (titles.get(pmid, "") or "").lower()
        if not title:
            return 0.0
        score = 0.0
        for t in token_pool:
            if t in title:
                score += 1.0
        if "mechanism" in title or "mode of action" in title:
            score += 0.6
        if "pathophysiology" in title or "pathogenesis" in title:
            score += 0.5
        if "phosphorylation" in title:
            score += 0.4
        if "mutation" in title:
            score += 0.4
        if "expression" in title:
            score += 0.3
        if "activation" in title or "inhibition" in title:
            score += 0.3
        if "review" in title:
            score -= 0.4
        if "resistance" in title:
            score -= 0.3
        return score
    
    # 排序并选择最相关的PMID（基于蛋白-药物模块策略）
    ranked_pmids = sorted(candidate_pmids, key=rank_pmid, reverse=True)
    
    # 白名单：6个PMID（用于验证）
    allowed_pmids = ranked_pmids[:6]
    
    # 详细内容：只获取前3篇的摘要（提高效率）
    articles = []
    for pmid in allowed_pmids[:3]:
        xml_text = fetch_pubmed_article_xml(pmid)
        meta = extract_pubmed_title_abstract_from_xml(xml_text) if xml_text else {
            "title": titles.get(pmid, ""), 
            "abstract": "", 
            "journal": "", 
            "year": ""
        }
        article = {
            "pmid": pmid,
            "title": meta.get("title", "") or titles.get(pmid, ""),
            "journal": meta.get("journal", ""),
            "year": meta.get("year", ""),
            "abstract": (meta.get("abstract", "") or "")[:800],  # 限制摘要长度
        }
        articles.append(article)
    
    print(f"      白名单PMID: {len(allowed_pmids)} 个，详细内容: {len(articles)} 篇")
    
    # 返回白名单和详细内容
    return {
        "allowed_pmids": allowed_pmids,
        "articles": articles
    }


# M4-B2 P4：PubMed/NCBI 检索（fetch_pubmed_pmids / fetch_pubmed_titles / fetch_pubmed_article_xml /
# extract_pubmed_title_abstract_from_xml）已收敛至 core.pubmed，经 cached_get 获得 SQLite 缓存（B2）
# 与 NCBI 3 req/s 限流（B3）；顶部 import 保持模块命名空间兼容。
# 同时删除未被调用的 validate_pmid / fetch_pubmed_abstract_snippet(patho 版 max_chars=400)；
# 原 patho 版 fetch_pubmed_pmids 的调试 print 与 timeout=10 不再保留（统一 core 版 timeout=15）。


def node_collect(state: PathologyState) -> PathologyState:
    """Collector智能体：收集病理环境相关证据"""
    pair = state["pair"]
    disease = pair["disease"]
    pathology = pair.get("pathology", pair.get("disease", ""))
    protein = pair["protein"]
    uniprot_id = pair["uniprot_id"]
    
    print(f"  [Collector] 收集: {disease} - {pathology} -> {protein}")
    
    try:
        # 1. 获取UniProt病理信息
        print(f"  [Collector] 获取UniProt信息...")
        uniprot_info = fetch_uniprot_pathology(uniprot_id)
        gene = uniprot_info.get("gene", "") or protein
        print(f"  [Collector] UniProt完成: {uniprot_info.get('status', 'unknown')}")
        
        # 2. 获取PubMed病理机制文献（基于蛋白-药物模块策略）
        print(f"  [Collector] 获取PubMed文献...")
        pubmed_result = fetch_pubmed_pathology_articles(disease, protein, gene, retmax=6)
        
        # 提取白名单PMID和详细文章
        allowed_pmids = pubmed_result.get("allowed_pmids", [])
        pubmed_articles = pubmed_result.get("articles", [])
        
        print(f"  [Collector] PubMed完成: 白名单{len(allowed_pmids)}个PMID，详细内容{len(pubmed_articles)}篇")
        
        # 3. 构建证据包（包含白名单）
        evidence = {
            "uniprot": uniprot_info,
            "pubmed_pmids": allowed_pmids,  # 白名单PMID
            "pubmed_articles": pubmed_articles,
            "disease": disease,
            "pathology": pathology,
            "protein": protein,
            "gene": gene,
        }
        
        state["evidence"] = evidence
        
        # 生成问题
        question = f"How does {pathology} in {disease} affect {protein} function?"
        state["question"] = question
        
        print(f"  [Collector] 完成: {len(pubmed_articles)} 篇文献")
        return state
        
    except Exception as e:
        print(f"  [Collector] 错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回空的证据包，让后续流程处理
        state["evidence"] = {
            "uniprot": {"status": "error"},
            "pubmed_articles": [],
            "disease": disease,
            "pathology": pathology,
            "protein": protein,
            "gene": protein,
        }
        state["question"] = f"How does {pathology} in {disease} affect {protein} function?"
        
        return state


PATHOLOGY_RESPONDER_PROMPT = """You are a biomedical domain writer creating high-quality training samples about pathological conditions.
You must answer based only on the provided evidence package.
Do not add facts not supported by provided evidence.

Output must keep this structure:
【Core Conclusion】...
【Protein Information】...
【Condition】...
【Clinical Significance】...
【Evidence】...

CRITICAL REQUIREMENTS:
1. ENGLISH ONLY - Remove any non-English characters or content
2. Include molecular mechanism details (phosphorylation sites, mutation positions, expression changes)
3. Ensure UniProt disease associations are included in Protein Information section
4. Use only valid PMIDs (7-8 digit numbers)
5. Avoid content duplication between sections

Core Conclusion requirements:
- Provide a concise summary of how the pathological condition affects the target protein's function
- Keep it brief (1-2 sentences maximum)
- Focus on the key molecular mechanism
- The detailed logical chain should be in the Evidence section, not here
- Avoid background, definitions, and broad disease descriptions in this section

Protein Information section must include:
- UniProt ID and protein name only (keep concise)
- Do NOT include detailed function descriptions or PMIDs in this section
- Disease associations should be brief if included

Evidence section requirements:
- Provide the detailed logical chain: [pathological trigger] → [biochemical function alteration] → [cellular consequence]
- Cite 1-2 relevant PMIDs only (quality over quantity)
- Each PMID must support a specific mechanism statement
- Include specific molecular details (phosphorylation sites, mutation positions)
- Keep evidence concise and focused on the core mechanism

Language and formatting:
- English only - remove any non-English content
- Use standard protein names and avoid ambiguous abbreviations
- Maintain consistent formatting across all sections
- Avoid content duplication between sections
"""


def build_pathology_user_prompt(state: PathologyState) -> str:
    """构建 Responder 用户提示（M5-A 从 node_respond 原样抽取，行为保持不变；供机制自省渲染真实拼接预览）。"""
    pair = state["pair"]
    evidence = state["evidence"]

    disease = pair['disease']
    pathology = pair.get('pathology', pair.get('disease', ''))
    protein = pair['protein']
    uniprot_id = pair['uniprot_id']

    pubmed_articles = evidence.get("pubmed_articles", [])

    # 构建简洁证据摘要（基于蛋白-药物模块优化）
    allowed_pmids = evidence.get("pubmed_pmids", [])

    evidence_text = f"""Disease: {disease}
Pathology: {pathology}
Target Protein: {protein} (UniProt: {uniprot_id})
Allowed PMIDs: {', '.join(allowed_pmids)}

PubMed Articles (focus on quality, use 1-2 most relevant):
"""

    # 只使用最相关的1-2篇文献（基于蛋白-药物模块策略）
    for i, article in enumerate(pubmed_articles[:2], 1):
        evidence_text += f"\n{i}. PMID: {article.get('pmid', '')}"
        evidence_text += f"\n   Title: {article.get('title', '')}"
        # 简化摘要，只保留关键信息
        if article.get("abstract"):
            abstract = article.get('abstract', '')
            if len(abstract) > 80:
                evidence_text += f"\n   Key finding: {abstract[:80]}..."
            else:
                evidence_text += f"\n   Key finding: {abstract}"

    # 添加白名单约束提示
    evidence_text += f"\n\nIMPORTANT: You can only cite PMIDs from the allowed list above."

    user_prompt = f"""Generate a high-quality answer about how {pathology} in {disease} affects {protein} function.

Evidence Package:
{evidence_text}

Answer based ONLY on the provided evidence.

Structure your answer as follows:
- 【Core Conclusion】: Brief summary (1-2 sentences)
- 【Protein Information】: UniProt ID and protein name only
- 【Condition】: Brief description of the pathological condition
- 【Clinical Significance】: Clinical relevance
- 【Evidence】: Detailed logical chain with 1-2 PMIDs

Keep the answer concise and focused."""
    return user_prompt


def node_respond(state: PathologyState) -> PathologyState:
    """Responder智能体：生成病理环境回答"""
    print(f"  [Responder] 生成回答...")

    user_prompt = build_pathology_user_prompt(state)

    client = get_client()
    answer = llm_call(client, PATHOLOGY_RESPONDER_PROMPT, user_prompt, temperature=0.4)

    state["answer"] = answer
    print(f"  [Responder] 完成 ({len(answer)} chars)")

    return state


PATHOLOGY_VERIFIER_PROMPT = """You are a strict biomedical quality auditor for pathological condition answers.
Score the answer using only the given question, pair metadata and evidence package.
Return JSON only:
{
  "pathology_mechanism_score": 0-4,
  "disease_relevance_score": 0-3,
  "evidence_support_score": 0-2,
  "format_score": 0-1,
  "errors": ["..."],
  "critical_errors": ["..."],
  "revision_advice": ["..."]
}

Scoring rubric (STRICT MODE - errors must impact scores):
- Pathology mechanism score (0-4):
  4 = Clear, specific mechanism with molecular details (e.g., phosphorylation, mutation sites)
  3 = Correct mechanism but lacks molecular specificity
  2 = Vague or incomplete mechanism
  1 = Incorrect or missing mechanism
  0 = No mechanism described

- Disease relevance score (0-3):
  3 = Strong, specific disease relevance with context
  2 = Moderate relevance but could be more specific
  1 = Weak or generic disease relevance
  0 = No disease relevance

- Evidence support score (0-2):
  2 = Strong evidence support with valid, specific PMIDs
  1 = Some evidence but with issues (e.g., invalid PMIDs)
  0 = No or insufficient evidence

- Format score (0-1):
  1 = Proper structure with all required sections, consistent format
  0 = Missing sections, inconsistent format, or content duplication

CRITICAL ERRORS (must reduce corresponding scores):
- Invalid PMIDs → evidence_support_score = 0
- Missing required sections → format_score = 0
- Mechanism contradicts evidence → pathology_mechanism_score ≤ 2
- Content duplication → format_score = 0
- Non-English content → format_score = 0
- Missing UniProt disease associations in Protein Information section → disease_relevance_score ≤ 1
- Missing molecular mechanism details (phosphorylation sites, mutation positions) → pathology_mechanism_score ≤ 2

SCORING MUST REFLECT ERRORS: If errors are identified, corresponding scores must be reduced.

SPECIFIC ERROR CHECKLIST:
1. Check for non-English content (Chinese characters, etc.)
2. Verify Protein Information includes UniProt disease associations
3. Verify Core Conclusion includes specific molecular mechanisms
4. Check for content duplication between sections
5. Validate all PMIDs are 7-8 digit numbers

Provide specific, actionable feedback for each error.
"""


def node_verify(state: PathologyState) -> PathologyState:
    """Verifier智能体：验证回答质量（基于蛋白-药物模块的分层评分系统）"""
    answer = state["answer"]
    pair = state["pair"]
    evidence = state["evidence"]
    
    print("  [Verifier] 验证回答...")
    
    # 执行确定性检查
    det_checks = deterministic_checks(state)
    
    # 构建验证提示（基于蛋白-药物模块）
    user_prompt = "\n".join([
        f"Question: {state['question']}",
        f"Disease: {pair['disease']}",
        f"Pathology: {pair.get('pathology', pair.get('disease', ''))}",
        f"Protein: {pair['protein']}",
        f"UniProt ID: {pair['uniprot_id']}",
        f"Allowed PMIDs: {', '.join(det_checks.get('allowed_pmids', []))}",
        f"UniProt function snippet: {evidence.get('uniprot', {}).get('function', '')}",
        "",
        "Answer:",
        answer,
    ])
    
    client = get_client()
    verification_result_text = llm_call(client, PATHOLOGY_VERIFIER_PROMPT, user_prompt, temperature=0.0, max_tokens=900)
    
    # 解析验证结果（P5：平衡块解析器；M5-B 第一层：Pydantic 结构化校验，错误精确到字段）
    llm_judge, judge_err = validate_verifier_output(parse_json_object(verification_result_text), "pathology")
    if judge_err:
        # M5-B 第二层：instructor 结构化抢救（尽力而为，失败回退降级）
        rescued = core_llm.llm_call_json(client, PATHOLOGY_VERIFIER_PROMPT, user_prompt, StrictPathologyVerifierJudge, model=MODEL, max_tokens=900)
        if rescued is not None:
            llm_judge, judge_err = rescued, None
    if judge_err:
        llm_judge = {
            "pathology_mechanism_score": 0,
            "disease_relevance_score": 0,
            "evidence_support_score": 0,
            "format_score": 0,
            "errors": [f"无法解析验证结果: {judge_err}"],
            "critical_errors": [],
            "revision_advice": []
        }
    
    # 分层评分系统（基于蛋白-药物模块）
    pathology_mechanism = float(llm_judge.get("pathology_mechanism_score", 1))
    disease_relevance = float(llm_judge.get("disease_relevance_score", 1))
    evidence_support = float(llm_judge.get("evidence_support_score", 1))
    format_score = float(llm_judge.get("format_score", 1))
    
    llm_errors = llm_judge.get("errors", []) if isinstance(llm_judge.get("errors", []), list) else []
    llm_critical_raw = llm_judge.get("critical_errors", []) if isinstance(llm_judge.get("critical_errors", []), list) else []
    advice = llm_judge.get("revision_advice", []) if isinstance(llm_judge.get("revision_advice", []), list) else []
    
    # 关键错误分类（基于蛋白-药物模块）
    llm_critical = []
    llm_noncritical_from_critical = []
    critical_markers = (
        "unsupported", "wrong target", "wrong disease", "contradiction",
        "inconsistent id", "fabricated", "halluc", "not in the allowed",
        "not included in the allowed", "evidence does not support"
    )
    
    for item in llm_critical_raw:
        lowered = str(item).lower()
        if lowered.startswith("unsupported_pmids:"):
            llm_noncritical_from_critical.append(item)
            continue
        if any(k in lowered for k in critical_markers):
            llm_critical.append(item)
        else:
            llm_noncritical_from_critical.append(item)
    
    # 合并所有错误
    all_errors = det_checks["det_errors"] + llm_errors + llm_noncritical_from_critical
    all_critical = det_checks["det_critical_errors"] + llm_critical
    
    # 基础事实性评分（分层系统）
    if all_critical:
        pathology_mechanism_base = 0.0
    elif any(x in det_checks["det_errors"] for x in (
        "disease_name_missing_or_inconsistent",
        "protein_name_missing_or_inconsistent",
        "uniprot_id_missing_or_inconsistent"
    )):
        pathology_mechanism_base = 1.0
    elif "evidence_section_missing_pmid" in det_checks["det_errors"]:
        pathology_mechanism_base = 2.0
    elif len(llm_errors) >= 3:
        pathology_mechanism_base = 2.0
    elif len(det_checks["det_errors"]) == 0 and len(llm_errors) == 0:
        pathology_mechanism_base = 4.0
    else:
        pathology_mechanism_base = 3.0
    
    # 最终分数计算
    if pathology_mechanism >= 3.5:
        pathology_mechanism_adjust = 1.0
    elif pathology_mechanism <= 1.5:
        pathology_mechanism_adjust = -1.0
    else:
        pathology_mechanism_adjust = 0.0
    
    # M4-B2 修复：min(4.0) 钳制（此前 base 4 + adjust 1 可达 5 分越界）
    pathology_mechanism_effective = max(0.0, min(4.0, pathology_mechanism_base + pathology_mechanism_adjust))

    # 计算总分（M4-B2 修复：LLM 四维之和，上限 10 分）
    # 确定性检查经 base 折减与错误清单参与评分，不再与旧体系 det format/consistency 分数混加（曾致 12 分越界）
    total_score = (
        pathology_mechanism_effective +
        disease_relevance +
        evidence_support +
        format_score
    )
    
    # 存储验证结果
    # M4-B2 修复：顶层四维统一为 LLM 口径（format_score 用 LLM 0-1 分，det 明细移入 deterministic 键），
    # finalize 按"pathology+disease+evidence+format"求和天然 ≤10（此前 det format ≤2 混入致 12 分）
    verification_result = {
        "pathology_mechanism_score": pathology_mechanism_effective,
        "disease_relevance_score": disease_relevance,
        "evidence_support_score": evidence_support,
        "format_score": format_score,
        "errors": all_errors,
        "critical_errors": all_critical,
        "revision_advice": advice,
        "subscores": {
            "pathology_mechanism_base": pathology_mechanism_base,
            "pathology_mechanism_llm_raw": pathology_mechanism,
            "pathology_mechanism_effective": pathology_mechanism_effective,
            "grounding_llm_raw": disease_relevance,
            "format_llm_raw": format_score,
            "consistency_llm_raw": evidence_support
        },
        "deterministic": det_checks
    }
    
    state["verification"] = verification_result
    
    # 基于蛋白-药物模块的路由决策
    correction_count = state.get("correction_count", 0)
    max_corrections = state.get("max_corrections", 2)
    
    # 调试日志
    print(f"  [DEBUG] 修正计数: {correction_count}/{max_corrections}")
    print(f"  [DEBUG] 关键错误: {all_critical}")
    print(f"  [DEBUG] 一般错误: {all_errors[:3] if all_errors else '无'}")
    
    # 路由逻辑（优先级从高到低）
    if total_score >= 9.0:
        route = "pass"
    elif total_score >= 8.0 and len(all_critical) == 0:
        route = "pass"
    elif correction_count >= max_corrections:
        route = "stop"
    elif len(all_critical) > 0 and total_score < 7.0:
        route = "regenerate"
    else:
        route = "refine"
    
    # 存储路由决策
    state["route"] = route
    
    print(f"  [Verifier] 得分: {total_score}/10, 路由: {route}")
    
    return state


def should_refine(state: PathologyState) -> Literal["refine", "pass", "regenerate", "stop"]:
    """决定路由（基于蛋白-药物模块精确复制）"""
    # 直接返回存储在状态中的路由决策
    route = state.get("route", "stop")
    print(f"  [路由决策] 从状态获取路由: {route}")
    return route


PATHOLOGY_REFINER_PROMPT = """You are a biomedical editor for pathological condition answers.
Revise the draft answer by fixing listed errors.
Keep all five required sections with the same tags.
Stay strictly within provided evidence package.
Prefer concise training-style scientific writing.
English only.
"""


def node_regenerate(state: PathologyState) -> PathologyState:
    """Regenerate智能体：重新生成回答（基于蛋白-药物模块）"""
    verification = state.get("verification", {})
    pair = state["pair"]
    evidence = state["evidence"]
    
    # 修正次数计数（在节点执行时增加）
    current_count = state.get("correction_count", 0)
    correction_count = current_count + 1
    state["correction_count"] = correction_count
    
    print(f"  [Regenerate] 重新生成 (第{correction_count}次)...")
    
    critical_errors = verification.get("critical_errors", [])
    
    # 构建用户提示（包含已知关键问题）
    disease = pair['disease']
    pathology = pair.get('pathology', pair.get('disease', ''))
    protein = pair['protein']
    uniprot_id = pair['uniprot_id']
    
    uniprot_info = evidence.get("uniprot", {})
    pubmed_articles = evidence.get("pubmed_articles", [])
    
    evidence_text = f"""Disease: {disease}
Pathology: {pathology}
Target Protein: {protein} (UniProt: {uniprot_id})

UniProt Information:
- Protein Name: {uniprot_info.get('protein_name', '')}
- Function: {uniprot_info.get('function', '')[:300]}

PubMed Articles (focus on quality over quantity):
"""
    
    for i, article in enumerate(pubmed_articles[:2], 1):
        evidence_text += f"\n{i}. PMID: {article.get('pmid', '')}"
        evidence_text += f"\n   Title: {article.get('title', '')}"
        if article.get("abstract"):
            abstract = article.get('abstract', '')
            if len(abstract) > 150:
                evidence_text += f"\n   Abstract: {abstract[:150]}..."
            else:
                evidence_text += f"\n   Abstract: {abstract}"
    
    user_prompt = f"""Generate a high-quality answer about how {pathology} in {disease} affects {protein} function.

Evidence Package:
{evidence_text}

Known critical issues to avoid:
""" + "\n".join(f"- {x}" for x in critical_errors)
    
    client = get_client()
    regenerated = llm_call(client, PATHOLOGY_RESPONDER_PROMPT, user_prompt, temperature=0.15)
    
    # 标准化格式
    regenerated = normalize_sections(regenerated)
    regenerated = normalize_evidence_citations(regenerated)
    
    state["answer"] = regenerated
    print(f"  [Regenerate] 完成 ({len(regenerated)} chars)")
    
    return state


def node_refine(state: PathologyState) -> PathologyState:
    """Refiner智能体：修正回答"""
    verification = state["verification"]
    answer = state["answer"]
    pair = state["pair"]
    
    # 修正次数计数（在节点执行时增加）
    current_count = state.get("correction_count", 0)
    correction_count = current_count + 1
    state["correction_count"] = correction_count
    
    print(f"  [Refiner] 修正 (第{correction_count}次)...")
    
    errors = verification.get("errors", [])
    critical_errors = verification.get("critical_errors", [])
    revision_advice = verification.get("revision_advice", [])
    
    user_prompt = f"""Disease: {pair['disease']}
Pathology: {pair.get('pathology', pair.get('disease', ''))}
Protein: {pair['protein']}

Errors to fix:
{chr(10).join(f'- {e}' for e in errors + critical_errors)}

Revision advice:
{chr(10).join(f'- {a}' for a in revision_advice)}

Original Answer:
{answer}

Fix the errors while maintaining the required format."""
    
    client = get_client()
    corrected = llm_call(client, PATHOLOGY_REFINER_PROMPT, user_prompt, temperature=0.2)
    
    # 如果修正后的回答包含核心结构，则使用
    if "【Core Conclusion】" in corrected:
        state["answer"] = corrected
    
    print(f"  [Refiner] 完成")
    return state


def node_finalize(state: PathologyState) -> PathologyState:
    """Finalize智能体：生成最终样本"""
    pair = state["pair"]
    question = state["question"]
    answer = state["answer"]
    verification = state.get("verification", {})
    correction_count = state.get("correction_count", 0)
    
    print(f"  [Finalize] 生成最终样本...")
    
    # 清理格式
    answer = normalize_sections(answer)
    answer = normalize_evidence_citations(answer)
    
    # 使用与验证节点一致的评分逻辑
    pathology_score = verification.get("pathology_mechanism_score", 0)
    disease_score = verification.get("disease_relevance_score", 0)
    evidence_score = verification.get("evidence_support_score", 0)
    format_score = verification.get("format_score", 0)
    
    # 计算总分（与验证节点一致）
    quality_score = pathology_score + disease_score + evidence_score + format_score
    
    # 构建最终样本
    final_sample = {
        "question": question,
        "answer": answer,
        "category": "Basic-Pathological Environment",
        "disease": pair["disease"],
        "pathology": pair.get("pathology", pair.get("disease", "")),
        "protein": pair["protein"],
        "uniprot_id": pair["uniprot_id"],
        "quality_score": round(quality_score, 1),
        "quality_log": verification,
        "elapsed_seconds": state.get("elapsed_seconds", 0),
    }
    
    state["final_sample"] = final_sample
    print(f"  [Finalize] 完成 (得分: {quality_score:.1f}/10)")
    
    return state


# M4-B2 P4：段落重组算法收敛至 core.text_utils，此处仅绑定本模块的段落标签
SECTION_TAGS = ["【Core Conclusion】", "【Protein Information】", "【Condition】", "【Clinical Significance】", "【Evidence】"]


def normalize_sections(answer: str) -> str:
    """标准化回答格式"""
    return core_text.normalize_sections(answer, SECTION_TAGS)


def filter_non_english_content(text: str) -> str:
    """过滤非英文内容，确保纯英文输出"""
    # 移除中文字符
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    # 移除其他非ASCII字符（保留英文标点）
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def deterministic_checks(state: PathologyState) -> Dict[str, Any]:
    """确定性检查机制（基于蛋白-药物模块）"""
    pair = state["pair"]
    answer = state.get("answer", "")
    evidence = state.get("evidence", {})
    
    # 检查必填标签
    required_tags = ["【Core Conclusion】", "【Protein Information】", "【Condition】", "【Clinical Significance】", "【Evidence】"]
    missing_tags = [tag for tag in required_tags if tag not in answer]
    
    # 检查关键信息一致性
    errors = []
    critical_errors = []
    
    if pair["disease"].lower() not in answer.lower():
        errors.append("disease_name_missing_or_inconsistent")
    if pair["uniprot_id"].lower() not in answer.lower():
        errors.append("uniprot_id_missing_or_inconsistent")
    if pair["protein"].lower() not in answer.lower():
        errors.append("protein_name_missing_or_inconsistent")
    
    # 使用白名单机制进行PMID验证（基于蛋白-药物模块）
    allowed_pmids = set(evidence.get("pubmed_pmids", []))
    
    # 提取答案中的PMID
    pmids_in_answer = []
    pmid_pattern = r"PMID\s*:\s*(\d+)"
    for match in re.finditer(pmid_pattern, answer, re.IGNORECASE):
        pmids_in_answer.append(match.group(1))
    
    # 检查PMID支持性
    unsupported_pmids = [p for p in pmids_in_answer if p not in allowed_pmids]
    if unsupported_pmids:
        critical_errors.append(f"unsupported_pmids:{','.join(sorted(set(unsupported_pmids)))}")
    
    # 检查PMID数量
    if allowed_pmids and len(pmids_in_answer) == 0:
        errors.append("evidence_section_missing_pmid")
    
    # 检查PMID重复
    if len(pmids_in_answer) != len(set(pmids_in_answer)):
        errors.append("duplicate_pmids_in_evidence")
    
    # 计算基础分数
    format_score = 2 if not missing_tags else max(0, 2 - len(missing_tags))
    
    grounding_score = 3
    if unsupported_pmids:
        grounding_score = 0
    elif len(pmids_in_answer) == 0:
        grounding_score = 1
    
    consistency_score = 1 if (
        "disease_name_missing_or_inconsistent" not in errors
        and "uniprot_id_missing_or_inconsistent" not in errors
        and "protein_name_missing_or_inconsistent" not in errors
    ) else 0
    
    return {
        "det_errors": errors,
        "det_critical_errors": critical_errors,
        "format_score": format_score,
        "grounding_score": grounding_score,
        "consistency_score": consistency_score,
        "pmids_in_answer": pmids_in_answer,
        "allowed_pmids": list(allowed_pmids)
    }


def normalize_evidence_citations(answer: str) -> str:
    """标准化证据引用格式"""
    answer = re.sub(r"PubMed\s*:\s*(\d+)", r"PMID:\1", answer, flags=re.IGNORECASE)
    evidence_match = re.search(r"(【Evidence】[\s\S]*)$", answer)
    if not evidence_match:
        return answer
    evidence_block = evidence_match.group(1)
    pmids = re.findall(r"PMID[\s\s]*(\d+)", evidence_block, flags=re.IGNORECASE)
    if not pmids:
        return answer
    unique_pmids: List[str] = []
    for p in pmids:
        if p not in unique_pmids:
            unique_pmids.append(p)
    normalized_evidence = re.sub(r"(PMID[\s\s]*\d+[;,]?\s*)+", "", evidence_block, flags=re.IGNORECASE).strip()
    pmid_suffix = "; ".join(f"PMID:{p}" for p in unique_pmids[:3])
    if normalized_evidence.endswith("】"):
        normalized_evidence = f"{normalized_evidence} {pmid_suffix}"
    else:
        normalized_evidence = f"{normalized_evidence} {pmid_suffix}"
    return answer[:evidence_match.start()] + normalized_evidence


def create_pathology_graph():
    """创建病理环境LangGraph（基于蛋白-药物模块精确复制）"""
    workflow = StateGraph(PathologyState)
    
    # 添加节点
    workflow.add_node("collect", node_collect)
    workflow.add_node("respond", node_respond)
    workflow.add_node("verify", node_verify)
    workflow.add_node("refine", node_refine)
    workflow.add_node("regenerate", node_regenerate)
    workflow.add_node("finalize", node_finalize)
    
    # 设置入口点
    workflow.set_entry_point("collect")
    
    # 添加边
    workflow.add_edge("collect", "respond")
    workflow.add_edge("respond", "verify")
    workflow.add_conditional_edges(
        "verify",
        should_refine,
        {
            "pass": "finalize",
            "refine": "refine",
            "regenerate": "regenerate",
            "stop": "finalize",
        },
    )
    workflow.add_edge("refine", "verify")  # 修正后重新验证
    workflow.add_edge("regenerate", "verify")  # 重新生成后重新验证
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


def process_single_pathology(pair: Dict[str, Any], max_corrections: int = 2) -> Dict[str, Any]:
    """处理单个病理环境样本"""
    start_time = time.time()
    
    print(f"\n处理: {pair['disease']} - {pair.get('pathology', pair.get('disease', ''))} -> {pair['protein']}")
    
    # 初始化状态
    initial_state: PathologyState = {
        "pair": pair,
        "max_corrections": max_corrections,
        "correction_count": 0,
    }
    
    # 执行图
    graph = create_pathology_graph()
    final_state = graph.invoke(initial_state)
    
    # 计算耗时
    elapsed = time.time() - start_time
    final_state["elapsed_seconds"] = elapsed
    
    print(f"总耗时: {elapsed:.1f}秒")
    
    return final_state.get("final_sample", {})


def batch_process_pathology(pairs: List[Dict[str, Any]], output_file: Path, max_corrections: int = 2, max_workers: int = 4) -> List[Dict[str, Any]]:
    """批量处理病理环境样本（M4-B2 P4：断点续传/并发/失败审计骨架收敛至 core.batch）。

    与其他模块不同，patho 的单条执行 process_single_pathology 内部自建图并计时，
    故无需在此处 build_graph；日志文案统一为 core 版（信息量等价）。
    """
    return core_batch.run_batch(
        pairs,
        output_file,
        process_one=lambda pair: process_single_pathology(pair, max_corrections=max_corrections),
        pair_id=lambda p: f"{p.get('disease')}::{p.get('protein')}",
        describe=lambda r: f"{r.get('disease')} -> {r.get('protein')}",
        error_fields=lambda p: {"disease": p["disease"], "protein": p["protein"]},
        max_workers=max_workers,
        batch_label="pathology items",
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="病理环境模块LangGraph实现")
    parser.add_argument("--count", type=int, default=None, help="处理指定数量的样本")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FILE), help="输出文件路径")
    parser.add_argument("--max-corrections", type=int, default=2, help="最大修正次数")
    parser.add_argument("--workers", type=int, default=4, help="最大线程数")
    
    args = parser.parse_args()
    
    if not PAIRS_FILE.exists():
        print(f"错误: 输入文件不存在 {PAIRS_FILE}")
        return
    
    with open(PAIRS_FILE, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    if args.count:
        pairs = pairs[:args.count]
    
    output_file = Path(args.output)
    
    print("=" * 60)
    print("BioReasonPipeline - 病理环境模块 (LangGraph实现)")
    print("=" * 60)
    
    batch_process_pathology(pairs, output_file, args.max_corrections, args.workers)


if __name__ == "__main__":
    main()