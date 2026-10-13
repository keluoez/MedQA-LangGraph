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
from core.batch import run_one
from core.http import cached_get
from core.json_utils import parse_json_object
from core.schemas import StrictVerifierJudge, validate_verifier_output
from core.pubmed import (
    extract_pubmed_title_abstract_from_xml,
    fetch_pubmed_article_xml,
    fetch_pubmed_pmids,
    fetch_pubmed_titles,
)
from core.text_utils import extract_pmids

REQUESTS_HEADERS = {"User-Agent": "BioReasonPipeline/0.1 (mailto:local)"}

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
PAIRS_FILE = BASE_DIR / "physio_condition_pairs.json"
DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "langgraph_batch_output.json"
LEGACY_PIPELINE_FILE = BASE_DIR / "pipeline.py"

# 加载项目根目录 .env（密钥配置）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_URL = os.getenv("BIOREASON_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.getenv("BIOREASON_MODEL", "meta/llama-3.2-90b-vision-instruct")  # M4-B2：maverick 已 EOL，换默认骨干
API_KEY = os.getenv("BIOREASON_API_KEY") or os.getenv("NVIDIA_API_KEY") or ""

def resolve_api_key() -> str:
    return core_llm.resolve_api_key(API_KEY)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PhysiologyState(TypedDict, total=False):
    pair: Dict[str, Any]          # condition, protein, uniprot_id
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


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    return core_llm.get_client(resolve_api_key(), BASE_URL)


def llm_call(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 2400,
) -> str:
    """LLM 调用（M4-B2 P4：重试/退避实现收敛至 core.llm，此处绑定模块 MODEL 配置；
    保留本模块 max_tokens=2400 默认值）。"""
    return core_llm.llm_call(client, system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens, model=MODEL)


# ---------------------------------------------------------------------------
# UniProt
# ---------------------------------------------------------------------------

def fetch_uniprot_physiology(uniprot_id: str) -> Dict[str, Any]:
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

        catalytic_activity = ""
        for comment in data.get("comments", []):
            if comment.get("commentType") != "CATALYTIC ACTIVITY":
                continue
            reaction = comment.get("reaction", {})
            reaction_name = reaction.get("name", "")
            if reaction_name:
                catalytic_activity = reaction_name
                break

        keywords = []
        for kw in data.get("keywords", [])[:12]:
            v = kw.get("value")
            if v:
                keywords.append(v)

        cofactor_info = ""
        for comment in data.get("comments", []):
            if comment.get("commentType") != "COFACTOR":
                continue
            cofactors = comment.get("cofactors", [])
            names = []
            for c in cofactors:
                name = c.get("name", {})
                if isinstance(name, dict):
                    names.append(name.get("value", ""))
                elif isinstance(name, str):
                    names.append(name)
            if names:
                cofactor_info = "; ".join(n for n in names if n)
            break

        tissue_specificity = extract_comment("TISSUE SPECIFICITY")
        subunit = extract_comment("SUBUNIT")
        induction = extract_comment("INDUCTION")

        biophysicochemical = ""
        for comment in data.get("comments", []):
            if comment.get("commentType") != "BIOPHYSICOCHEMICAL PROPERTIES":
                continue
            ph_dep = comment.get("phDependence", {})
            if isinstance(ph_dep, dict):
                texts = ph_dep.get("texts", [])
                if texts:
                    biophysicochemical += "pH dependence: " + texts[0].get("value", "") + " "
            temp_dep = comment.get("temperatureDependence", {})
            if isinstance(temp_dep, dict):
                texts = temp_dep.get("texts", [])
                if texts:
                    biophysicochemical += "Temperature dependence: " + texts[0].get("value", "")
            biophysicochemical = biophysicochemical.strip()
            break

        return {
            "status": "ok",
            "uniprot_id": uniprot_id,
            "protein_name": protein_name,
            "gene": gene,
            "function": function_text,
            "pathway": pathway_text,
            "catalytic_activity": catalytic_activity,
            "cofactor": cofactor_info,
            "tissue_specificity": tissue_specificity,
            "subunit": subunit,
            "induction": induction,
            "biophysicochemical": biophysicochemical,
            "keywords": keywords,
        }
    except Exception as e:
        return {"status": f"error:{type(e).__name__}"}


# ---------------------------------------------------------------------------
# PubMed helpers
# ---------------------------------------------------------------------------

# M4-B2 P4：PubMed/NCBI 检索（fetch_pubmed_pmids / fetch_pubmed_titles / fetch_pubmed_article_xml /
# extract_pubmed_title_abstract_from_xml）已收敛至 core.pubmed，经 cached_get 获得 SQLite 缓存（B2）
# 与 NCBI 3 req/s 限流（B3）；顶部 import 保持模块命名空间兼容。
# 注：原 phys 版 fetch_pubmed_article_xml timeout=15，统一为 core 版 timeout=25（更宽容，仅影响超时边界）。


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

PHYSIOLOGY_SECTIONS = [
    "【Core Conclusion】",
    "【Protein Information】",
    "【Condition】",
    "【Physiological Importance】",
    "【Evidence】",
]


def normalize_sections(answer: str) -> str:
    # M4-B2 P4：段落重组算法收敛至 core.text_utils，绑定 PHYSIOLOGY_SECTIONS 标签
    return core_text.normalize_sections(answer, PHYSIOLOGY_SECTIONS)


def normalize_evidence_citations(answer: str) -> str:
    answer = re.sub(r"PubMed\s*:\s*(\d+)", r"PMID:\1", answer, flags=re.IGNORECASE)
    evidence_match = re.search(r"(【Evidence】[\s\S]*)$", answer)
    if not evidence_match:
        return answer
    evidence_block = evidence_match.group(1)
    pmids = re.findall(r"PMID[:\s]*(\d+)", evidence_block, flags=re.IGNORECASE)
    if not pmids:
        return answer
    unique_pmids: List[str] = []
    for p in pmids:
        if p not in unique_pmids:
            unique_pmids.append(p)
    normalized_evidence = re.sub(
        r"(PMID[:\s]*\d+[;,]?\s*)+", "", evidence_block, flags=re.IGNORECASE
    ).strip()
    pmid_suffix = "; ".join(f"PMID:{p}" for p in unique_pmids[:4])
    normalized_evidence = f"{normalized_evidence} {pmid_suffix}"
    return answer[: evidence_match.start()] + normalized_evidence


def extract_evidence_text_for_support(state: PhysiologyState) -> str:
    evidence = state.get("evidence", {})
    parts: List[str] = []
    u = evidence.get("uniprot", {})
    if isinstance(u, dict):
        for k in (
            "protein_name", "gene", "function", "pathway",
            "catalytic_activity", "cofactor", "tissue_specificity",
            "subunit", "induction", "biophysicochemical",
        ):
            v = u.get(k)
            if v:
                parts.append(str(v))
    for art in evidence.get("pubmed_articles", []) or []:
        if isinstance(art, dict):
            for k in ("title", "abstract"):
                v = art.get(k)
                if v:
                    parts.append(str(v))
    return "\n".join(parts)


def find_unsupported_specifics(state: PhysiologyState) -> List[str]:
    answer = state.get("answer", "")
    evidence_text = extract_evidence_text_for_support(state).lower()
    issues: List[str] = []
    residue_matches = re.findall(
        r"\b(?:Ser|Cys|Lys|Asp|Glu|Arg|His|Tyr|Thr|Gly|Ala|Val|Leu|Ile|Pro|Asn|Gln|Met|Phe|Trp)\s*-?\s*(\d{2,4})\b",
        answer,
    )
    if residue_matches:
        for m in set(residue_matches):
            if m.lower() not in evidence_text:
                issues.append("unsupported_residue_number_detail")
                break
    if re.search(r"\bIC50\b|\bKi\b|\bKd\b", answer):
        if not re.search(r"\bIC50\b|\bKi\b|\bKd\b", evidence_text, flags=re.IGNORECASE):
            issues.append("unsupported_binding_affinity_numbers")
    return sorted(set(issues))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PHYSIOLOGY_RESPONDER_PROMPT = """You are a biomedical domain writer creating high-quality training samples about physiological conditions and protein function.
You must answer based only on the provided evidence package.
Do not add facts not supported by provided IDs or snippets.

CRITICAL FORMAT REQUIREMENT — YOU MUST OUTPUT ALL FIVE SECTION HEADERS EXACTLY:
【Core Conclusion】
【Protein Information】
【Condition】
【Physiological Importance】
【Evidence】

Each header must appear on its own line, exactly as shown above with the 【 】 brackets.
Do NOT use numbered lists, bullet points, or alternative headings as substitutes for these headers.
Do NOT merge or skip any section.

REASONING CHAIN (apply internally when writing):
  [physiological condition's physical/chemical property]
    → [how it alters protein conformation, active site, or binding capacity]
      → [downstream physiological outcome at cell/tissue/system level]

Section-by-section rules:

【Core Conclusion】
- Write 1–2 sentences strictly following: condition → protein activity/conformation change → physiological result.
- Do NOT write only the result without the intermediate mechanism.
- Avoid background, definitions, or disease-related content.
- No unsupported numeric constants (residue positions, Kd, IC50) unless in evidence.

【Protein Information】
- Include the UniProt ID and standard protein name.
- Optionally include gene name, cofactors, or tissue expression from evidence.
- Keep concise; no functional descriptions here.

【Condition】
- Describe the physiological condition precisely (include numeric values if available).
- Note that this is a normal human physiological state.

【Physiological Importance】
- Write 2–3 focused paragraphs (NOT a numbered list).
- Paragraph 1: how the condition affects protein structure or activity (molecular level).
- Paragraph 2: cellular or tissue-level consequence.
- Paragraph 3 (optional, only if supported): broader systemic significance.

【Evidence】
- This section is MANDATORY. Do not omit it.
- Cite the UniProt ID.
- Cite 1–3 PMIDs strictly from the Allowed PMIDs list provided. Do NOT cite any PMID not in that list.
- Format: "UniProt: <ID>; PMID:<number>; PMID:<number>"
- Each PMID must correspond to a concrete statement in your answer.
- If no PubMed evidence is available, write: "UniProt: <ID>; no PubMed evidence available."

Other constraints:
1) English only.
2) No duplicate content between sections."""


PHYSIOLOGY_VERIFIER_PROMPT = """You are a biomedical quality auditor reviewing answers about physiological conditions and protein function.
Score the answer using only the given question, pair metadata and evidence package.
Return JSON only — no other text:
{
  "factuality_score": 0-4,
  "grounding_score": 0-3,
  "format_score": 0-2,
  "consistency_score": 0-1,
  "errors": ["..."],
  "critical_errors": ["..."],
  "revision_advice": ["..."]
}

Factuality rubric (0-4) — evaluates the physiological reasoning chain:
0 = physiological function is fundamentally wrong or directly contradicts UniProt evidence
1 = major mechanism mismatch; the condition→protein→outcome chain is absent or contradicted
2 = correct direction but chain is incomplete (missing one step)
3 = mostly correct reasoning chain with only minor imprecision
4 = full condition → protein activity/conformation change → downstream physiological outcome chain, supported by UniProt or PubMed evidence

Grounding rubric (0-3) — note: physiological mechanism topics have sparse PubMed coverage; score accordingly:
3 = cited PMIDs are all in the allowed list AND each supports a specific claim
2 = PMIDs valid and relevant, but not every claim has a paired citation
1 = no PMID cited or PMIDs unavailable, but UniProt alone grounds the core claims AND content is scientifically plausible
0 = cited PMIDs are NOT in the allowed list, OR content directly contradicts the UniProt record

Format rubric (0-2):
2 = all five sections present with correct 【】 tags
1 = exactly one section missing or slightly malformed
0 = two or more sections missing or tags seriously wrong

Consistency rubric (0-1):
1 = protein name, UniProt ID, and physiological condition are all consistent throughout
0 = any entity inconsistency (e.g., wrong protein name or contradictory UniProt ID)

Critical errors (only flag these if truly certain):
- PMIDs cited that are NOT in the allowed list
- Wrong protein identity (completely different protein than specified)
- Mechanism that directly contradicts UniProt functional annotation
- Fabricated numeric binding constants not present in evidence

IMPORTANT: Be fair and accurate. Most physiology answers will have factuality 3-4 and grounding 1-2 since PubMed evidence is sparse for mechanistic physiology questions. Only deduct when there is clear evidence of error."""


PHYSIOLOGY_REFINER_PROMPT = """You are a biomedical editor specializing in physiological condition answers.
Revise the draft answer by fixing the listed errors.
Keep all five required sections with the same tags:
  【Core Conclusion】 【Protein Information】 【Condition】 【Physiological Importance】 【Evidence】
Maintain the reasoning chain structure: condition → protein activity change → physiological outcome.
Stay strictly within the provided evidence package.
Prefer concise training-style scientific writing.

CRITICAL: Output ONLY the corrected answer in full.
Do NOT write 'Changes made:', 'Note:', 'The original response was...', or any meta-commentary.
English only."""


# ---------------------------------------------------------------------------
# Node: collect
# ---------------------------------------------------------------------------

def node_collect(state: PhysiologyState) -> PhysiologyState:
    pair = state["pair"]
    condition = pair["condition"]
    protein = pair["protein"]
    uniprot_id = pair["uniprot_id"]

    uniprot_info = fetch_uniprot_physiology(uniprot_id)
    gene = uniprot_info.get("gene", "") or protein

    condition_terms = _extract_condition_keywords(condition)
    physiology_keywords = (
        "physiology[Title/Abstract] OR function[Title/Abstract] OR "
        "regulation[Title/Abstract] OR homeostasis[Title/Abstract] OR "
        "mechanism[Title/Abstract] OR pathway[Title/Abstract]"
    )
    pubmed_query = (
        f"({protein}[Title/Abstract] OR {gene}[Title/Abstract]) AND "
        f"({condition_terms}[Title/Abstract] OR {physiology_keywords})"
    )

    candidate_pmids = fetch_pubmed_pmids(pubmed_query, retmax=30)
    if not candidate_pmids:
        candidate_pmids = fetch_pubmed_pmids(f"{protein} {condition} function", retmax=30)

    titles = fetch_pubmed_titles(candidate_pmids)

    def rank_pmid(pmid: str) -> float:
        title = (titles.get(pmid, "") or "").lower()
        if not title: return 0.0
        score = 0.0
        if protein.lower() in title: score += 1.0
        if gene.lower() in title: score += 1.0
        if "physiology" in title or "function" in title: score += 0.6
        if "regulation" in title or "homeostasis" in title: score += 0.4
        if "mechanism" in title or "pathway" in title: score += 0.3
        if "review" in title: score -= 0.2
        return score

    ranked_pmids = sorted(candidate_pmids, key=rank_pmid, reverse=True)
    pmids = ranked_pmids[:6]
    titles = fetch_pubmed_titles(pmids)

    articles = []
    for pmid in pmids[:3]:
        xml = fetch_pubmed_article_xml(pmid)
        meta = extract_pubmed_title_abstract_from_xml(xml) if xml else {}
        articles.append({
            "pmid": pmid,
            "title": meta.get("title", "") or titles.get(pmid, ""),
            "journal": meta.get("journal", ""),
            "year": meta.get("year", ""),
            "abstract": (meta.get("abstract", "") or "")[:1200],
        })

    state["question"] = f"What is the function of {protein} under {condition}?"
    state["evidence"] = {
        "uniprot": uniprot_info,
        "pubmed_pmids": pmids,
        "pubmed_titles": titles,
        "pubmed_articles": articles,
    }
    return state


def _extract_condition_keywords(condition: str) -> str:
    clean = re.sub(r"\(.*?\)", "", condition).strip()
    keyword_map = {
        "temperature": "temperature", "37": "temperature",
        "ph": "pH", "7.4": "pH", "7.35": "pH",
        "fasting": "fasting", "fed": "postprandial",
        "exercise": "exercise", "circadian": "circadian",
        "aging": "aging", "pregnancy": "pregnancy",
    }
    tokens = re.findall(r"[A-Za-z0-9.]+", clean.lower())
    selected = [keyword_map.get(tok, tok) for tok in tokens if len(tok) > 3 or keyword_map.get(tok)]
    return " ".join(dict.fromkeys(selected))


# ---------------------------------------------------------------------------
# Node: generate
# ---------------------------------------------------------------------------

def build_generation_user_prompt(state: PhysiologyState) -> str:
    pair = state["pair"]
    evidence = state["evidence"]
    uniprot = evidence["uniprot"]
    pmids = evidence.get("pubmed_pmids", [])
    articles = evidence.get("pubmed_articles", [])

    lines = [
        f"Question: {state['question']}",
        f"Protein: {pair['protein']}",
        f"Physiological Condition: {pair['condition']}",
        f"UniProt ID: {pair['uniprot_id']}",
        "",
        "Evidence Package (use ONLY this information):"
    ]
    for k in ["protein_name", "gene", "function", "catalytic_activity", "cofactor", "biophysicochemical", "tissue_specificity", "subunit", "induction", "pathway"]:
        if uniprot.get(k): lines.append(f"{k.replace('_', ' ').capitalize()}: {uniprot[k]}")

    lines.append("")
    whitelist_str = "; ".join(f"PMID:{p}" for p in pmids if p)
    lines.append(f"ALLOWED PMIDs: {whitelist_str if whitelist_str else 'none'}")

    for art in articles:
        lines.append(f"- PMID:{art.get('pmid')} | {art.get('title')}\n  abstract: {art.get('abstract')[:700]}")

    lines.append("\nREMINDER: Output ALL headers 【Core Conclusion】【Protein Information】【Condition】【Physiological Importance】【Evidence】. Cite only allowed PMIDs.")
    return "\n".join(lines)


def node_generate(state: PhysiologyState) -> PhysiologyState:
    client = get_client()
    user_prompt = build_generation_user_prompt(state)
    answer = llm_call(client, PHYSIOLOGY_RESPONDER_PROMPT, user_prompt, temperature=0.2)
    state["answer"] = normalize_evidence_citations(normalize_sections(answer))
    state["best_score"] = -1.0 if "best_score" not in state else state["best_score"]
    return state


# ---------------------------------------------------------------------------
# Node: verify
# ---------------------------------------------------------------------------

def node_verify(state: PhysiologyState) -> PhysiologyState:
    client = get_client()
    pair = state["pair"]
    answer = state["answer"]
    evidence = state["evidence"]
    allowed_pmids = set(evidence.get("pubmed_pmids", []))

    # Determinstic checks
    missing = [t for t in PHYSIOLOGY_SECTIONS if t not in answer]
    pmids_in_answer = extract_pmids(answer)
    unsupported = [p for p in pmids_in_answer if p not in allowed_pmids]
    
    det_errors = []
    if missing: det_errors.append(f"missing_sections:{','.join(missing)}")
    if unsupported: det_errors.append(f"unsupported_pmids:{','.join(unsupported)}")
    if not pmids_in_answer and allowed_pmids: det_errors.append("evidence_missing_pmid")

    # LLM judge
    user_prompt = f"Question: {state['question']}\nProtein: {pair['protein']}\nAnswer:\n{answer}"
    raw = llm_call(client, PHYSIOLOGY_VERIFIER_PROMPT, user_prompt, temperature=0.0)
    # M5-B 第一层：解析 + Pydantic 结构化校验（格式幻觉降级，错误精确到字段；缺失分默认 0 与本模块口径一致）
    llm_judge, judge_err = validate_verifier_output(parse_json_object(raw), "four_dim", score_default=0.0)
    if judge_err:
        # M5-B 第二层：instructor 结构化抢救（尽力而为，失败回退降级）
        rescued = core_llm.llm_call_json(client, PHYSIOLOGY_VERIFIER_PROMPT, user_prompt, StrictVerifierJudge, model=MODEL)
        if rescued is not None:
            llm_judge, judge_err = rescued, None
    if judge_err:
        llm_judge = {
            "factuality_score": 0,
            "grounding_score": 0,
            "format_score": 0,
            "consistency_score": 0,
            "errors": [judge_err],
            "critical_errors": [],
            "revision_advice": [],
        }

    # Sum all 4 sub-scores: factuality(0-4) + grounding(0-3) + format(0-2) + consistency(0-1) = max 10
    score = (
        float(llm_judge.get("factuality_score", 0))
        + float(llm_judge.get("grounding_score", 0))
        + float(llm_judge.get("format_score", 0))
        + float(llm_judge.get("consistency_score", 0))
    )
    critical = [e for e in llm_judge.get("critical_errors", []) if "unsupported" in str(e).lower()] + [f"unsupported_pmids:{p}" for p in unsupported]

    state["best_score"] = max(state.get("best_score", -1.0), score)
    # 只保存无批性错误且分数不低于当前最佳分的答案
    if not critical and score >= state.get("best_score", -1.0):
        state["best_answer"] = answer

    correction_count = state.get("correction_count", 0)
    if score >= 7.0 and not critical: route = "pass"
    elif correction_count >= state.get("max_corrections", 2): route = "stop"
    else: route = "refine" if not critical else "regenerate"

    state["verification"] = {"score": score, "errors": det_errors + llm_judge.get("errors", []), "critical_errors": critical, "revision_advice": llm_judge.get("revision_advice", [])}
    state["route"] = route
    return state


# ---------------------------------------------------------------------------
# Refine / Regenerate
# ---------------------------------------------------------------------------

def node_refine(state: PhysiologyState) -> PhysiologyState:
    v = state["verification"]
    prompt = f"Errors: {v['errors']}\nAdvice: {v['revision_advice']}\nCurrent answer:\n{state['answer']}"
    state["answer"] = normalize_evidence_citations(normalize_sections(llm_call(get_client(), PHYSIOLOGY_REFINER_PROMPT, prompt)))
    state["correction_count"] = state.get("correction_count", 0) + 1
    return state

def node_regenerate(state: PhysiologyState) -> PhysiologyState:
    v = state["verification"]
    prompt = build_generation_user_prompt(state) + f"\nAvoid these issues: {v['critical_errors']}"
    state["answer"] = normalize_evidence_citations(normalize_sections(llm_call(get_client(), PHYSIOLOGY_RESPONDER_PROMPT, prompt)))
    state["correction_count"] = state.get("correction_count", 0) + 1
    return state

def node_finalize(state: PhysiologyState) -> PhysiologyState:
    v = state.get("verification", {})
    pair = state["pair"]
    best_answer = state.get("best_answer") or state.get("answer", "")
    
    # 清除 Refiner 元评论残迹
    best_answer = re.sub(
        r'\n+(?:Note:|Changes made:|The original response|Corrections made:|The revised response).*$',
        '', best_answer, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    
    state["final_sample"] = {
        "question": state.get("question"),
        "answer": best_answer,
        "category": "Basic-Physiological Conditions",
        "condition": pair["condition"],
        "protein": pair["protein"],
        "uniprot_id": pair.get("uniprot_id", ""),
        "quality_score": round(state.get("best_score", v.get("score", 0)), 2),
        "quality_log": v
    }
    return state

# ---------------------------------------------------------------------------
# Graph & CLI
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(PhysiologyState)
    g.add_node("collect", node_collect); g.add_node("generate", node_generate); g.add_node("verify", node_verify)
    g.add_node("refine", node_refine); g.add_node("regenerate", node_regenerate); g.add_node("finalize", node_finalize)
    g.set_entry_point("collect")
    g.add_edge("collect", "generate"); g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", lambda x: x["route"], {"pass": "finalize", "refine": "refine", "regenerate": "regenerate", "stop": "finalize"})
    g.add_edge("refine", "verify"); g.add_edge("regenerate", "verify"); g.add_edge("finalize", END)
    return g.compile()

# M4-B2 P4：parse_json_object 已收敛至 core.json_utils（顶部 import，实现见 core/json_utils.py）

def run_batch(pairs: List[Dict[str, Any]], output_file: Path, max_corrections: int = 2, max_workers: int = 10) -> List[Dict[str, Any]]:
    app = build_graph()
    return core_batch.run_batch(
        pairs,
        output_file,
        process_one=lambda pair: run_one(app, pair, max_corrections=max_corrections),
        # 使用 condition::uniprot_id 作为断点key，允许同一蛋白质名的不同变体各自处理
        pair_id=lambda p: f"{p.get('condition')}::{p.get('uniprot_id', p.get('protein', ''))}",
        describe=lambda r: f"{r.get('condition')} -> {r.get('protein')}",
        error_fields=lambda p: {"condition": p["condition"], "protein": p["protein"]},
        max_workers=max_workers,
        batch_label="physiology-condition items",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--max-corrections", type=int, default=2)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FILE))
    args = parser.parse_args()
    
    pairs = json.loads(PAIRS_FILE.read_text(encoding="utf-8"))
    if args.count:
        pairs = pairs[:args.count]
    run_batch(pairs, Path(args.output), max_corrections=args.max_corrections, max_workers=args.workers)
