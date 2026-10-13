import argparse
import json
import os
import re
import statistics
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    fetch_pmc_fulltext_excerpt,
    fetch_pmcid_for_pmid,
    fetch_pubmed_abstract_snippet,
    fetch_pubmed_article_xml,
    fetch_pubmed_pmids,
    fetch_pubmed_titles,
)
from core.text_utils import extract_pmids

REQUESTS_HEADERS = {"User-Agent": "BioReasonPipeline/0.1 (mailto:local)"}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
PAIRS_FILE = os.path.join(BASE_DIR, "drug_target_pairs.json")
DEFAULT_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "langgraph_batch_output.json")

# 加载项目根目录 .env（密钥配置）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("BIOREASON_API_KEY") or os.getenv("NVIDIA_API_KEY") or ""
# M4-B2：原骨干 llama-4-maverick 已于 2026-07-27 EOL，模型改为经 .env 可配置（与 exp/phys 模块对齐）
BASE_URL = os.getenv("BIOREASON_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.getenv("BIOREASON_MODEL", "meta/llama-3.2-90b-vision-instruct")


def resolve_api_key() -> str:
    return core_llm.resolve_api_key(API_KEY)


class PipelineState(TypedDict, total=False):
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


def fetch_uniprot(uniprot_id: str) -> Dict[str, str]:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        r = cached_get(url, timeout=15)  # M4-B2 B2：经 SQLite 缓存
        if r.status_code != 200:
            return {"status": "not_found", "protein_name": "", "function": ""}
        data = r.json()
        protein_name = (
            data.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value", "")
        )
        function_text = ""
        for comment in data.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    function_text = texts[0].get("value", "")
                    break
        return {"status": "ok", "protein_name": protein_name, "function": function_text}
    except Exception as e:
        return {"status": f"error:{type(e).__name__}", "protein_name": "", "function": ""}


def fetch_uniprot_core(uniprot_id: str) -> Dict[str, Any]:
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

        return {
            "status": "ok",
            "uniprot_id": uniprot_id,
            "protein_name": protein_name,
            "gene": gene,
            "function": function_text,
            "pathway": pathway_text,
            "catalytic_activity": catalytic_activity,
            "keywords": keywords,
        }
    except Exception as e:
        return {"status": f"error:{type(e).__name__}"}


def fetch_chembl(chembl_id: str) -> Dict[str, str]:
    if not chembl_id:
        return {"status": "empty", "pref_name": "", "molecule_type": ""}
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
    try:
        r = cached_get(url, timeout=15)  # M4-B2 B2：经 SQLite 缓存
        if r.status_code != 200:
            return {"status": "not_found", "pref_name": "", "molecule_type": ""}
        data = r.json()
        return {
            "status": "ok",
            "pref_name": data.get("pref_name", ""),
            "molecule_type": data.get("molecule_type", ""),
        }
    except Exception as e:
        return {"status": f"error:{type(e).__name__}", "pref_name": "", "molecule_type": ""}


def fetch_chembl_drug_mechanisms(molecule_chembl_id: str, target_hint: str = "", max_items: int = 6) -> List[Dict[str, Any]]:
    if not molecule_chembl_id:
        return []
    url = "https://www.ebi.ac.uk/chembl/api/data/drug_mechanism.json"
    try:
        r = cached_get(url, params={"molecule_chembl_id": molecule_chembl_id, "limit": 100}, timeout=20, headers=REQUESTS_HEADERS)  # M4-B2 B2
        if r.status_code != 200:
            return []
        data = r.json()
        items = data.get("drug_mechanisms", []) or data.get("drug_mechanism", []) or []
        normalized: List[Dict[str, Any]] = []
        hint = (target_hint or "").lower().strip()
        for it in items:
            mech = (it.get("mechanism_of_action") or "").strip()
            action = (it.get("action_type") or "").strip()
            target_name = (it.get("target_name") or "").strip()
            target_chembl_id = (it.get("target_chembl_id") or "").strip()
            ref = (it.get("reference") or "").strip()
            if not mech and not action and not target_name:
                continue
            score = 0
            merged = f"{mech} {action} {target_name} {target_chembl_id}".lower()
            if hint and hint in merged:
                score += 2
            if "inhib" in merged:
                score += 1
            if "agon" in merged or "antagon" in merged:
                score += 1
            if "bind" in merged:
                score += 1
            normalized.append(
                {
                    "mechanism_of_action": mech,
                    "action_type": action,
                    "target_name": target_name,
                    "target_chembl_id": target_chembl_id,
                    "reference": ref,
                    "_score": score,
                }
            )
        normalized.sort(key=lambda x: x.get("_score", 0), reverse=True)
        for it in normalized:
            it.pop("_score", None)
        return normalized[:max_items]
    except Exception:
        return []


# M4-B2 P4：PubMed/NCBI 检索（fetch_pubmed_pmids / fetch_pubmed_titles / fetch_pubmed_abstract_snippet /
# fetch_pubmed_article_xml / extract_pubmed_title_abstract_from_xml / fetch_pmcid_for_pmid /
# fetch_pmc_fulltext_excerpt）已收敛至 core.pubmed，经 cached_get 获得 SQLite 缓存（B2）与
# NCBI 3 req/s 限流（B3）；顶部 import 保持模块命名空间兼容（demo 防腐层依赖这些名字）。


# M4-B2 P4：段落重组算法收敛至 core.text_utils，此处仅绑定本模块的段落标签
SECTION_TAGS = ["【Core Conclusion】", "【Protein Information】", "【Condition】", "【Clinical Significance】", "【Evidence】"]


def normalize_sections(answer: str) -> str:
    return core_text.normalize_sections(answer, SECTION_TAGS)


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
    normalized_evidence = evidence_block.strip()
    pmid_suffix = "; ".join(f"PMID:{p}" for p in unique_pmids[:4])
    if pmid_suffix not in normalized_evidence:
        normalized_evidence = f"{normalized_evidence} {pmid_suffix}"
    return answer[: evidence_match.start()] + normalized_evidence


def extract_evidence_text_for_support(state: PipelineState) -> str:
    evidence = state.get("evidence", {})
    parts: List[str] = []
    u = evidence.get("uniprot", {})
    if isinstance(u, dict):
        for k in ("protein_name", "gene", "function", "pathway", "catalytic_activity"):
            v = u.get(k)
            if v:
                parts.append(str(v))
    for it in evidence.get("chembl_mechanisms", []) or []:
        if isinstance(it, dict):
            for k in ("mechanism_of_action", "action_type", "target_name"):
                v = it.get(k)
                if v:
                    parts.append(str(v))
    for art in evidence.get("pubmed_articles", []) or []:
        if isinstance(art, dict):
            for k in ("title", "abstract", "fulltext_excerpt"):
                v = art.get(k)
                if v:
                    parts.append(str(v))
    return "\n".join(parts)


def find_unsupported_specifics(state: PipelineState) -> List[str]:
    answer = state.get("answer", "")
    evidence_text = extract_evidence_text_for_support(state).lower()
    issues: List[str] = []

    residue_matches = re.findall(r"\b(?:Ser|Cys|Lys|Asp|Glu|Arg|His|Tyr|Thr|Gly|Ala|Val|Leu|Ile|Pro|Asn|Gln|Met|Phe|Trp)\s*-?\s*(\d{2,4})\b", answer)
    if residue_matches:
        for m in set(residue_matches):
            token = m.lower()
            if token and token not in evidence_text:
                issues.append("unsupported_residue_number_detail")
                break

    if re.search(r"\bIC50\b|\bKi\b|\bKd\b", answer):
        if not re.search(r"\bIC50\b|\bKi\b|\bKd\b", evidence_text, flags=re.IGNORECASE):
            issues.append("unsupported_binding_affinity_numbers")

    if re.search(r"\bATP-binding\b|\bATP binding\b", answer, flags=re.IGNORECASE):
        if "atp" not in evidence_text:
            issues.append("unsupported_atp_binding_claim")

    return sorted(set(issues))

RESPONDER_PROMPT = """You are a biomedical domain writer creating high-quality training samples.
You must answer based only on the provided evidence package.
Do not add facts not supported by provided IDs or snippets.

Output must keep this structure:
【Core Conclusion】...
【Protein Information】...
【Condition】...
【Clinical Significance】...
【Evidence】...

Core Conclusion requirements:
- Explain ONLY how the drug changes the target protein's activity or function.
- Include the interaction type if supported (e.g., inhibitor/agonist/antagonist/binder) and the immediate biochemical consequence (e.g., blocks catalytic activity, prevents phosphorylation).
- Avoid background, definitions, and broad pharmacology in this section.
- Do not include residue positions, numeric constants, binding pocket details, or “typical” claims unless explicitly supported by the evidence package text.

Other constraints:
1) Avoid long bullet stacks. Prefer concise paragraphs.
2) Evidence must cite UniProt, DrugBank and 1-3 PMIDs from the provided list only. NEVER fabricate or recall PMIDs from your training data. You may ONLY use PMIDs listed under "Allowed PMIDs for citation". Any PMID not in that list will cause automatic rejection.
3) Each PMID cited must correspond to at least one concrete mechanism statement in the answer.
4) Always include the UniProt ID and the drug name in 【Protein Information】.
5) If evidence is insufficient, explicitly state "Direct experimental evidence linking this drug to this specific protein target is limited in the provided package" in the Evidence section. Do not extrapolate from studies on different enzymes or organisms to fill the gap.
6) English only."""


VERIFIER_PROMPT = """You are a strict biomedical quality auditor.
Score the answer using only the given question, pair metadata and evidence package.
Return JSON only:
{
  "factuality_score": 0-4,
  "grounding_score": 0-3,
  "format_score": 0-2,
  "consistency_score": 0-1,
  "errors": ["..."],
  "critical_errors": ["..."],
  "revision_advice": ["..."]
}

Factuality rubric (0-4):
0 = core mechanism is wrong or contradicts evidence
1 = major mechanism mismatch or mostly unsupported claims
2 = partially correct direction but key mechanism links unsupported
3 = mostly correct mechanism with minor imprecision
4 = mechanism is directly supported and internally consistent

Critical errors include unsupported evidence, wrong target/drug entity, major biomedical contradiction."""


REFINER_PROMPT = """You are a biomedical editor.
Revise the draft answer by fixing listed errors.
Keep all five required sections with the same tags.
Stay strictly within provided evidence package.
Prefer concise training-style scientific writing.
English only."""


def node_collect(state: PipelineState) -> PipelineState:
    pair = state["pair"]
    uniprot_core = fetch_uniprot_core(pair["uniprot_id"])
    chembl = fetch_chembl(pair.get("chembl_id", ""))
    chembl_mechanisms = fetch_chembl_drug_mechanisms(pair.get("chembl_id", ""), target_hint=pair.get("gene", pair.get("protein", "")))
    gene = pair.get("gene", "") or pair.get("protein", "")
    drug = pair["drug"]
    protein = pair["protein"]

    pubmed_query = (
        f"({drug}[Title/Abstract]) AND "
        f"(({protein}[Title/Abstract]) OR ({gene}[Title/Abstract])) AND "
        f"(mechanism[Title/Abstract] OR \"mode of action\"[Title/Abstract] OR inhibit*[Title/Abstract] OR bind*[Title/Abstract])"
    )
    candidate_pmids = fetch_pubmed_pmids(pubmed_query, retmax=30)
    if len(candidate_pmids) < 3:
        fallback_query1 = f"({drug}[Title/Abstract]) AND (({protein}[Title/Abstract]) OR ({gene}[Title/Abstract]))"
        extra = fetch_pubmed_pmids(fallback_query1, retmax=20)
        for p in extra:
            if p not in candidate_pmids:
                candidate_pmids.append(p)
    if len(candidate_pmids) < 3:
        fallback_query2 = f"{drug} {protein} mechanism"
        extra = fetch_pubmed_pmids(fallback_query2, retmax=20)
        for p in extra:
            if p not in candidate_pmids:
                candidate_pmids.append(p)
    if len(candidate_pmids) < 3 and gene and gene.lower() != protein.lower():
        fallback_query3 = f"{drug} {gene} mechanism"
        extra = fetch_pubmed_pmids(fallback_query3, retmax=20)
        for p in extra:
            if p not in candidate_pmids:
                candidate_pmids.append(p)
    titles = fetch_pubmed_titles(candidate_pmids)

    token_pool = {drug.lower(), protein.lower(), str(gene).lower()}
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
        if "inhibit" in title or "inhibition" in title:
            score += 0.4
        if "bind" in title or "binding" in title:
            score += 0.3
        if "resistance" in title:
            score -= 0.4
        if "review" in title:
            score -= 0.2
        return score

    ranked_pmids = sorted(candidate_pmids, key=rank_pmid, reverse=True)
    pmids = ranked_pmids[:6]
    titles = fetch_pubmed_titles(pmids)
    pmids = [p for p in pmids if p]
    articles: List[Dict[str, Any]] = []
    for pmid in pmids[:3]:
        xml_text = fetch_pubmed_article_xml(pmid)
        meta = extract_pubmed_title_abstract_from_xml(xml_text) if xml_text else {"title": titles.get(pmid, ""), "abstract": "", "journal": "", "year": ""}
        article = {
            "pmid": pmid,
            "pmcid": "",
            "title": meta.get("title", "") or titles.get(pmid, ""),
            "journal": meta.get("journal", ""),
            "year": meta.get("year", ""),
            "abstract": (meta.get("abstract", "") or "")[:1200],
            "fulltext_excerpt": "",
        }
        articles.append(article)

    question = f"What is the mechanism by which {pair['drug']} affects {pair['protein']}?"
    state["question"] = question
    state["evidence"] = {
        "uniprot": uniprot_core,
        "chembl": chembl,
        "chembl_mechanisms": chembl_mechanisms,
        "pubmed_pmids": pmids,
        "pubmed_titles": titles,
        "pubmed_articles": articles,
        "drugbank_id": pair.get("drugbank_id", ""),
    }
    return state


def build_generation_user_prompt(state: PipelineState) -> str:
    pair = state["pair"]
    evidence = state["evidence"]
    uniprot = evidence["uniprot"]
    chembl = evidence["chembl"]
    chembl_mechanisms = evidence.get("chembl_mechanisms", [])
    pmids = evidence.get("pubmed_pmids", [])
    articles = evidence.get("pubmed_articles", [])
    lines = []
    lines.append(f"Question: {state['question']}")
    lines.append(f"Drug: {pair['drug']}")
    lines.append(f"Protein: {pair['protein']}")
    lines.append(f"UniProt ID: {pair['uniprot_id']}")
    lines.append(f"DrugBank ID: {pair.get('drugbank_id', 'N/A')}")
    lines.append(f"ChEMBL ID: {pair.get('chembl_id', 'N/A')}")
    lines.append("")
    lines.append("Evidence Package (focus on drug-target mechanism)")
    if uniprot.get("protein_name"):
        lines.append(f"UniProt protein name: {uniprot.get('protein_name', '')}")
    if uniprot.get("function"):
        lines.append(f"UniProt function (core): {uniprot.get('function', '')[:600]}")
    if chembl.get("pref_name"):
        lines.append(f"ChEMBL drug name: {chembl.get('pref_name', '')}")
    if chembl_mechanisms:
        lines.append("")
        lines.append("ChEMBL Drug Mechanism Records:")
        for it in chembl_mechanisms[:3]:
            mech = it.get("mechanism_of_action", "")
            action = it.get("action_type", "")
            tgt = it.get("target_name", "")
            ref = it.get("reference", "")
            lines.append(f"- action_type: {action}; target: {tgt}; mechanism: {mech}; reference: {ref}")
    lines.append("")
    lines.append("Allowed PMIDs for citation:")
    lines.append("; ".join(f"PMID:{p}" for p in pmids if p))
    if articles:
        lines.append("")
        lines.append("PubMed Articles (title + abstract):")
        for art in articles:
            lines.append(f"- PMID:{art.get('pmid','')} | {art.get('title','')}")
            if art.get("abstract"):
                lines.append(f"  abstract: {art.get('abstract','')[:900]}")
    return "\n".join(lines)


def node_generate(state: PipelineState) -> PipelineState:
    client = get_client()
    user_prompt = build_generation_user_prompt(state)
    answer = llm_call(client, RESPONDER_PROMPT, user_prompt, temperature=0.2, max_tokens=1500)
    answer = normalize_sections(answer)
    answer = normalize_evidence_citations(answer)
    state["answer"] = answer
    if "best_score" not in state:
        state["best_score"] = -1.0
    return state


def deterministic_checks(state: PipelineState) -> Dict[str, Any]:
    pair = state["pair"]
    answer = state.get("answer", "")
    evidence = state.get("evidence", {})
    allowed_pmids = set(evidence.get("pubmed_pmids", []))
    tags = ["【Core Conclusion】", "【Protein Information】", "【Condition】", "【Clinical Significance】", "【Evidence】"]
    errors: List[str] = []
    critical: List[str] = []

    missing = [t for t in tags if t not in answer]
    if missing:
        errors.append(f"missing_sections:{','.join(missing)}")

    if pair["drug"].lower() not in answer.lower():
        errors.append("drug_name_missing_or_inconsistent")
    if pair["uniprot_id"].lower() not in answer.lower():
        errors.append("uniprot_id_missing_or_inconsistent")

    pmids_in_answer = extract_pmids(answer)
    unsupported = [p for p in pmids_in_answer if p not in allowed_pmids]
    if unsupported:
        critical.append(f"unsupported_pmids:{','.join(sorted(set(unsupported)))}")

    if allowed_pmids and len(pmids_in_answer) == 0:
        errors.append("evidence_section_missing_pmid")

    if len(pmids_in_answer) != len(set(pmids_in_answer)):
        pass # errors.append("duplicate_pmids_in_evidence")

    if pair["protein"].lower() not in answer.lower():
        errors.append("protein_name_missing_or_inconsistent")

    unsupported_specifics = find_unsupported_specifics(state)
    if unsupported_specifics:
        critical.extend(unsupported_specifics)

    format_score = 2 if not missing else max(0, 2 - len(missing))
    grounding_score = 3
    if unsupported:
        grounding_score = 0
    elif len(pmids_in_answer) == 0:
        grounding_score = 1
    consistency_score = 1 if (
        "drug_name_missing_or_inconsistent" not in errors
        and "uniprot_id_missing_or_inconsistent" not in errors
        and "protein_name_missing_or_inconsistent" not in errors
    ) else 0

    return {
        "det_format_score": format_score,
        "det_grounding_score": grounding_score,
        "det_consistency_score": consistency_score,
        "det_errors": errors,
        "det_critical_errors": critical,
        "pmids_in_answer": pmids_in_answer,
    }


# M4-B2 P4：parse_json_object 已收敛至 core.json_utils（顶部 import，实现见 core/json_utils.py）


def node_verify(state: PipelineState) -> PipelineState:
    client = get_client()
    pair = state["pair"]
    evidence = state["evidence"]
    answer = state["answer"]
    det = deterministic_checks(state)
    user_prompt = "\n".join(
        [
            f"Question: {state['question']}",
            f"Drug: {pair['drug']}",
            f"Protein: {pair['protein']}",
            f"UniProt ID: {pair['uniprot_id']}",
            f"DrugBank ID: {pair.get('drugbank_id', 'N/A')}",
            f"Allowed PMIDs: {', '.join(evidence.get('pubmed_pmids', []))}",
            f"UniProt function snippet: {evidence.get('uniprot', {}).get('function', '')}",
            "",
            "Answer:",
            answer,
        ]
    )
    raw = llm_call(client, VERIFIER_PROMPT, user_prompt, temperature=0.0, max_tokens=900)
    # M5-B 第一层：解析 + Pydantic 结构化校验（格式幻觉降级，错误精确到字段）
    llm_judge, judge_err = validate_verifier_output(parse_json_object(raw), "four_dim")
    if judge_err:
        # M5-B 第二层：instructor 结构化抢救（尽力而为，失败回退降级）
        rescued = core_llm.llm_call_json(client, VERIFIER_PROMPT, user_prompt, StrictVerifierJudge, model=MODEL, max_tokens=900)
        if rescued is not None:
            llm_judge, judge_err = rescued, None
    if judge_err:
        llm_judge = {
            "factuality_score": 1,
            "grounding_score": 1,
            "format_score": 1,
            "consistency_score": 1,
            "errors": [judge_err],
            "critical_errors": [],
            "revision_advice": [],
        }

    factuality = float(llm_judge.get("factuality_score", 1))
    grounding = float(llm_judge.get("grounding_score", 1))
    format_score = float(llm_judge.get("format_score", 1))
    consistency = float(llm_judge.get("consistency_score", 1))
    llm_errors = llm_judge.get("errors", []) if isinstance(llm_judge.get("errors", []), list) else []
    llm_critical_raw = llm_judge.get("critical_errors", []) if isinstance(llm_judge.get("critical_errors", []), list) else []
    advice = llm_judge.get("revision_advice", []) if isinstance(llm_judge.get("revision_advice", []), list) else []
    llm_critical: List[str] = []
    llm_noncritical_from_critical: List[str] = []
    critical_markers = (
        "unsupported",
        "wrong target",
        "wrong drug",
        "contradiction",
        "inconsistent id",
        "fabricated",
        "halluc",
        "not in the allowed",
        "not included in the allowed",
        "evidence does not support",
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

    all_errors = det["det_errors"] + llm_errors + llm_noncritical_from_critical
    all_critical = det["det_critical_errors"] + llm_critical

    if all_critical:
        # Check if the ONLY critical issue is unsupported_pmids (mechanism itself may be correct)
        only_pmid_issue = all(str(e).startswith("unsupported_pmids") for e in all_critical)
        if only_pmid_issue and factuality >= 3.0:
            factuality_base = 2.0  # Softer penalty: mechanism is likely correct, just citation is wrong
        else:
            factuality_base = 0.0
    elif any(
        x in det["det_errors"]
        for x in (
            "drug_name_missing_or_inconsistent",
            "protein_name_missing_or_inconsistent",
            "uniprot_id_missing_or_inconsistent",
        )
    ):
        factuality_base = 1.0
    elif "evidence_section_missing_pmid" in det["det_errors"]:
        factuality_base = 2.0
    elif len(llm_errors) >= 3:
        factuality_base = 2.0
    elif len(det["det_errors"]) == 0 and len(llm_errors) == 0:
        factuality_base = 4.0
    else:
        factuality_base = 3.0

    if factuality >= 3.5:
        factuality_adjust = 1.0
    elif factuality <= 1.5:
        factuality_adjust = -1.0
    else:
        factuality_adjust = 0.0

    factuality_effective = max(0.0, min(4.0, factuality_base + factuality_adjust))

    blended_score = factuality_effective + min(grounding, det["det_grounding_score"]) + min(format_score, det["det_format_score"]) + min(consistency, det["det_consistency_score"])
    error_penalty = min(1.5, 0.25 * len(det["det_errors"]) + 0.15 * len(llm_errors))
    blended_score -= error_penalty
    blended_score = round(float(blended_score), 2)
    if all_critical:
        blended_score = min(blended_score, 5.9)
    correction_count = state.get("correction_count", 0)
    max_corrections = state.get("max_corrections", 2)

    if not all_critical and blended_score > state.get("best_score", -1):
        state["best_score"] = blended_score
        state["best_answer"] = answer

    pass_now = blended_score >= 8.0 and len(all_critical) == 0
    if pass_now:
        route: Literal["pass", "refine", "regenerate", "stop"] = "pass"
    elif correction_count >= max_corrections:
        route = "stop"
    elif len(all_critical) > 0 or blended_score < 6.0:
        route = "regenerate"
    else:
        route = "refine"

    state["verification"] = {
        "score": blended_score,
        "subscores": {
            "factuality_base": factuality_base,
            "factuality_llm_raw": factuality,
            "factuality_effective": factuality_effective,
            "grounding_llm_raw": grounding,
            "format_llm_raw": format_score,
            "consistency_llm_raw": consistency,
        },
        "errors": all_errors,
        "critical_errors": all_critical,
        "revision_advice": advice,
        "raw_llm_judge": llm_judge,
        "deterministic": det,
    }
    state["route"] = route
    return state


def node_refine(state: PipelineState) -> PipelineState:
    client = get_client()
    verification = state.get("verification", {})
    advice = verification.get("revision_advice", [])
    errors = verification.get("errors", [])
    pair = state["pair"]
    evidence = state["evidence"]
    user_prompt = "\n".join(
        [
            f"Question: {state['question']}",
            f"Drug: {pair['drug']}",
            f"Protein: {pair['protein']}",
            f"UniProt ID: {pair['uniprot_id']}",
            f"DrugBank ID: {pair.get('drugbank_id', 'N/A')}",
            f"Allowed PMIDs: {', '.join(evidence.get('pubmed_pmids', []))}",
            "Errors to fix:",
            *[f"- {e}" for e in errors],
            "Revision advice:",
            *[f"- {a}" for a in advice],
            "",
            "Current answer:",
            state["answer"],
        ]
    )
    revised = llm_call(client, REFINER_PROMPT, user_prompt, temperature=0.2, max_tokens=1700)
    revised = normalize_sections(revised)
    revised = normalize_evidence_citations(revised)
    state["answer"] = revised
    state["correction_count"] = state.get("correction_count", 0) + 1
    return state


def node_regenerate(state: PipelineState) -> PipelineState:
    client = get_client()
    verification = state.get("verification", {})
    critical = verification.get("critical_errors", [])
    user_prompt = build_generation_user_prompt(state) + "\n\nKnown critical issues to avoid:\n" + "\n".join(f"- {x}" for x in critical)
    regenerated = llm_call(client, RESPONDER_PROMPT, user_prompt, temperature=0.15, max_tokens=1700)
    regenerated = normalize_sections(regenerated)
    regenerated = normalize_evidence_citations(regenerated)
    state["answer"] = regenerated
    state["correction_count"] = state.get("correction_count", 0) + 1
    return state


def route_from_verify(state: PipelineState) -> str:
    return state.get("route", "stop")


def node_finalize(state: PipelineState) -> PipelineState:
    pair = state["pair"]
    current_answer = state.get("answer", "")
    best_answer = state.get("best_answer", current_answer)
    verification = state.get("verification", {})
    final_score = state.get("best_score", -1)
    if final_score < 0:
        final_score = verification.get("score", 0)
    sample = {
        "question": state.get("question", ""),
        "answer": best_answer,
        "category": "Basic-Drug Interaction",
        "drug": pair["drug"],
        "protein": pair["protein"],
        "uniprot_id": pair["uniprot_id"],
        "quality_score": final_score,
        "quality_log": {
            "last_route": state.get("route", "stop"),
            "correction_count": state.get("correction_count", 0),
            "subscores": verification.get("subscores", {}),
            "errors": verification.get("errors", []),
            "critical_errors": verification.get("critical_errors", []),
        },
    }
    state["final_sample"] = sample
    return state


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("collect", node_collect)
    graph.add_node("generate", node_generate)
    graph.add_node("verify", node_verify)
    graph.add_node("refine", node_refine)
    graph.add_node("regenerate", node_regenerate)
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("collect")
    graph.add_edge("collect", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        route_from_verify,
        {
            "pass": "finalize",
            "refine": "refine",
            "regenerate": "regenerate",
            "stop": "finalize",
        },
    )
    graph.add_edge("refine", "verify")
    graph.add_edge("regenerate", "verify")
    graph.add_edge("finalize", END)
    return graph.compile()


# M4-B2 P4：run_one 已收敛至 core.batch（顶部 import）；run_batch 收敛为 core 骨架的领域配置调用


def run_batch(pairs: List[Dict[str, Any]], output_file: Path, max_corrections: int = 2, max_workers: int = 3) -> List[Dict[str, Any]]:
    app = build_graph()
    return core_batch.run_batch(
        pairs,
        output_file,
        process_one=lambda pair: run_one(app, pair, max_corrections=max_corrections),
        pair_id=lambda p: f"{p.get('drug')}::{p.get('protein')}",
        describe=lambda r: f"{r.get('drug')} -> {r.get('protein')}",
        error_fields=lambda p: {"drug": p["drug"], "protein": p["protein"]},
        max_workers=max_workers,
        batch_label="drug-target items",
        summarize=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None, help="Number of items to test (from start)")
    parser.add_argument("--start", type=int, default=0, help="Start index of data pairs")
    parser.add_argument("--end", type=int, default=None, help="End index of data pairs")
    parser.add_argument("--workers", type=int, default=3, help="Max number of concurrent threads")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--max-corrections", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(PAIRS_FILE):
        raise FileNotFoundError(f"输入文件不存在: {PAIRS_FILE}")
    import json
    with open(PAIRS_FILE, "r", encoding="utf-8") as f:
        pairs = json.load(f)
        
    start_idx = args.start
    end_idx = args.end
    
    if args.count is not None and args.count > 0:
        end_idx = start_idx + args.count

    if end_idx is not None:
        pairs = pairs[start_idx:end_idx]
    else:
        pairs = pairs[start_idx:]
        
    run_batch(pairs, args.output, max_corrections=args.max_corrections, max_workers=args.workers)

if __name__ == "__main__":
    main()
