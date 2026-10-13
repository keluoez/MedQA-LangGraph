"""M4 第二批重构（M4-B2）离线单元测试，无需 API key、无网络调用。

- 病理 12 分评分口径修复：verifier min(4.0) 钳制 + total_score 改为 LLM 四维之和（≤10）
- finalize quality_score 与 verifier 同口径（≤10）
- 路由阈值语义自洽（≥9 或 ≥8 且无 critical → pass；critical 且 <7 → regenerate；
  修正次数耗尽 → stop；其余 → refine）
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basic_pathological_environment import langgraph_pipeline as patho


PAIR = {
    "disease": "lung adenocarcinoma",
    "pathology": "lung adenocarcinoma",
    "protein": "EGFR",
    "uniprot_id": "P00533",
}

GOOD_ANSWER = """【Core Conclusion】 EGFR drives tumor growth in lung adenocarcinoma via aberrant kinase activation.
【Protein Information】 EGFR (UniProt: P00533) is a receptor tyrosine kinase.
【Condition】 Lung adenocarcinoma is often driven by EGFR mutation.
【Clinical Significance】 EGFR inhibition improves clinical outcomes.
【Evidence】 PMID:12345678 supports the mechanism.
"""

BAD_PMID_ANSWER = GOOD_ANSWER.replace("PMID:12345678", "PMID:99999999")

EVIDENCE = {
    "pubmed_pmids": ["12345678"],
    "uniprot": {"function": "Receptor tyrosine kinase binding EGF."},
}


def _base_state(answer=GOOD_ANSWER, correction_count=0, max_corrections=2):
    return {
        "pair": dict(PAIR),
        "question": "What is the role of EGFR in lung adenocarcinoma?",
        "evidence": dict(EVIDENCE),
        "answer": answer,
        "correction_count": correction_count,
        "max_corrections": max_corrections,
    }


def _patch_llm(monkeypatch, judge: dict):
    monkeypatch.setattr(patho, "get_client", lambda: object())
    monkeypatch.setattr(patho, "llm_call", lambda *a, **k: json.dumps(judge))


def _judge(mech, disease, evidence, fmt):
    return {
        "pathology_mechanism_score": mech,
        "disease_relevance_score": disease,
        "evidence_support_score": evidence,
        "format_score": fmt,
        "errors": [],
        "critical_errors": [],
        "revision_advice": [],
    }


MAX_JUDGE = _judge(4.0, 3.0, 2.0, 1.0)


def _dims_sum(verification: dict) -> float:
    return (
        verification["pathology_mechanism_score"]
        + verification["disease_relevance_score"]
        + verification["evidence_support_score"]
        + verification["format_score"]
    )


# ---------- 病理评分钳制与口径 ----------

def test_verifier_clamps_mechanism_to_4_and_total_to_10(monkeypatch):
    # 历史越界场景：det 全净 + LLM 全满 → 旧代码 base4+adjust1=5、再混加 det format(2)/consistency(1) → 12/13 分
    _patch_llm(monkeypatch, dict(MAX_JUDGE))
    state = patho.node_verify(_base_state())

    v = state["verification"]
    # min(4.0) 钳制生效：base 4 + adjust 1 → 4 而非 5
    assert v["pathology_mechanism_score"] == 4.0
    assert v["subscores"]["pathology_mechanism_base"] == 4.0
    assert v["subscores"]["pathology_mechanism_effective"] == 4.0
    # total_score 为 LLM 四维之和，上限 10
    assert _dims_sum(v) == 10.0
    # 顶层 format_score 为 LLM 口径（0-1），det 明细保留在 deterministic 键
    assert v["format_score"] == 1.0
    assert v["deterministic"]["format_score"] == 2
    assert v["deterministic"]["consistency_score"] == 1
    assert state["route"] == "pass"


def test_verifier_ignores_det_scores_in_total(monkeypatch):
    # det format/consistency 分数再高也不应抬高总分：构造 det 全净（format2/consistency1）但 LLM 低分
    _patch_llm(monkeypatch, _judge(2.0, 1.0, 0.5, 0.0))
    state = patho.node_verify(_base_state())
    v = state["verification"]
    # base 4（det/llm 无错）+ adjust 0 → effective 4；total = 4+1+0.5+0 = 5.5，与 det 分数无关
    assert v["pathology_mechanism_score"] == 4.0
    assert _dims_sum(v) == 5.5
    assert _dims_sum(v) <= 10.0


def test_finalize_quality_score_matches_verifier_and_caps_at_10(monkeypatch):
    _patch_llm(monkeypatch, dict(MAX_JUDGE))
    state = patho.node_verify(_base_state())
    state = patho.node_finalize(state)
    sample = state["final_sample"]
    assert sample["quality_score"] == 10.0
    assert sample["quality_score"] <= 10.0  # 旧代码此处可达 12
    assert sample["category"] == "Basic-Pathological Environment"
    # quality_log 携带四维 + deterministic 明细，便于事后审计
    assert _dims_sum(sample["quality_log"]) == 10.0
    assert "deterministic" in sample["quality_log"]


def test_verifier_parse_failure_fallback_keeps_score_within_bounds(monkeypatch):
    # LLM 输出不可解析 → 全 0 降级，评分仍需在 10 分口径内
    monkeypatch.setattr(patho, "get_client", lambda: object())
    monkeypatch.setattr(patho, "llm_call", lambda *a, **k: "not a json at all")
    state = patho.node_verify(_base_state())
    v = state["verification"]
    assert _dims_sum(v) <= 10.0
    # base 3（有 1 条解析错误）+ adjust -1（raw 0）→ effective 2
    assert v["pathology_mechanism_score"] == 2.0
    assert state["route"] == "refine"  # 5 分无 critical 且修正次数未耗尽


# ---------- 路由阈值语义（保持既有 9/8/7 阈值不变）----------

def test_route_pass_at_8_without_critical(monkeypatch):
    # raw 3.0 → effective 4.0（base4+adjust0），total = 4+2+1.5+0.5 = 8.0 → pass
    _patch_llm(monkeypatch, _judge(3.0, 2.0, 1.5, 0.5))
    state = patho.node_verify(_base_state())
    assert _dims_sum(state["verification"]) == 8.0
    assert state["route"] == "pass"


def test_route_refine_between_7_and_8(monkeypatch):
    _patch_llm(monkeypatch, _judge(3.0, 2.0, 1.5, 0.0))  # total 7.5
    state = patho.node_verify(_base_state())
    assert state["route"] == "refine"


def test_route_stop_when_corrections_exhausted(monkeypatch):
    _patch_llm(monkeypatch, _judge(3.0, 2.0, 1.5, 0.0))  # total 7.5，未达 pass 线
    state = patho.node_verify(_base_state(correction_count=2, max_corrections=2))
    assert state["route"] == "stop"


def test_route_regenerate_on_critical_and_low_score(monkeypatch):
    # unsupported PMID → det critical；base 0 → effective 0；total = 0+2+1.5+0.5 = 4.0 < 7 → regenerate
    _patch_llm(monkeypatch, _judge(2.0, 2.0, 1.5, 0.5))
    state = patho.node_verify(_base_state(answer=BAD_PMID_ANSWER))
    v = state["verification"]
    assert v["critical_errors"]
    assert v["pathology_mechanism_score"] == 0.0  # 下界钳制保留
    assert _dims_sum(v) <= 10.0
    assert state["route"] == "regenerate"
