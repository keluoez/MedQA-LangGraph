"""M5 离线测试：A 机制自省（无网络 / 无 LLM 调用）。

覆盖点：
- introspect 返回 4 领域完整元数据（检索/评分/路由/prompt）；
- prompt 全文直接引用模块运行时常量（非拷贝，永不漂移）；
- context_preview 由各模块真实 user_prompt 构建函数渲染（截断等行为与线上一致）；
- 路由已注册到 FastAPI app。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.adapters import pipeline_bridge


def _introspect() -> dict:
    return pipeline_bridge.introspect_pipeline()


def test_introspect_returns_four_domains():
    data = _introspect()
    assert set(data["domains"]) == {"drug", "experimental", "pathological", "physiological"}
    assert {a["id"] for a in data["agents"]} == {
        "collect", "generate", "verify", "refine", "regenerate", "finalize",
    }


def test_introspect_prompts_are_module_constants_not_copies():
    data = _introspect()
    from basic_drug_interaction import langgraph_pipeline as drug
    from basic_pathological_environment import langgraph_pipeline as patho
    from basic_physiological_conditions import langgraph_pipeline as phys

    # is 同一性：manifest 侧永远与运行代码一致
    assert data["domains"]["drug"]["prompts"]["responder"] is drug.RESPONDER_PROMPT
    assert data["domains"]["drug"]["prompts"]["verifier"] is drug.VERIFIER_PROMPT
    assert data["domains"]["pathological"]["prompts"]["responder"] is patho.PATHOLOGY_RESPONDER_PROMPT
    assert data["domains"]["physiological"]["prompts"]["refiner"] is phys.PHYSIOLOGY_REFINER_PROMPT
    for d in data["domains"].values():
        for text in d["prompts"].values():
            assert len(text) > 100  # prompt 全文非空


def test_introspect_context_preview_reflects_real_assembly():
    data = _introspect()

    drug_preview = data["domains"]["drug"]["context_preview"]
    assert "Question: What is the mechanism by which Aspirin affects COX-2?" in drug_preview
    assert "Allowed PMIDs for citation:" in drug_preview
    assert "PMID:25277428" in drug_preview
    assert "ChEMBL Drug Mechanism Records:" in drug_preview

    exp_preview = data["domains"]["experimental"]["context_preview"]
    assert "Experimental Condition: low pH" in exp_preview
    assert "UniProt catalytic activity:" in exp_preview

    patho_preview = data["domains"]["pathological"]["context_preview"]
    # top-2 文献 + Key finding 截断（>80 字符加省略号）+ 白名单强约束
    assert "Allowed PMIDs: 3097456, 12345678" in patho_preview
    assert "Key finding:" in patho_preview
    assert "IMPORTANT: You can only cite PMIDs from the allowed list above." in patho_preview

    phys_preview = data["domains"]["physiological"]["context_preview"]
    assert "ALLOWED PMIDs:" in phys_preview
    assert "REMINDER: Output ALL headers" in phys_preview


def test_introspect_metadata_completeness():
    data = _introspect()
    for domain_id, d in data["domains"].items():
        assert len(d["retrieval"]["steps"]) >= 4, domain_id
        assert d["retrieval"]["whitelist_size"] == 6, domain_id
        assert d["retrieval"]["detail_articles"] == 3, domain_id
        assert any(w["weight"] < 0 for w in d["retrieval"]["rank_weights"]), domain_id  # 有负向权重
        assert len(d["sections"]) == 5, domain_id
        assert len(d["scoring"]["dimensions"]) == 4, domain_id
        assert d["scoring"]["max_score"] == 10, domain_id
        assert {r["route"] for r in d["scoring"]["routing"]} == {"pass", "refine", "regenerate", "stop"}, domain_id
        assert len(d["tools"]) >= 2, domain_id
        assert d["model"], domain_id  # MODEL 经 getenv 绑定，非空


def test_mechanism_route_registered():
    from api.main import app

    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/pipeline/introspect" in paths


def test_patho_prompt_extraction_behavior_preserved():
    """M5-A 抽取的 build_pathology_user_prompt 与原 node_respond 内联构建逐字一致。"""
    from basic_pathological_environment import langgraph_pipeline as patho

    state = {
        "pair": {"disease": "D", "pathology": "P", "protein": "X", "uniprot_id": "U1"},
        "evidence": {
            "pubmed_pmids": ["111", "222"],
            "pubmed_articles": [
                {"pmid": "111", "title": "T1", "abstract": "a" * 100},  # >80 → 截断加省略号
                {"pmid": "222", "title": "T2", "abstract": "short"},    # ≤80 → 原样
            ],
        },
    }
    prompt = patho.build_pathology_user_prompt(state)
    assert "Allowed PMIDs: 111, 222" in prompt
    assert f"Key finding: {'a' * 80}..." in prompt
    assert "Key finding: short" in prompt
    assert "【Clinical Significance】" in prompt


# ── M5-B：Verifier Pydantic 结构化校验（第一层） + instructor 抢救（第二层） ──

from core import llm as core_llm
from core.schemas import PathologyVerifierJudge, VerifierJudge, validate_verifier_output


def test_schema_coerces_numeric_strings_and_wraps_lists():
    cleaned, err = validate_verifier_output(
        {
            "factuality_score": "3",       # 数字字符串 → 3.0
            "grounding_score": 2,
            "format_score": "1.5",
            "consistency_score": None,     # 缺失/None → score_default
            "errors": "single string",     # 单字符串 → 列表
            "critical_errors": None,       # None → []
            "revision_advice": [{"k": 1}],  # dict 元素 → JSON 字符串
            "extra_unknown": 123,           # 未知字段忽略
        },
        "four_dim",
    )
    assert err is None
    assert cleaned["factuality_score"] == 3.0
    assert cleaned["format_score"] == 1.5
    assert cleaned["consistency_score"] == 1.0
    assert cleaned["errors"] == ["single string"]
    assert cleaned["critical_errors"] == []
    assert cleaned["revision_advice"] == ['{"k": 1}']
    assert "extra_unknown" not in cleaned


def test_schema_clamps_out_of_range_scores():
    cleaned, err = validate_verifier_output(
        {"factuality_score": 9, "grounding_score": -2, "format_score": 5, "consistency_score": 3},
        "four_dim",
    )
    assert err is None
    assert cleaned["factuality_score"] == 4.0
    assert cleaned["grounding_score"] == 0.0
    assert cleaned["format_score"] == 2.0
    assert cleaned["consistency_score"] == 1.0


def test_schema_rejects_garbage_with_field_precise_error():
    cleaned, err = validate_verifier_output(
        {"factuality_score": "high", "grounding_score": 2, "format_score": 1, "consistency_score": 1},
        "four_dim",
    )
    assert cleaned is None
    assert "factuality_score" in err and "格式无效" in err


def test_schema_empty_input_and_pathology_kind():
    cleaned, err = validate_verifier_output({}, "pathology")
    assert cleaned is None and "为空" in err
    cleaned, err = validate_verifier_output("not a dict", "pathology")
    assert cleaned is None

    cleaned, err = validate_verifier_output(
        {"pathology_mechanism_score": 4, "disease_relevance_score": 3,
         "evidence_support_score": 2, "format_score": 1},
        "pathology",
    )
    assert err is None
    assert cleaned["pathology_mechanism_score"] == 4.0
    assert cleaned["format_score"] == 1.0  # patho format 上限 1


def test_llm_call_json_env_off_and_graceful_fallback(monkeypatch):
    # 环境开关关闭 → 直接 None（第一层兜底）
    monkeypatch.setenv("BIOREASON_STRUCTURED_OUTPUT", "0")
    assert core_llm.llm_call_json(object(), "s", "u", VerifierJudge, model="m") is None
    # 开关开启但 client 非法 / instructor 链路异常 → 吞异常返回 None，不击穿业务
    monkeypatch.setenv("BIOREASON_STRUCTURED_OUTPUT", "1")
    assert core_llm.llm_call_json(object(), "s", "u", VerifierJudge, model="m") is None


def test_drug_verify_degrades_on_hallucinated_json(monkeypatch):
    """旧代码 float('high') 会抛 ValueError 击穿样本；M5-B 后降级为默认分 + 字段级错误。"""
    from basic_drug_interaction import langgraph_pipeline as drug

    state = {
        "pair": {"drug": "Aspirin", "protein": "COX-2", "uniprot_id": "P23219", "gene": "PTGS2"},
        "question": "What is the mechanism by which Aspirin affects COX-2?",
        "answer": (
            "【Core Conclusion】x\n【Protein Information】Aspirin COX-2 P23219\n"
            "【Condition】x\n【Clinical Significance】x\n【Evidence】PMID:1\n"
        ),
        "evidence": {"pubmed_pmids": ["1"], "uniprot": {"function": "f"}},
        "correction_count": 0,
        "max_corrections": 2,
    }
    monkeypatch.setattr(drug, "get_client", lambda: object())
    monkeypatch.setattr(
        drug, "llm_call",
        lambda *a, **k: '{"factuality_score": "high", "grounding_score": 2, "format_score": 1, "consistency_score": 1}',
    )
    out = drug.node_verify(state)  # 不再抛异常
    assert any("factuality_score" in e for e in out["verification"]["errors"])
    assert 0 <= out["verification"]["score"] <= 10
    assert out["route"] in {"pass", "refine", "regenerate", "stop"}


def test_physio_verify_degrades_with_zero_defaults(monkeypatch):
    """phys 模块降级默认分为 0（与其既有 .get(...,0) 口径一致）。"""
    from basic_physiological_conditions import langgraph_pipeline as phys

    state = {
        "pair": {"condition": "fasting", "protein": "PYGL", "uniprot_id": "P11216"},
        "question": "What is the function of PYGL under fasting?",
        "answer": (
            "【Core Conclusion】x\n【Protein Information】PYGL\n【Condition】x\n"
            "【Physiological Importance】x\n【Evidence】PMID:1\n"
        ),
        "evidence": {"pubmed_pmids": ["1"], "uniprot": {}},
        "correction_count": 0,
        "max_corrections": 2,
    }
    monkeypatch.setattr(phys, "get_client", lambda: object())
    monkeypatch.setattr(phys, "llm_call", lambda *a, **k: "total garbage not json")
    out = phys.node_verify(state)
    assert out["verification"]["score"] == 0.0
    assert any("verifier JSON" in e for e in out["verification"]["errors"])
