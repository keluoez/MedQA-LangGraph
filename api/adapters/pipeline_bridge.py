"""防腐层：整个 api 演示项目中唯一感知原有 backend 代码/数据布局的文件。

阶段 2 重构 backend 目录结构时，只需更新本文件的路径映射与 import，
api 层其余代码（事件协议、SSE、mock 回放）不受影响。
"""
from pathlib import Path

# api/adapters/pipeline_bridge.py -> api -> 项目根
ROOT = Path(__file__).resolve().parents[2]

# 领域 -> 已生成数据集文件（output/*.jsonl）
# 注：combo（联合用药）是早期函数式管线产物，输出无 quality_score/quality_log，
# 不在论文 22,571 统计口径内，故不纳入演示数据集；待 M4 重构迁移 LangGraph 后再启用。
DOMAIN_FILES: dict[str, Path] = {
    "drug": ROOT / "basic_drug_interaction" / "output" / "langgraph_batch_output.jsonl",
    "experimental": ROOT / "basic_experimental_conditions" / "output" / "langgraph_batch_output.jsonl",
    "pathological": ROOT / "basic_pathological_environment" / "output" / "langgraph_batch_output.jsonl",
    "physiological": ROOT / "basic_physiological_conditions" / "output" / "langgraph_batch_output.jsonl",
}

DOMAIN_LABELS: dict[str, str] = {
    "drug": "Drug-Target Interaction",
    "experimental": "Experimental Conditions",
    "pathological": "Pathological Environment",
    "physiological": "Physiological Conditions",
}

# live 模式当前使用的生成管线（basic_drug_interaction 为主参考实现）。
# 延迟 import：仅在 live 模式真正需要跑管线时才加载 openai/langgraph 依赖链。
PAIRS_FILE = ROOT / "basic_drug_interaction" / "drug_target_pairs.json"
RECORDED_RUNS_DIR = ROOT / "api" / "mock" / "recorded_runs"


def build_graph():
    """live 模式：编译 LangGraph 状态图（唯一 import 原业务代码的位置）。"""
    from basic_drug_interaction.langgraph_pipeline import build_graph as _build

    return _build()


def fetch_evidence_apis():
    """半合成 mock 生成用：公开免费检索 API（UniProt/ChEMBL/PubMed，无需 LLM 密钥）。"""
    from basic_drug_interaction import langgraph_pipeline as lp

    return {
        "fetch_uniprot_core": lp.fetch_uniprot_core,
        "fetch_chembl": lp.fetch_chembl,
        "fetch_chembl_drug_mechanisms": lp.fetch_chembl_drug_mechanisms,
        "fetch_pubmed_pmids": lp.fetch_pubmed_pmids,
        "fetch_pubmed_titles": lp.fetch_pubmed_titles,
        "fetch_pubmed_article_xml": lp.fetch_pubmed_article_xml,
        "extract_pubmed_title_abstract_from_xml": lp.extract_pubmed_title_abstract_from_xml,
    }


def introspect_pipeline() -> dict:
    """M5-A 机制自省：合并静态 manifest 与运行时事实（prompt 全文 / MODEL / 真实上下文拼接预览）。

    延迟 import 各领域模块（模块加载本身无网络请求，离线安全）；
    prompt 全文直接取自模块常量，保证与运行代码永远一致；
    context_preview 调用各模块真实的 user_prompt 构建函数渲染占位 demo 状态，
    展示 Responder 实际收到的上下文拼接结构（截断长度等行为与线上完全相同）。
    """
    from importlib import import_module

    from . import mechanism_manifest as mm

    domains = {}
    for domain_id, spec in mm.domain_specs().items():
        mod = import_module(spec["module"])
        entry = {
            k: v
            for k, v in spec.items()
            if k not in ("module", "prompt_attrs", "context_builder", "demo_pair", "demo_evidence")
        }
        entry["model"] = getattr(mod, "MODEL", "")
        # prompt 全文（运行时常量，非拷贝）
        entry["prompts"] = {
            role: getattr(mod, attr) for role, attr in spec["prompt_attrs"].items()
        }
        # 真实上下文拼接预览：模块自身的构建函数 + 占位 demo 状态（无网络调用）
        demo_state = {
            "pair": spec["demo_pair"],
            "question": spec["question_template"].format(**spec["demo_pair"]),
            "evidence": spec["demo_evidence"],
        }
        builder = getattr(mod, spec["context_builder"])
        entry["context_preview"] = builder(demo_state)
        domains[domain_id] = entry

    return {"agents": mm.agents_flow(), "domains": domains}
