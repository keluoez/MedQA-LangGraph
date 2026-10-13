"""M5-B：Verifier 输出的 Pydantic 结构化校验（第一层防线）。

parse_json_object 之后、业务消费之前，对 verifier JSON 做模式校验：
- 分数字段：int/float/数字字符串可 coerce（"3" → 3.0）；缺失/None → 调用方给定的默认分；
  无法解析（如 "high"、列表、对象）→ 硬失败，返回精确到字段的错误消息；
- 越界分数：钳制回合法区间（与既有 min()/base 逻辑双保险，不做硬失败）；
- errors/critical_errors/revision_advice：None → []；单字符串 → [str]；dict → JSON 字符串；元素一律 str 化。

validate_verifier_output(data, kind) 返回 (cleaned, None) 或 (None, err_msg)。
调用方拿错误消息走降级（默认分 + errors 记录），格式幻觉不再击穿整条样本。
"""
import json
from typing import Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


def _clamp(v, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _str_list(v):
    """列表字段宽容规整：LLM 常见幻觉形态（None / 单字符串 / dict / 混合列表）→ 字符串列表。"""
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        return [json.dumps(v, ensure_ascii=False)]
    if isinstance(v, (list, tuple)):
        return [json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x) for x in v]
    return [str(v)]


class VerifierJudge(BaseModel):
    """drug / experimental / physiological 共用四维口径：0-4 / 0-3 / 0-2 / 0-1。"""
    model_config = ConfigDict(extra="ignore")

    factuality_score: Optional[float] = None
    grounding_score: Optional[float] = None
    format_score: Optional[float] = None
    consistency_score: Optional[float] = None
    errors: List[str] = []
    critical_errors: List[str] = []
    revision_advice: List[str] = []

    @field_validator("factuality_score", mode="after")
    @classmethod
    def _clamp_fact(cls, v):
        return None if v is None else _clamp(v, 0, 4)

    @field_validator("grounding_score", mode="after")
    @classmethod
    def _clamp_ground(cls, v):
        return None if v is None else _clamp(v, 0, 3)

    @field_validator("format_score", mode="after")
    @classmethod
    def _clamp_fmt(cls, v):
        return None if v is None else _clamp(v, 0, 2)

    @field_validator("consistency_score", mode="after")
    @classmethod
    def _clamp_cons(cls, v):
        return None if v is None else _clamp(v, 0, 1)

    @field_validator("errors", "critical_errors", "revision_advice", mode="before")
    @classmethod
    def _wrap_lists(cls, v):
        return _str_list(v)


class PathologyVerifierJudge(BaseModel):
    """pathological 四维口径：pathology_mechanism 0-4 / disease_relevance 0-3 / evidence_support 0-2 / format 0-1。"""
    model_config = ConfigDict(extra="ignore")

    pathology_mechanism_score: Optional[float] = None
    disease_relevance_score: Optional[float] = None
    evidence_support_score: Optional[float] = None
    format_score: Optional[float] = None
    errors: List[str] = []
    critical_errors: List[str] = []
    revision_advice: List[str] = []

    @field_validator("pathology_mechanism_score", mode="after")
    @classmethod
    def _clamp_mech(cls, v):
        return None if v is None else _clamp(v, 0, 4)

    @field_validator("disease_relevance_score", mode="after")
    @classmethod
    def _clamp_dis(cls, v):
        return None if v is None else _clamp(v, 0, 3)

    @field_validator("evidence_support_score", mode="after")
    @classmethod
    def _clamp_evi(cls, v):
        return None if v is None else _clamp(v, 0, 2)

    @field_validator("format_score", mode="after")
    @classmethod
    def _clamp_fmt(cls, v):
        return None if v is None else _clamp(v, 0, 1)

    @field_validator("errors", "critical_errors", "revision_advice", mode="before")
    @classmethod
    def _wrap_lists(cls, v):
        return _str_list(v)


class StrictVerifierJudge(VerifierJudge):
    """instructor 生成端约束专用：分数字段必填，缺失触发 instructor 自动重问（而非静默 None）。"""
    factuality_score: float
    grounding_score: float
    format_score: float
    consistency_score: float


class StrictPathologyVerifierJudge(PathologyVerifierJudge):
    """instructor 生成端约束专用（patho 口径）：分数字段必填，缺失触发自动重问。"""
    pathology_mechanism_score: float
    disease_relevance_score: float
    evidence_support_score: float
    format_score: float


# kind → (模型, 分数字段清单)
_MODELS: Dict[str, Tuple[Type[BaseModel], Tuple[str, ...]]] = {
    "four_dim": (
        VerifierJudge,
        ("factuality_score", "grounding_score", "format_score", "consistency_score"),
    ),
    "pathology": (
        PathologyVerifierJudge,
        ("pathology_mechanism_score", "disease_relevance_score", "evidence_support_score", "format_score"),
    ),
}


def validate_verifier_output(
    data, kind: str, score_default: float = 1.0
) -> Tuple[Optional[Dict], Optional[str]]:
    """校验 verifier JSON。返回 (cleaned_dict, None) 或 (None, 字段级错误消息)。

    score_default：分数字段缺失/None 时填充的默认分（各模块沿用既有 .get 默认口径）。
    """
    model, score_fields = _MODELS[kind]
    if not isinstance(data, dict) or not data:
        return None, "verifier JSON 为空或非对象"
    try:
        obj = model.model_validate(data)
    except ValidationError as e:
        detail = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        return None, f"verifier JSON 格式无效 [{detail}]"
    cleaned = obj.model_dump()
    for key in score_fields:
        if cleaned[key] is None:
            cleaned[key] = float(score_default)
    return cleaned, None
