"""数据集样例浏览接口。

读取原有管线产出的 output/*.jsonl，构建内存索引，
支持领域/质量分层筛选、关键词搜索、分页。
P3 兜底：error / 空 answer 记录在索引构建时即被过滤。
"""
import json
import re
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..adapters.pipeline_bridge import DOMAIN_FILES, DOMAIN_LABELS

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

SECTION_RE = re.compile(r"【([^】]+)】")

_index: Optional[list] = None
_lock = threading.Lock()


def _tier(score: float) -> str:
    if score >= 8.0:
        return "gold"
    if score >= 6.0:
        return "silver"
    return "bronze"


def _parse_sections(answer: str) -> list:
    """把【Tag】结构化回答拆成有序段落列表。"""
    matches = list(SECTION_RE.finditer(answer))
    if not matches:
        return []
    out = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(answer)
        text = answer[start:end].strip()
        if text:
            out.append({"title": m.group(1), "text": text})
    return out


def _iter_valid_records(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            # P3 兜底：跳过失败样本与无效记录
            if "error" in rec:
                continue
            if not str(rec.get("answer", "")).strip():
                continue
            try:
                score = float(rec.get("quality_score") or 0)
            except (TypeError, ValueError):
                continue
            if score <= 0:
                continue
            yield rec


def _build_index() -> list:
    items = []
    for domain, path in DOMAIN_FILES.items():
        if not path.exists():
            continue
        for i, rec in enumerate(_iter_valid_records(path)):
            qlog = rec.get("quality_log", {}) or {}
            items.append(
                {
                    "id": f"{domain}-{i:05d}",
                    "domain": domain,
                    "domain_label": DOMAIN_LABELS[domain],
                    "question": rec.get("question", ""),
                    "answer": rec["answer"],
                    "sections": _parse_sections(rec["answer"]),
                    "drug": rec.get("drug", ""),
                    "protein": rec.get("protein", ""),
                    "uniprot_id": rec.get("uniprot_id", ""),
                    "quality_score": round(float(rec["quality_score"]), 2),
                    "tier": _tier(float(rec["quality_score"])),
                    "correction_count": qlog.get("correction_count", 0),
                    "last_route": qlog.get("last_route", ""),
                    "subscores": qlog.get("subscores", {}),
                    "errors": qlog.get("errors", []),
                    "critical_errors": qlog.get("critical_errors", []),
                    "elapsed_seconds": rec.get("elapsed_seconds", 0),
                }
            )
    return items


def get_index() -> list:
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                _index = _build_index()
    return _index


def _stats(items: list) -> dict:
    n = len(items)
    if n == 0:
        return {"total": 0}
    scores = [x["quality_score"] for x in items]
    tiers = {"gold": 0, "silver": 0, "bronze": 0}
    for x in items:
        tiers[x["tier"]] += 1
    return {
        "total": n,
        "mean_score": round(sum(scores) / n, 2),
        "gold": tiers["gold"],
        "silver": tiers["silver"],
        "bronze": tiers["bronze"],
        "gold_pct": round(tiers["gold"] / n * 100, 1),
        "silver_pct": round(tiers["silver"] / n * 100, 1),
        "bronze_pct": round(tiers["bronze"] / n * 100, 1),
    }


@router.get("/samples")
def list_samples(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    domain: Optional[str] = Query(None, description="drug|experimental|pathological|physiological|combo"),
    tier: Optional[str] = Query(None, description="gold|silver|bronze"),
    q: Optional[str] = Query(None, description="搜索 drug/protein/question"),
    sort: str = Query("score", description="score|original"),
):
    idx = get_index()
    items = idx
    if domain:
        items = [x for x in items if x["domain"] == domain]
    if tier:
        items = [x for x in items if x["tier"] == tier]
    if q:
        ql = q.lower()
        items = [
            x
            for x in items
            if ql in x["drug"].lower()
            or ql in x["protein"].lower()
            or ql in x["question"].lower()
        ]
    if sort == "score":
        items = sorted(items, key=lambda x: x["quality_score"], reverse=True)
    total = len(items)
    start = (page - 1) * page_size
    page_items = items[start : start + page_size]
    # 列表页只给摘要，answer/sections 留给详情接口
    summaries = [
        {k: v for k, v in x.items() if k not in ("answer", "sections")}
        for x in page_items
    ]
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": summaries,
    }


@router.get("/samples/{sample_id}")
def get_sample(sample_id: str):
    idx = get_index()
    for x in idx:
        if x["id"] == sample_id:
            return x
    raise HTTPException(status_code=404, detail=f"sample not found: {sample_id}")


@router.get("/stats")
def dataset_stats():
    idx = get_index()
    per_domain = {}
    for domain, label in DOMAIN_LABELS.items():
        sub = [x for x in idx if x["domain"] == domain]
        s = _stats(sub)
        s["label"] = label
        per_domain[domain] = s
    return {"overall": _stats(idx), "by_domain": per_domain}
