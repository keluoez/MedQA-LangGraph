"""半合成 mock 生成器：零密钥构建可回放场景。

策略：generate/verify/final 事件取自真实产出数据集（output/*.jsonl 中的
真实样本与 quality_log）；collect 事件的证据包通过公开免费 API
（UniProt / ChEMBL / PubMed，无需 LLM 密钥）现场拉取，保证证据真实。

适用场景：单轮通过（first_pass）、证据不足声明（insufficient_evidence）。
多轮修正场景请用 record_mock_runs.py 做真实录制（需 BIOREASON_API_KEY）。

用法：
  python scripts/synthesize_mock_runs.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.adapters.pipeline_bridge import DOMAIN_FILES, fetch_evidence_apis
from api.adapters.stream_adapter import _slim_evidence, ROUTE_REASON
from api.routes.dataset import _parse_sections, _tier

# 相对时间戳（秒）——参照真实运行节奏（论文均值 33.8s/样本）
T_COLLECT = 8.0
T_GENERATE = 19.5
T_VERIFY = 24.0
T_FINAL = 25.0


def build_evidence(pair: dict) -> dict:
    """复用原管线检索函数，从公开 API 拉取真实证据包。"""
    api = fetch_evidence_apis()
    uniprot = api["fetch_uniprot_core"](pair["uniprot_id"])
    chembl = api["fetch_chembl"](pair.get("chembl_id", ""))
    mechanisms = api["fetch_chembl_drug_mechanisms"](
        pair.get("chembl_id", ""), target_hint=pair.get("gene", pair.get("protein", ""))
    )
    gene = pair.get("gene", "") or pair["protein"]
    query = (
        f"({pair['drug']}[Title/Abstract]) AND "
        f"(({pair['protein']}[Title/Abstract]) OR ({gene}[Title/Abstract])) AND "
        f"(mechanism[Title/Abstract] OR \"mode of action\"[Title/Abstract] "
        f"OR inhibit*[Title/Abstract] OR bind*[Title/Abstract])"
    )
    pmids = api["fetch_pubmed_pmids"](query, retmax=6)
    titles = api["fetch_pubmed_titles"](pmids)
    articles = []
    for pmid in pmids[:3]:
        xml = api["fetch_pubmed_article_xml"](pmid)
        meta = api["extract_pubmed_title_abstract_from_xml"](xml) if xml else {}
        articles.append(
            {
                "pmid": pmid,
                "pmcid": "",
                "title": meta.get("title", "") or titles.get(pmid, ""),
                "journal": meta.get("journal", ""),
                "year": meta.get("year", ""),
                "abstract": (meta.get("abstract", "") or "")[:1200],
                "fulltext_excerpt": "",
            }
        )
        time.sleep(0.4)  # NCBI 礼貌限速
    return {
        "uniprot": uniprot,
        "chembl": chembl,
        "chembl_mechanisms": mechanisms,
        "pubmed_pmids": pmids,
        "pubmed_titles": titles,
        "pubmed_articles": articles,
        "drugbank_id": pair.get("drugbank_id", ""),
    }


def load_records(domain: str) -> list:
    path = DOMAIN_FILES[domain]
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "error" in rec or not str(rec.get("answer", "")).strip():
                continue
            if not rec.get("quality_score"):
                continue
            out.append(rec)
    return out


def synthesize(scenario_id: str, description: str, rec: dict, pair: dict) -> dict:
    question = rec.get("question", "")
    qlog = rec.get("quality_log", {}) or {}
    score = rec.get("quality_score", 0)
    route = qlog.get("last_route", "pass")
    evidence = build_evidence(pair)

    events = [
        {"type": "node_output", "node": "collect", "iteration": 0, "t_ms": int(T_COLLECT * 1000),
         "payload": {"question": question, "evidence": _slim_evidence(evidence)}},
        {"type": "node_output", "node": "generate", "iteration": 1, "t_ms": int(T_GENERATE * 1000),
         "payload": {"answer": rec["answer"]}},
        {"type": "node_output", "node": "verify", "iteration": 1, "t_ms": int(T_VERIFY * 1000),
         "payload": {
             "score": score,
             "subscores": qlog.get("subscores", {}),
             "errors": qlog.get("errors", []),
             "critical_errors": qlog.get("critical_errors", []),
             "revision_advice": [],
         }},
        {"type": "route", "from": "verify", "to": route, "score": score,
         "reason": ROUTE_REASON.get(route, route), "iteration": 1, "t_ms": int(T_VERIFY * 1000)},
        {"type": "final", "score": score, "iterations": 1, "elapsed": T_FINAL,
         "t_ms": int(T_FINAL * 1000),
         "sample": {
             "question": question,
             "answer": rec["answer"],
             "sections": _parse_sections(rec["answer"]),
             "drug": rec.get("drug", ""),
             "protein": rec.get("protein", ""),
             "uniprot_id": rec.get("uniprot_id", ""),
             "quality_score": score,
             "tier": _tier(float(score)),
         }},
    ]

    return {
        "scenario_id": scenario_id,
        "description": description,
        "source": "synthesized_from_real_data",
        "pair": pair,
        "question": question,
        "events": events,
        "final": {"score": score, "iterations": 1},
        "total_seconds": T_FINAL,
    }


def find_first_pass(records: list) -> dict:
    for rec in records:
        qlog = rec.get("quality_log", {}) or {}
        if qlog.get("last_route") == "pass" and qlog.get("correction_count", 0) == 0:
            if float(rec.get("quality_score", 0)) >= 9.5:
                return rec
    raise RuntimeError("no first_pass sample found")


def find_insufficient_evidence(records: list) -> dict:
    for rec in records:
        ans = rec.get("answer", "")
        if "limited in the provided package" in ans and float(rec.get("quality_score", 0)) >= 6.0:
            return rec
    raise RuntimeError("no insufficient_evidence sample found")


def rec_to_pair(rec: dict) -> dict:
    return {
        "drug": rec.get("drug", ""),
        "protein": rec.get("protein", ""),
        "uniprot_id": rec.get("uniprot_id", ""),
        "gene": rec.get("protein", ""),
    }


def main():
    from api.adapters.pipeline_bridge import RECORDED_RUNS_DIR

    records = load_records("drug")
    print(f"loaded {len(records)} drug records")

    plan = [
        ("first_pass", "One-pass Gold generation (Imatinib-BCR-ABL style)", find_first_pass),
        ("insufficient_evidence", "Explicit uncertainty statement when evidence is limited", find_insufficient_evidence),
    ]
    RECORDED_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for scenario_id, desc, finder in plan:
        rec = finder(records)
        pair = rec_to_pair(rec)
        print(f"synthesizing {scenario_id}: {rec.get('drug')} -> {rec.get('protein')} (score {rec.get('quality_score')})")
        data = synthesize(scenario_id, desc, rec, pair)
        out_path = RECORDED_RUNS_DIR / f"{scenario_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
