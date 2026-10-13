"""SSE 事件协议 + live 模式适配器。

事件 schema（前后端契约，mock 回放与 live 完全一致）：
  meta        {run_id, mode, pair, question, model, ts}
  node_output {node, iteration, ts, payload}
              node ∈ collect | generate | verify | refine | regenerate
  route       {from, to, score, reason, iteration, ts}
              to ∈ pass | refine | regenerate | stop
  final       {score, iterations, elapsed, sample, ts}
  error       {message, ts}

实现说明：LangGraph stream_mode="updates" 在节点执行完成后才 yield 增量，
因此事件的 ts 表示"节点完成时刻"；前端由上一事件点亮下一节点 running 态，
动画节奏不受影响。
"""
import time
import uuid

from .pipeline_bridge import build_graph

ROUTE_REASON = {
    "pass": "score >= 8.0 and no critical errors",
    "refine": "6.0 <= score < 8.0, minor format issues",
    "regenerate": "critical errors or score < 6.0",
    "stop": "correction budget exhausted, output best answer",
}


def _slim_evidence(evidence: dict) -> dict:
    """精简证据包，保留前端展示所需字段。"""
    ev = evidence or {}
    articles = []
    for art in ev.get("pubmed_articles", []) or []:
        articles.append(
            {
                "pmid": art.get("pmid", ""),
                "title": art.get("title", ""),
                "journal": art.get("journal", ""),
                "year": art.get("year", ""),
                "abstract": (art.get("abstract") or "")[:1200],
            }
        )
    return {
        "uniprot": ev.get("uniprot", {}),
        "chembl": {"pref_name": (ev.get("chembl") or {}).get("pref_name", "")},
        "chembl_mechanisms": [
            {
                "action_type": m.get("action_type", ""),
                "target_name": m.get("target_name", ""),
                "mechanism_of_action": m.get("mechanism_of_action", ""),
                "reference": m.get("reference", ""),
            }
            for m in (ev.get("chembl_mechanisms", []) or [])[:4]
        ],
        "pubmed_pmids": ev.get("pubmed_pmids", []),
        "pubmed_titles": ev.get("pubmed_titles", {}),
        "pubmed_articles": articles,
        "drugbank_id": ev.get("drugbank_id", ""),
    }


def stream_events(pair: dict, max_corrections: int = 2):
    """live 模式：执行真实 LangGraph 管线，逐节点 yield 事件 dict。"""
    app = build_graph()
    run_id = uuid.uuid4().hex[:12]
    initial = {
        "pair": pair,
        "correction_count": 0,
        "max_corrections": max_corrections,
        "best_score": -1.0,
    }
    t0 = time.time()
    iter_state = 0

    yield {
        "type": "meta",
        "run_id": run_id,
        "mode": "live",
        "pair": pair,
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "ts": 0.0,
    }

    try:
        for chunk in app.stream(initial, stream_mode="updates"):
            ts = round(time.time() - t0, 2)
            for node, delta in chunk.items():
                delta = delta or {}
                if node == "collect":
                    iter_state = 0
                    yield {
                        "type": "node_output",
                        "node": "collect",
                        "iteration": 0,
                        "ts": ts,
                        "payload": {
                            "question": delta.get("question", ""),
                            "evidence": _slim_evidence(delta.get("evidence", {})),
                        },
                    }
                elif node == "generate":
                    iter_state = 1
                    yield {
                        "type": "node_output",
                        "node": "generate",
                        "iteration": 1,
                        "ts": ts,
                        "payload": {"answer": delta.get("answer", "")},
                    }
                elif node == "verify":
                    verification = delta.get("verification", {}) or {}
                    route = delta.get("route", "stop")
                    score = verification.get("score", 0)
                    yield {
                        "type": "node_output",
                        "node": "verify",
                        "iteration": iter_state,
                        "ts": ts,
                        "payload": {
                            "score": score,
                            "subscores": verification.get("subscores", {}),
                            "errors": verification.get("errors", []),
                            "critical_errors": verification.get("critical_errors", []),
                            "revision_advice": verification.get("revision_advice", []),
                        },
                    }
                    yield {
                        "type": "route",
                        "from": "verify",
                        "to": route,
                        "score": score,
                        "reason": ROUTE_REASON.get(route, route),
                        "iteration": iter_state,
                        "ts": ts,
                    }
                elif node in ("refine", "regenerate"):
                    iter_state = delta.get("correction_count", iter_state) + 1
                    yield {
                        "type": "node_output",
                        "node": node,
                        "iteration": iter_state,
                        "ts": ts,
                        "payload": {
                            "answer": delta.get("answer", ""),
                            "correction_count": delta.get("correction_count", 0),
                        },
                    }
                elif node == "finalize":
                    sample = delta.get("final_sample", {}) or {}
                    yield {
                        "type": "final",
                        "score": sample.get("quality_score", 0),
                        "iterations": iter_state,
                        "elapsed": ts,
                        "sample": sample,
                        "ts": ts,
                    }
    except Exception as e:  # live 模式失败兜底（api 层整体重试策略由调用方决定）
        yield {
            "type": "error",
            "message": f"{type(e).__name__}: {e}",
            "ts": round(time.time() - t0, 2),
        }


def collect_events(pair: dict, max_corrections: int = 2) -> list:
    """同步收集整个运行的事件序列（录制脚本用）。"""
    return list(stream_events(pair, max_corrections=max_corrections))
