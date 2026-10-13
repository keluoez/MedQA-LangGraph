"""mock 回放器：按录制轨迹重放 SSE 事件，与 live 模式共用同一事件 schema。"""
import json
import time
from typing import Optional

from .pipeline_bridge import RECORDED_RUNS_DIR


def list_scenarios() -> list:
    """扫描已录制的运行轨迹，返回场景元信息。"""
    out = []
    if not RECORDED_RUNS_DIR.exists():
        return out
    for path in sorted(RECORDED_RUNS_DIR.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            out.append(
                {
                    "scenario_id": data.get("scenario_id", path.stem),
                    "description": data.get("description", ""),
                    "pair": data.get("pair", {}),
                    "iterations": data.get("final", {}).get("iterations", 1),
                    "duration_ms": max(
                        (e.get("t_ms", 0) for e in data.get("events", [])), default=0
                    ),
                    "source": data.get("source", "recorded"),
                }
            )
        except Exception:
            continue
    return out


def get_scenario(scenario_id: str) -> Optional[dict]:
    path = RECORDED_RUNS_DIR / f"{scenario_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def play(scenario_id: str, speed: float = 2.0):
    """生成器：按录制时间戳以 speed 倍速重放事件。

    事件结构与 stream_adapter.stream_events 输出完全一致，
    mode 字段标记为 "mock"，前端数据通路与 live 无差别。
    """
    data = get_scenario(scenario_id)
    if data is None:
        yield {"type": "error", "message": f"scenario not found: {scenario_id}", "ts": 0.0}
        return

    events = data.get("events", [])
    if not events:
        yield {"type": "error", "message": "empty scenario", "ts": 0.0}
        return

    pair = data.get("pair", {})
    yield {
        "type": "meta",
        "run_id": f"mock-{scenario_id}",
        "mode": "mock",
        "scenario_id": scenario_id,
        "pair": pair,
        "question": data.get("question", ""),
        "model": "replay of recorded run",
        "ts": 0.0,
    }

    prev_t = 0.0
    for ev in events:
        t = ev.get("t_ms", 0) / 1000.0
        wait = (t - prev_t) / max(speed, 0.1)
        if wait > 0:
            time.sleep(wait)
        prev_t = t
        out = dict(ev)
        out.pop("t_ms", None)
        out["ts"] = t
        out["mode"] = "mock"
        yield out
