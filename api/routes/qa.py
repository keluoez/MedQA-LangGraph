"""问答流水线接口：SSE 流式返回各 Agent 独立输出与时序事件。"""
import json
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..adapters import mock_player, stream_adapter
from ..adapters.pipeline_bridge import PAIRS_FILE

router = APIRouter(prefix="/api/qa", tags=["qa"])

_pairs_cache: Optional[list] = None
_pairs_lock = threading.Lock()


class RunRequest(BaseModel):
    mode: str = Field("mock", description="mock | live")
    scenario_id: Optional[str] = Field(None, description="mock 模式：预录场景 id")
    pair: Optional[dict] = Field(None, description="live 模式：药物-蛋白交互对")
    speed: float = Field(2.0, ge=0.1, le=16.0, description="mock 回放倍速")
    max_corrections: int = Field(2, ge=1, le=4)


def _sse(event: dict) -> str:
    return f"event: {event.get('type', 'message')}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/run")
def run_qa(req: RunRequest):
    if req.mode == "mock":
        scenario_id = req.scenario_id
        if not scenario_id:
            scenarios = mock_player.list_scenarios()
            if not scenarios:
                raise HTTPException(503, "no recorded scenarios; run scripts/record_mock_runs.py first")
            scenario_id = scenarios[0]["scenario_id"]
        gen = mock_player.play(scenario_id, speed=req.speed)
    elif req.mode == "live":
        if not req.pair or not req.pair.get("drug") or not req.pair.get("protein"):
            raise HTTPException(400, "live mode requires pair with drug and protein")
        gen = stream_adapter.stream_events(req.pair, max_corrections=req.max_corrections)
    else:
        raise HTTPException(400, f"invalid mode: {req.mode}")

    def event_stream():
        for event in gen:
            yield _sse(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _load_pairs() -> list:
    global _pairs_cache
    if _pairs_cache is None:
        with _pairs_lock:
            if _pairs_cache is None:
                with open(PAIRS_FILE, encoding="utf-8") as f:
                    _pairs_cache = json.load(f)
    return _pairs_cache


@router.get("/pairs")
def list_pairs(
    limit: int = 50,
    offset: int = 0,
    q: Optional[str] = None,
):
    pairs = _load_pairs()
    if q:
        ql = q.lower()
        pairs = [p for p in pairs if ql in p.get("drug", "").lower() or ql in p.get("protein", "").lower()]
    total = len(pairs)
    return {"total": total, "items": pairs[offset : offset + limit]}


@router.get("/scenarios")
def list_scenarios():
    return {"scenarios": mock_player.list_scenarios()}
