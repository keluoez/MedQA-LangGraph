"""录制脚本：跑一次真实（live）管线，把 SSE 事件轨迹落盘为 mock 场景。

用法（需要有效的 BIOREASON_API_KEY，见 .env.example）：
  python scripts/record_mock_runs.py --pair-json "{\"drug\":\"Erlotinib\",\"protein\":\"EGFR\",\"uniprot_id\":\"P00533\",\"gene\":\"EGFR\",\"chembl_id\":\"CHEMBL248\"}" --scenario-id bronze_to_gold --description "Bronze -> Gold iterative correction"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.adapters import stream_adapter
from api.adapters.pipeline_bridge import RECORDED_RUNS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-json", required=True, help="交互对 JSON 字符串")
    ap.add_argument("--scenario-id", required=True, help="场景 id（文件名）")
    ap.add_argument("--description", default="")
    ap.add_argument("--max-corrections", type=int, default=2)
    args = ap.parse_args()

    pair = json.loads(args.pair_json)
    print(f"Recording live run: {args.scenario_id} | pair={pair}")
    events = stream_adapter.collect_events(pair, max_corrections=args.max_corrections)

    if not events or events[-1].get("type") == "error":
        print("[FAILED] run ended with error:", events[-1] if events else "no events")
        sys.exit(1)

    total_s = events[-1].get("ts", 0)
    recorded = []
    for ev in events:
        out = dict(ev)
        out["t_ms"] = int(out.pop("ts", 0) * 1000)
        out.pop("mode", None)
        recorded.append(out)

    meta = recorded[0]
    final = next((e for e in reversed(recorded) if e["type"] == "final"), None)

    RECORDED_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RECORDED_RUNS_DIR / f"{args.scenario_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario_id": args.scenario_id,
                "description": args.description,
                "source": "recorded_live",
                "pair": pair,
                "question": meta.get("payload", {}).get("question", "")
                or meta.get("question", ""),
                "events": recorded,
                "final": final,
                "total_seconds": total_s,
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
    print(f"Saved -> {out_path} ({len(recorded)} events, {total_s:.1f}s run)")


if __name__ == "__main__":
    main()
