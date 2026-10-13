"""M4-B2（P4 抽取）：批量执行共享骨架 —— 断点续传、并发执行、失败审计。

原 4 模块的 run_batch 收敛为单一参数化实现，领域差异通过回调注入：
- process_one(pair) -> sample：单条样本执行（模块闭包捕获 app/max_corrections）；
- pair_id(record_or_pair) -> str：主键（同时用于断点续读历史输出行）；
- describe(record) -> str：日志展示（如 "Aspirin -> P23219"）；
- error_fields(pair) -> dict：失败记录附加的领域字段。

行为语义（M4-B1 P3 保持不变）：
- resume 只统计成功行（跳过 error/空 answer/坏 JSON），失败样本重启自动重试；
- 失败写入 <output>.failures.jsonl 审计文件，不阻断批次。
"""
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List


def run_one(app, pair: Dict[str, Any], max_corrections: int = 2) -> Dict[str, Any]:
    start = time.time()
    initial = {
        "pair": pair,
        "correction_count": 0,
        "max_corrections": max_corrections,
        "best_score": -1.0,
    }
    out = app.invoke(initial)
    final_sample = out.get("final_sample", {})
    final_sample["elapsed_seconds"] = round(time.time() - start, 2)
    return final_sample


def run_batch(
    pairs: List[Dict[str, Any]],
    output_file: Path,
    *,
    process_one: Callable[[Dict[str, Any]], Dict[str, Any]],
    pair_id: Callable[[Dict[str, Any]], str],
    describe: Callable[[Dict[str, Any]], str],
    error_fields: Callable[[Dict[str, Any]], Dict[str, Any]],
    max_workers: int = 3,
    batch_label: str = "items",
    summarize: bool = False,
) -> List[Dict[str, Any]]:
    out_path = Path(output_file)
    if out_path.suffix == ".json":
        out_path = out_path.with_suffix(".jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed_pairs = set()
    failures_path = out_path.with_name(out_path.stem + ".failures.jsonl")
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    s = json.loads(line)
                    # P3 修复：失败/无效记录不算已完成，重启后自动重试
                    if "error" in s or not s.get("answer"):
                        continue
                    completed_pairs.add(pair_id(s))
                except Exception:
                    pass

    filtered_pairs = []
    for p in pairs:
        pid = pair_id(p)
        if pid not in completed_pairs:
            filtered_pairs.append(p)
        else:
            print(f"Skipping already processed: {pid}", flush=True)

    if not filtered_pairs:
        print("All pairs in this batch have already been processed.", flush=True)
        return []

    output: List[Dict[str, Any]] = []
    times: List[float] = []
    file_lock = threading.Lock()

    def process_pair(pair):
        try:
            sample = process_one(pair)
            with file_lock:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            return sample
        except Exception as e:
            # P3 修复：失败样本写入独立 failures 文件，保留审计记录且不阻断重启重试
            err = {"error": str(e)}
            err.update(error_fields(pair))
            with file_lock:
                with open(failures_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(err, ensure_ascii=False) + "\n")
            return err

    print(f"Starting batch of {len(filtered_pairs)} {batch_label} with {max_workers} thread(s)...", flush=True)
    completed_count = 0
    total_count = len(filtered_pairs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {executor.submit(process_pair, p): p for p in filtered_pairs}
        for future in as_completed(future_to_pair):
            completed_count += 1
            res = future.result()
            label = describe(res)
            if "error" in res:
                print(f"[{completed_count}/{total_count}] FAILED {label}: {res['error']}", flush=True)
            else:
                times.append(res.get("elapsed_seconds", 0))
                print(
                    f"[{completed_count}/{total_count}] SUCCESS {label} | score={res.get('quality_score')} | time={res.get('elapsed_seconds')}s",
                    flush=True,
                )
                output.append(res)

    if summarize and times:
        print(
            f"avg_time={round(statistics.mean(times), 2)}s median_time={round(statistics.median(times), 2)}s",
            flush=True,
        )
    print(f"saved -> {out_path}", flush=True)
    return output
