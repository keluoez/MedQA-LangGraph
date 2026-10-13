# -*- coding: utf-8 -*-
"""M4-B2 历史病理数据质量分重算脚本（评分口径修复后的离线校正）。

背景（12 分越界 bug，M4-B2 已在 verifier 修复）：
- 旧口径 total = pathology_mechanism(未钳制，可达 5) + disease(≤3) + evidence(≤2) + det格式分(≤2)
  → 最高 12 分；实证历史输出 4988 条中 876 条 >10（11 分 780 条 / 12 分 96 条）。
- 新口径 total = min(4, pathology_mechanism) + disease + evidence + LLM格式分(0-1) → 上限 10 分。

历史记录的 quality_log.subscores 中保留了 LLM 原始四维（format_llm_raw 等），
因此可完全离线重算，无需重新调用 LLM 或任何网络请求。

用法（默认 dry-run，只打印统计不写任何文件）：
    python recalculate_scores.py
    python recalculate_scores.py --input output/langgraph_batch_output.jsonl
    python recalculate_scores.py --write   # 写入 <input>.recalculated.jsonl（不覆盖原文件）

写入时对每条记录的改动（均可审计，原值保留在 quality_log.score_recalc）：
- quality_score                       -> 新口径总分（≤10）
- quality_log.pathology_mechanism_score-> min(4.0, 旧值)（钳制）
- quality_log.format_score            -> subscores.format_llm_raw（LLM 0-1 口径）
- quality_log.subscores.pathology_mechanism_effective -> min(4.0, 旧值)

注意：历史记录未存 deterministic_checks 明细，故 quality_log 不补 "deterministic" 键
（该键仅新管线产出）；若需彻底按新口径重跑（含 det 明细与路由），须用新管线重新生成。
"""
import argparse
import collections
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "output" / "langgraph_batch_output.jsonl"


def recalc_one(record: dict) -> float:
    """按 M4-B2 新口径重算单条总分；字段缺失时抛 KeyError 由调用方计为 skip。"""
    q = record["quality_log"]
    sub = q["subscores"]
    mech = max(0.0, min(4.0, float(q["pathology_mechanism_score"])))
    disease = float(q["disease_relevance_score"])
    evidence = float(q["evidence_support_score"])
    fmt = float(sub["format_llm_raw"])
    return round(mech + disease + evidence + fmt, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="历史病理质量分离线重算（M4-B2 新口径，≤10）")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="输入 JSONL 文件")
    parser.add_argument("--write", action="store_true",
                        help="写入 <input>.recalculated.jsonl（默认 dry-run 只打印统计）")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"错误: 输入文件不存在 {in_path}")
        return

    records, skipped = [], 0
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if not (isinstance(rec.get("quality_log"), dict)
                        and isinstance(rec["quality_log"].get("subscores"), dict)
                        and "format_llm_raw" in rec["quality_log"]["subscores"]):
                    skipped += 1
                    continue
                records.append(rec)
            except Exception:
                skipped += 1

    old_scores, new_scores = [], []
    for rec in records:
        old = rec.get("quality_score")
        old_scores.append(round(float(old), 1) if isinstance(old, (int, float)) else None)
        new_scores.append(recalc_one(rec))

    valid_old = [s for s in old_scores if s is not None]
    deltas = collections.Counter(
        round(n - o, 1) for o, n in zip(old_scores, new_scores) if o is not None
    )
    print("=" * 64)
    print("M4-B2 历史病理质量分重算（离线，无 LLM/网络调用）")
    print("=" * 64)
    print(f"输入: {in_path}")
    print(f"可重算记录: {len(records)}  |  跳过(缺 quality_log/subscores): {skipped}")
    print(f">10 分记录:  修复前 {sum(1 for s in valid_old if s > 10)}  ->  修复后 {sum(1 for s in new_scores if s > 10)}")
    print(f"平均分:     修复前 {round(sum(valid_old) / len(valid_old), 3)}  ->  修复后 {round(sum(new_scores) / len(new_scores), 3)}")
    print(f"分差分布:   {dict(sorted(deltas.items()))}")
    print(f"新分数分布: {dict(sorted(collections.Counter(new_scores).items()))}")

    if not args.write:
        print("\n[dry-run] 未写入任何文件。确认无误后加 --write 生成 *.recalculated.jsonl")
        return

    out_path = in_path.with_name(in_path.stem + ".recalculated.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec, old, new in zip(records, old_scores, new_scores):
            q = rec["quality_log"]
            clamped = max(0.0, min(4.0, float(q["pathology_mechanism_score"])))
            q["pathology_mechanism_score"] = clamped
            q["format_score"] = float(q["subscores"]["format_llm_raw"])  # LLM 0-1 口径
            q["subscores"]["pathology_mechanism_effective"] = clamped
            q["score_recalc"] = {
                "old_quality_score": old,
                "caliber": "M4-B2: min(4,mech)+disease+evidence+llm_format",
            }
            rec["quality_score"] = new
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n已写入 {out_path}（原文件未改动）")


if __name__ == "__main__":
    main()
