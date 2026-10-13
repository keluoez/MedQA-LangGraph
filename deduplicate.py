import json
from pathlib import Path

def deduplicate(fpath, key_fields, dry_run=False):
    fpath = Path(fpath)
    if not fpath.exists():
        print(f"[SKIP] 文件不存在: {fpath}")
        return

    records = []
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                records.append(json.loads(line))
            except:
                pass

    seen = set()
    unique = []
    for r in records:
        k = "::".join(str(r.get(f, "")) for f in key_fields)
        if k not in seen:
            seen.add(k)
            unique.append(r)

    removed = len(records) - len(unique)
    print(f"\n{'='*60}")
    print(f"文件: {fpath.name}")
    print(f"原始记录数: {len(records)}")
    print(f"去重后记录数: {len(unique)}")
    print(f"删除冗余记录: {removed} 条")

    if not dry_run:
        # 备份原文件（如已存在则覆盖旧备份）
        backup = fpath.with_suffix('.jsonl.bak')
        if backup.exists():
            backup.unlink()
        fpath.rename(backup)
        print(f"已备份原文件到: {backup.name}")

        with open(fpath, 'w', encoding='utf-8') as f:
            for r in unique:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"[DONE] 去重完成，写入: {fpath.name}")
    else:
        print(f"[DRY RUN] 不做实际修改")

# ---- 正式去重（将 dry_run=True 改为 False 执行真正写入）----
deduplicate(
    r"c:\Users\Jiang\Desktop\run-bdi\basic_physiological_conditions\output\langgraph_batch_output.jsonl",
    key_fields=["condition", "protein"],
    dry_run=False
)

deduplicate(
    r"c:\Users\Jiang\Desktop\run-bdi\basic_experimental_conditions\output\langgraph_batch_output.jsonl",
    key_fields=["condition", "protein"],
    dry_run=False
)

deduplicate(
    r"c:\Users\Jiang\Desktop\run-bdi\basic_pathological_environment\output\langgraph_batch_output.jsonl",
    key_fields=["condition", "protein"],
    dry_run=False
)

print("\n\n全部去重完成！原始文件已备份为 .jsonl.bak")
