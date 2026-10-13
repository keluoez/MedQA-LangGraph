import json
from collections import Counter

def check_duplicates(fpath, key_fields):
    records = []
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                records.append(json.loads(line))
            except:
                pass

    keys = []
    for r in records:
        k = "::".join(str(r.get(f, "")) for f in key_fields)
        keys.append(k)

    total = len(keys)
    counter = Counter(keys)
    dups = {k: v for k, v in counter.items() if v > 1}
    dup_records = sum(v - 1 for v in dups.values())

    print(f"\n{'='*60}")
    print(f"文件: {fpath}")
    print(f"总行数: {total}")
    print(f"重复的唯一Key数: {len(dups)}")
    print(f"其中冗余记录数 (重复次数-1之和): {dup_records}")
    if dups:
        print(f"\n TOP 5 重复最多的条目:")
        for k, cnt in sorted(dups.items(), key=lambda x: -x[1])[:5]:
            print(f"  [{cnt}x] {k[:100]}")

check_duplicates(
    r"c:\Users\Jiang\Desktop\run-bdi\complex_combination_therapy\output\batch_output.jsonl",
    key_fields=["drugs", "uniprot_id"]
)

check_duplicates(
    r"c:\Users\Jiang\Desktop\run-bdi\basic_physiological_conditions\output\langgraph_batch_output.jsonl",
    key_fields=["condition", "protein"]
)

check_duplicates(
    r"c:\Users\Jiang\Desktop\run-bdi\basic_experimental_conditions\output\langgraph_batch_output.jsonl",
    key_fields=["condition", "protein"]
)

check_duplicates(
    r"c:\Users\Jiang\Desktop\run-bdi\basic_pathological_environment\output\langgraph_batch_output.jsonl",
    key_fields=["condition", "protein"]
)
