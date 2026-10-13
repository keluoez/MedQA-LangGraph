import json, re
from pathlib import Path

def check_truncation(fpath, label):
    records = []
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            for l in f:
                if l.strip():
                    try:
                        d = json.loads(l)
                        if len(str(d.get('answer', ''))) > 10:
                            records.append(d)
                    except:
                        pass
    except FileNotFoundError:
        print(f"[SKIP] 文件不存在: {fpath}")
        return

    print(f"\n{'='*60}")
    print(f"[{label}]  总记录: {len(records)}")

    # 截断检测指标
    truncated = []
    for r in records:
        ans = r.get('answer', '')
        issues = []

        # 1. 末尾缺少句号/完整标点（最后一个字符不是正常结束符）
        stripped = ans.strip()
        if stripped and stripped[-1] not in '.。!！?？"\'】)）':
            issues.append('末尾无结束标点')

        # 2. 最后一个必要章节缺失（Evidence通常是末节）
        if '【Evidence】' in ans or '[Evidence]' in ans:
            # Evidence 之后的内容是否至少有30个字符
            ev_pos = max(ans.rfind('【Evidence】'), ans.rfind('[Evidence]'))
            after_ev = ans[ev_pos + 12:].strip() if ev_pos != -1 else ''
            if len(after_ev) < 30:
                issues.append('Evidence节内容过短(<30chars)')
        else:
            issues.append('缺失Evidence节')

        # 3. 句子在句中被截断（以逗号/连接词结尾）
        last_sentence = stripped[-80:] if len(stripped) > 80 else stripped
        if re.search(r'(?:,|and|or|but|which|that|with|by|of|in|the|a|an)\s*$', last_sentence, re.IGNORECASE):
            issues.append('末尾疑似句中截断')

        # 4. 异常短答案
        if len(ans) < 500:
            issues.append(f'答案过短({len(ans)}chars)')

        if issues:
            truncated.append((r, issues))

    print(f"疑似截断/不完整记录: {len(truncated)} / {len(records)} ({len(truncated)/len(records)*100:.1f}%)")

    if truncated:
        print("\nTOP 5 截断样本:")
        for r, issues in truncated[:5]:
            ans = r.get('answer', '')
            drugs = r.get('drugs', r.get('condition', r.get('question', ''))[:50])
            print(f"  [{', '.join(issues)}]")
            print(f"  标识: {str(drugs)[:60]}")
            print(f"  答案末尾50字符: ...{ans.strip()[-80:]!r}")
            print()

    return len(truncated), len(records)

datasets = [
    (r'complex_combination_therapy/output/batch_output.jsonl', 'Combination Therapy'),
    (r'basic_physiological_conditions/output/langgraph_batch_output.jsonl', 'Physiological Conditions'),
    (r'basic_experimental_conditions/output/langgraph_batch_output.jsonl', 'Experimental Conditions'),
    (r'basic_pathological_environment/output/langgraph_batch_output.jsonl', 'Pathological Environment'),
]

total_trunc = 0
total_all = 0
for fpath, label in datasets:
    result = check_truncation(fpath, label)
    if result:
        total_trunc += result[0]
        total_all += result[1]

print(f"\n{'='*60}")
print(f"全部数据集汇总: 疑似截断 {total_trunc} / {total_all} ({total_trunc/total_all*100:.1f}%)")
