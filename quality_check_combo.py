import json, re
from collections import Counter, defaultdict
import statistics

fpath = r'complex_combination_therapy/output/batch_output.jsonl'
records = []
with open(fpath, 'r', encoding='utf-8') as f:
    for l in f:
        if l.strip():
            try:
                d = json.loads(l)
                if len(str(d.get('answer', ''))) > 10:
                    records.append(d)
            except:
                pass

print(f"=== 联合治疗数据集 深度质量分析 ===")
print(f"当前记录数: {len(records)}\n")

# 1. 格式完整性检查
REQUIRED_SECTIONS = ['【Core Conclusion】', '【Target Proteins】', '【Clinical Significance】', '【Evidence】']
section_missing = defaultdict(int)
missing_any = 0
for r in records:
    ans = r.get('answer', '')
    missing = [s for s in REQUIRED_SECTIONS if s not in ans]
    if missing:
        missing_any += 1
        for s in missing:
            section_missing[s] += 1

print(f"--- 1. 格式完整性 ---")
print(f"缺少至少一个必要章节: {missing_any}/{len(records)} ({missing_any/len(records)*100:.1f}%)")
for sec, cnt in sorted(section_missing.items(), key=lambda x: -x[1]):
    print(f"  缺少 {sec}: {cnt} 条")

# 2. PMID 引用检查
print(f"\n--- 2. 引用质量 ---")
no_pmid = 0
pmid_counts = []
for r in records:
    ans = r.get('answer', '')
    pmids = re.findall(r'PMID[:\s]*(\d{6,9})', ans, re.IGNORECASE)
    if not pmids:
        no_pmid += 1
    pmid_counts.append(len(pmids))

print(f"无 PMID 引用的记录: {no_pmid}/{len(records)} ({no_pmid/len(records)*100:.1f}%)")
print(f"平均 PMID 数量: {statistics.mean(pmid_counts):.1f}")
print(f"最多引用: {max(pmid_counts)}, 最少: {min(pmid_counts)}")

# 3. 答案长度分析
print(f"\n--- 3. 答案长度分布 ---")
lens = [len(r.get('answer', '')) for r in records]
print(f"平均: {statistics.mean(lens):.0f} chars")
print(f"中位数: {statistics.median(lens):.0f} chars")
very_short = sum(1 for l in lens if l < 500)
very_long = sum(1 for l in lens if l > 6000)
print(f"过短(<500字符): {very_short} 条 — 可能内容不充分")
print(f"过长(>6000字符): {very_long} 条 — 可能包含冗余内容")

# 4. UniProt ID 问题检查
print(f"\n--- 4. UniProt ID 质量 ---")
bad_uniprot = 0
uniprot_pattern = re.compile(r'^[A-Z][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$')
for r in records:
    uid = r.get('uniprot_id', '')
    # UniProt ID 标准格式6-10位
    if not uid or len(uid) < 6 or len(uid) > 10:
        bad_uniprot += 1

print(f"UniProt ID 格式异常: {bad_uniprot} 条")

# 5. 答案质量采样 - 仔细看3条典型数据
print(f"\n--- 5. 随机抽样精读 (3条) ---")
import random
random.seed(42)
samples = random.sample(records, min(3, len(records)))
for i, r in enumerate(samples):
    ans = r.get('answer', '')
    print(f"\n[样本 {i+1}]")
    print(f"  药物组合: {r.get('drugs','')}")
    print(f"  场景: {r.get('scene','')}")
    print(f"  靶蛋白: {r.get('targets',[])} | UniProt: {r.get('uniprot_id','')}")
    print(f"  答案长度: {len(ans)} chars")
    # 看核心结论部分
    cc_start = ans.find('【Core Conclusion】')
    cc_end = ans.find('【Target Proteins】')
    if cc_start != -1 and cc_end != -1:
        core = ans[cc_start+len('【Core Conclusion】'):cc_end].strip()
        print(f"  Core Conclusion: {core[:350]}")
    # 看证据部分
    ev_start = ans.rfind('【Evidence】')
    if ev_start != -1:
        ev = ans[ev_start:ev_start+300].strip()
        print(f"  Evidence: {ev[:200]}")

# 6. "残迹/伪文字"检查
print(f"\n--- 6. 内容质量异常检测 ---")
hallucination_flags = 0
for r in records:
    ans = r.get('answer', '')
    flags = []
    if re.search(r'\[CHEMISTRY\]|\[RESIDUE\]|\[NOMENCLATURE\]', ans):
        flags.append('verification_artifact')
    if re.search(r'As an AI|I cannot|I don\'t have', ans, re.IGNORECASE):
        flags.append('ai_refusal')
    if re.search(r'Note:|Changes made:|The original response', ans):
        flags.append('meta_commentary_leak')
    if flags:
        hallucination_flags += 1

print(f"包含验证残迹/AI声明/元评论的: {hallucination_flags} 条")

# 看几条有残迹的
count = 0
for r in records:
    ans = r.get('answer', '')
    if re.search(r'Note:|Changes made:|The original response', ans) and count < 2:
        note_pos = max(ans.find('Note:'), ans.find('Changes made:'), ans.find('The original response'))
        print(f"  示例残迹: ...{ans[note_pos:note_pos+150]}...")
        count += 1

print(f"\n=== 分析完成 ===")
