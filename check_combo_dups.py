import json
from collections import Counter

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

# 统计问题重复情况
questions = [r.get('question', '') for r in records]
q_counter = Counter(questions)
dup_questions = {q: c for q, c in q_counter.items() if c > 1}

print(f'总记录数: {len(records)}')
print(f'唯一问题数: {len(set(questions))}')
print(f'重复的问题数: {len(dup_questions)}')

print(f'\nTOP 5 重复最多的问题:')
for q, cnt in sorted(dup_questions.items(), key=lambda x: -x[1])[:5]:
    print(f'  [{cnt}x] {q[:90]}')

# 检查同一问题下不同 uniprot_id 的情况
if dup_questions:
    top_q = sorted(dup_questions.items(), key=lambda x: -x[1])[0][0]
    same_q = [r for r in records if r.get('question') == top_q]
    print(f'\n--- 同一问题下的不同 uniprot_id ---')
    for r in same_q[:6]:
        uid = r.get('uniprot_id', 'N/A')
        tgt = r.get('targets', [])
        print(f'  uniprot={uid}  targets={tgt}')

# 检查源数据文件
print('\n--- 源数据文件重复检查 ---')
src = json.load(open(r'complex_combination_therapy/combination_therapy_pairs.json', encoding='utf-8'))
src_questions = set()
src_q_dup = 0
seen = set()
for item in src:
    drugs = item.get('drug_combination', item.get('drugs', ''))
    scene = item.get('therapeutic_area', item.get('scene', ''))
    q = f"What is the synergistic mechanism of {drugs} in {scene}?"
    if q in seen:
        src_q_dup += 1
    seen.add(q)
    src_questions.add(q)

print(f'源文件总记录: {len(src)}')
print(f'唯一问题数: {len(src_questions)}')
print(f'同问题重复对数: {src_q_dup}')
