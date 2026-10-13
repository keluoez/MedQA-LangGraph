import json

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

# 找出同一个问题的不同记录，比较答案差异
from collections import defaultdict
by_question = defaultdict(list)
for r in records:
    by_question[r.get('question', '')].append(r)

# 找一个有多条记录的问题
for q, recs in by_question.items():
    if len(recs) >= 3:
        print(f'问题: {q[:80]}')
        print(f'记录数: {len(recs)}')
        print()
        for i, r in enumerate(recs[:4]):
            uid = r.get('uniprot_id', '')
            tgt = r.get('targets', [])
            ans = r.get('answer', '')
            # 只看Core Conclusion部分
            core_start = ans.find('[Core Conclusion]') if '[Core Conclusion]' in ans else ans.find('Core Conclusion')
            core_end = ans.find('[Target Proteins]') if '[Target Proteins]' in ans else ans.find('Target Proteins')
            if core_start == -1:
                core_snippet = ans[:300]
            else:
                end = core_end if core_end > core_start else core_start + 300
                core_snippet = ans[core_start:end]
            
            print(f'--- 第{i+1}条 | uniprot={uid} | target={tgt} ---')
            print(core_snippet[:400])
            print()
        break
