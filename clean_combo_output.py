import json, re
from pathlib import Path

def clean_meta_commentary(answer: str) -> str:
    """清除 Refiner 的元评论残迹"""
    # 常见残迹模式
    patterns = [
        r'\n+(?:The original response|Changes made|Note:|Corrections made:).*$',
        r'\n+(?:\d+\.\s+(?:Corrected|Verified|Added|Updated|Revised|Fixed|Changed|Ensured|Maintained)).*$',
    ]
    for p in patterns:
        answer = re.sub(p, '', answer, flags=re.DOTALL | re.IGNORECASE)
    return answer.strip()

def fix_uniprot_id(uid: str) -> str:
    """修复拼接的 UniProt ID，只保留第一个有效的"""
    if not uid:
        return uid
    # 如果包含 + 或 / 分隔符，只取第一个
    for sep in ['+', '/', ',', ';', ' ']:
        if sep in uid:
            parts = [p.strip() for p in uid.split(sep) if p.strip()]
            if parts:
                return parts[0]
    return uid

fpath = Path(r'complex_combination_therapy/output/batch_output.jsonl')
backup = fpath.with_suffix('.jsonl.prebak')
if backup.exists():
    backup.unlink()
fpath.rename(backup)
print(f"原始文件已备份: {backup.name}")

records = []
with open(backup, 'r', encoding='utf-8') as f:
    for l in f:
        if l.strip():
            try:
                records.append(json.loads(l))
            except:
                pass

# 统计
cleaned_meta = 0
cleaned_uid = 0
dropped_short = 0
written = 0

with open(fpath, 'w', encoding='utf-8') as out:
    for r in records:
        ans = r.get('answer', '')
        
        # 跳过过短记录
        if len(ans) < 300:
            dropped_short += 1
            continue
        
        # 清除元评论
        cleaned = clean_meta_commentary(ans)
        if cleaned != ans:
            cleaned_meta += 1
            r['answer'] = cleaned
        
        # 修复 UniProt ID
        uid = r.get('uniprot_id', '')
        fixed_uid = fix_uniprot_id(uid)
        if fixed_uid != uid:
            cleaned_uid += 1
            r['uniprot_id'] = fixed_uid
        
        out.write(json.dumps(r, ensure_ascii=False) + '\n')
        written += 1

print(f"\n清洗完成:")
print(f"  原始记录数: {len(records)}")
print(f"  清除元评论: {cleaned_meta} 条")
print(f"  修复 UniProt ID: {cleaned_uid} 条")
print(f"  删除过短记录: {dropped_short} 条")
print(f"  最终写入记录: {written} 条")
