import json
from collections import Counter

data = []
with open(r'basic_physiological_conditions\output\langgraph_batch_output.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

# Dedup: keep highest score per (protein, condition) pair
best = {}
for d in data:
    key = (d['protein'], d['condition'])
    if key not in best or d['quality_score'] > best[key]['quality_score']:
        best[key] = d
deduped = list(best.values())
print(f'After dedup: {len(deduped)}')

# ---- Red flag phrases in Core Conclusion ----
RED_FLAGS = [
    # Biological irrelevance
    'not directly exposed to',
    'not typically exposed to',
    'not directly related to the gastric',
    'not directly related to this condition',
    'not relevant in this context',
    'not the normal physiological context',
    'not their normal physiological',
    'not applicable in this context',
    'not directly relevant',
    'this condition is not physiological for',
    'does not directly impact',
    'does not directly affect',
    'not directly affect',
    'not directly related to',
    'is not directly related',
    'not related to gastric',
    # Denaturation of intracellular proteins (nonsensical)
    'denatured and inactivated',
    'denatured or inactivated',
    'denatured due to the extreme',
    'likely to be denatured',
    'likely denatured',
]

def get_conclusion(record):
    answer = record['answer']
    if '\u3010Core Conclusion\u3011' in answer:
        return answer.split('\u3010Core Conclusion\u3011')[1].split('\u3010')[0].strip()
    return answer[:600]

def is_bad_conclusion(record):
    text = get_conclusion(record).lower()
    return any(flag.lower() in text for flag in RED_FLAGS)

# ---- Apply filters ----
filtered = []
stats = Counter()

for d in deduped:
    # Filter 1: critical errors
    if d.get('quality_log', {}).get('critical_errors'):
        stats['critical_errors'] += 1
        continue
    # Filter 2: bad core conclusion
    if is_bad_conclusion(d):
        stats['bad_conclusion'] += 1
        continue
    # Filter 3: score threshold
    if d['quality_score'] < 8:
        stats['low_score'] += 1
        continue
    filtered.append(d)
    stats['passed'] += 1

print(f"\nFilter stats:")
for k, v in stats.items():
    print(f"  {k}: {v}")

# ---- Show gastric acid subset ----
gastric = [d for d in filtered if 'Gastric acid' in d['condition']]
print(f"\nGastric acid passed: {len(gastric)}")
for d in gastric[:15]:
    conc = get_conclusion(d).replace('\n', ' ')[:110]
    score = d['quality_score']
    protein = d['protein'][:48]
    print(f"  [{score}] {protein:48s} | {conc}")

# ---- Condition breakdown ----
print(f"\nCondition breakdown ({len(filtered)} total):")
cond_c = Counter(d['condition'] for d in filtered)
for cond, cnt in sorted(cond_c.items(), key=lambda x: -x[1]):
    print(f"  {cnt:4d}  {cond}")

# ---- Score distribution ----
scores = [d['quality_score'] for d in filtered]
sc = Counter(scores)
print(f"\nScore distribution:")
for s in sorted(sc.keys(), reverse=True):
    print(f"  {s}: {sc[s]}")
print(f"Mean score: {sum(scores)/len(scores):.2f}")

# ---- Save filtered output ----
out_path = r'basic_physiological_conditions\output\filtered_high_quality.jsonl'
with open(out_path, 'w', encoding='utf-8') as f:
    for d in filtered:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
print(f"\nSaved {len(filtered)} records to {out_path}")
