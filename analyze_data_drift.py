import json
from collections import Counter
import re

def get_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in words if w not in {'and', 'the', 'for', 'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'receptor', 'subunit', 'protein', 'kinase', 'channel'}]

with open('drug_target_pairs.json', 'r', encoding='utf-8') as f:
    pairs = json.load(f)

p1 = pairs[:5000]
p2 = pairs[5000:8000]

print("=== GROUP 1 (0-5000) PROTEIN KEYWORDS ===")
c1 = Counter()
for p in p1:
    c1.update(get_keywords(p['protein']))
print(c1.most_common(20))

print("\n=== GROUP 2 (5000-8000) PROTEIN KEYWORDS ===")
c2 = Counter()
for p in p2:
    c2.update(get_keywords(p['protein']))
print(c2.most_common(20))

print("\n=== SAMPLE NAMES 0-5000 ===")
for p in p1[:10]:
    print("-", p['protein'])

print("\n=== SAMPLE NAMES 5000-8000 ===")
for p in p2[:10]:
    print("-", p['protein'])
