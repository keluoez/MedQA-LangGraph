import json
data = []
with open('c:/Users/Jiang/Desktop/run-bdi/basic_physiological_conditions/output/langgraph_batch_output.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10: break
        data.append(json.loads(line))
for d in data:
    score = d.get('quality_score')
    miss = d.get('missing_sections', [])
    unrec = d.get('unrecognized_pmids', [])
    mets = d.get('metrics', {})
    print(f"Score: {score} | Miss: {miss} | Unrec: {unrec} | Mets: {mets}")
