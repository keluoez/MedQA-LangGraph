import sys

with open('langgraph_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('pair["pathology"]', 'pair.get("pathology", pair.get("disease", ""))')
content = content.replace("pair['pathology']", "pair.get('pathology', pair.get('disease', ''))")

with open('langgraph_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)
