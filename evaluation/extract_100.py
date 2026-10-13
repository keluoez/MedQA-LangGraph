import json
from pathlib import Path

def main():
    input_file = Path(r"c:\Users\Jiang\Desktop\run-bdi\basic_drug_interaction\output\langgraph_batch_output.jsonl")
    output_file = Path(r"c:\Users\Jiang\Desktop\run-bdi\evaluation\our_data_100_for_eval.json")
    
    selected = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get("quality_score", 0) >= 10.0:
                selected.append({
                    "question": item["question"],
                    "answer": item["answer"]
                })
            if len(selected) == 100:
                break
                
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
        
    print(f"Extracted {len(selected)} items to {output_file}")

if __name__ == "__main__":
    main()
