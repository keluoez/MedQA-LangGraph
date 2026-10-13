import json
import statistics

def evaluate_quality(lines):
    total = 0
    failures = 0
    success = 0
    
    scores = []
    times = []
    failed_drugs = []
    ans_lengths = []
    
    quality_issues = {
        "hallucination": 0,
        "no_citations": 0,
        "low_quality_score": 0,
        "missing_sections": 0,
        "empty_answer": 0,
        "network_or_api_error": 0
    }
    
    for line in lines:
        total += 1
        record = json.loads(line)
        
        answer = record.get("answer", "")
        ans_lower = answer.lower()
        
        # Check for failure (robust to scientific text)
        is_failure = False
        error_indicators = ["generation failed", "api error", "connection error", "timeout", "rate limit exceeded", "max retries exceeded"]
        if any(indicator in ans_lower for indicator in error_indicators) or answer.strip() == "":
            if "api" in ans_lower or "connection" in ans_lower or "timeout" in ans_lower or "max retries" in ans_lower or "rate limit" in ans_lower:
                quality_issues["network_or_api_error"] += 1
            elif answer.strip() == "":
                quality_issues["empty_answer"] += 1
            is_failure = True
        
        score = record.get("quality_score", None)
        
        log = record.get("quality_log", "")
        log_str = json.dumps(log).lower() if isinstance(log, dict) else str(log).lower()
        if "hallucinat" in log_str or "unsupported" in log_str:
            quality_issues["hallucination"] += 1
            
        if "citation" in log_str and "missing" in log_str:
            quality_issues["no_citations"] += 1
            
        if isinstance(log, dict):
            errs = log.get("errors", []) + log.get("critical_errors", [])
            if errs:
                for err in errs:
                    if "network" in str(err).lower() or "api" in str(err).lower() or "connection" in str(err).lower():
                        quality_issues["network_or_api_error"] += 1
                        is_failure = True
        
        if is_failure or score is None or score == 0:
            is_failure = True
            failures += 1
            failed_drugs.append(f"{record.get('drug', 'Unknown')} - {record.get('protein', 'Unknown')}")
            continue
        
        ans_length = len(answer)
        success += 1
        scores.append(score)
        ans_lengths.append(ans_length)
        
        if record.get("elapsed_seconds"):
            times.append(record.get("elapsed_seconds"))
            
        if score < 0.8: # Assuming scale is 0.0-1.0
            quality_issues["low_quality_score"] += 1
        
        # Check sections
        sections = ["Core Conclusion", "Protein Information", "Condition", "Clinical Significance", "Evidence"]
        for sec in sections:
            if sec not in answer:
                quality_issues["missing_sections"] += 1
                break
                
    print(f"Total Records: {total}")
    print(f"Successful: {success} ({success/total*100:.2f}%)")
    print(f"Failures: {failures} ({failures/total*100:.2f}%)")
    
    if failures > 0:
        print(f"\nSample of Failed Entries:")
        for fd in failed_drugs[:10]:
            print(f"- {fd}")
    
    if success > 0:
        print("\nStatistics (Successful Records):")
        print(f"Average Quality Score: {sum(scores)/len(scores):.4f}")
        if len(scores) > 1:
            print(f"Median Quality Score: {statistics.median(scores):.4f}")
            print(f"Min Quality Score: {min(scores):.4f}")
            print(f"Max Quality Score: {max(scores):.4f}")
        
        if ans_lengths:
            print(f"Average Answer Length: {sum(ans_lengths)/len(ans_lengths):.0f} characters")
        
        if times:
            print(f"Average Time taken: {sum(times)/len(times):.2f}s")
    
    print("\nQuality Issue Counts (Including failures mapped to issues):")
    for k, v in quality_issues.items():
        print(f"- {k}: {v}")
        
if __name__ == '__main__':
    with open('output/langgraph_batch_output.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print("=== EVALUATING LATEST 2998 RECORDS ===")
        evaluate_quality(lines[-2998:])
