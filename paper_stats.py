"""Generate detailed statistics for the paper."""
import json
import statistics
from collections import Counter

def main():
    with open('output/langgraph_batch_output.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = [json.loads(l) for l in lines if l.strip()]
    print(f"=== FULL DATASET STATISTICS ===")
    print(f"Total records: {len(data)}")
    
    # Filter out error records
    valid = [d for d in data if d.get('answer','').strip() and d.get('quality_score', 0) > 0]
    failed = [d for d in data if not d.get('answer','').strip() or d.get('quality_score', 0) == 0]
    print(f"Valid records: {len(valid)} ({len(valid)/len(data)*100:.1f}%)")
    print(f"Failed records: {len(failed)} ({len(failed)/len(data)*100:.1f}%)")
    
    # Score statistics
    scores = [d['quality_score'] for d in valid]
    print(f"\n--- Quality Score Statistics ---")
    print(f"Mean: {statistics.mean(scores):.4f}")
    print(f"Median: {statistics.median(scores):.4f}")
    print(f"Stdev: {statistics.stdev(scores):.4f}")
    print(f"Min: {min(scores):.4f}")
    print(f"Max: {max(scores):.4f}")
    
    # Score buckets
    print(f"\n--- Score Distribution ---")
    for threshold in [10, 9, 8, 7, 6, 5]:
        count = len([s for s in scores if s >= threshold])
        print(f"  >= {threshold}: {count} ({count/len(valid)*100:.1f}%)")
    
    # Quality tiers
    gold = [d for d in valid if d['quality_score'] >= 8.0]
    silver = [d for d in valid if 6.0 <= d['quality_score'] < 8.0]
    bronze = [d for d in valid if d['quality_score'] < 6.0]
    print(f"\n--- Quality Tiers ---")
    print(f"Gold (>=8.0):   {len(gold)} ({len(gold)/len(valid)*100:.1f}%)")
    print(f"Silver (6-8):   {len(silver)} ({len(silver)/len(valid)*100:.1f}%)")
    print(f"Bronze (<6.0):  {len(bronze)} ({len(bronze)/len(valid)*100:.1f}%)")
    
    # Route distribution
    routes = Counter(d.get('quality_log',{}).get('last_route','unknown') for d in valid)
    print(f"\n--- Route Distribution ---")
    for r, c in routes.most_common():
        print(f"  {r}: {c} ({c/len(valid)*100:.1f}%)")
    
    # Correction counts
    corrs = Counter(d.get('quality_log',{}).get('correction_count', 0) for d in valid)
    print(f"\n--- Correction Count Distribution ---")
    for c_val, count in sorted(corrs.items()):
        print(f"  {c_val} corrections: {count} ({count/len(valid)*100:.1f}%)")
    
    # Error analysis
    err_counter = Counter()
    crit_counter = Counter()
    for d in valid:
        log = d.get('quality_log', {})
        for e in log.get('errors', []):
            err_counter[e.split(':')[0]] += 1
        for e in log.get('critical_errors', []):
            crit_counter[e.split(':')[0]] += 1
    
    print(f"\n--- Error Types (top 10) ---")
    for e, c in err_counter.most_common(10):
        print(f"  {e}: {c}")
    
    print(f"\n--- Critical Error Types (top 10) ---")
    for e, c in crit_counter.most_common(10):
        print(f"  {e}: {c}")
    
    # Time statistics
    times = [d.get('elapsed_seconds', 0) for d in valid if d.get('elapsed_seconds')]
    if times:
        print(f"\n--- Processing Time ---")
        print(f"Mean: {statistics.mean(times):.2f}s")
        print(f"Median: {statistics.median(times):.2f}s")
        print(f"Min: {min(times):.2f}s")
        print(f"Max: {max(times):.2f}s")
    
    # Answer length
    lengths = [len(d.get('answer','')) for d in valid]
    print(f"\n--- Answer Length (chars) ---")
    print(f"Mean: {statistics.mean(lengths):.0f}")
    print(f"Median: {statistics.median(lengths):.0f}")
    
    # Subscore analysis
    fact_base = [d.get('quality_log',{}).get('subscores',{}).get('factuality_base', 0) for d in valid if d.get('quality_log',{}).get('subscores')]
    fact_eff = [d.get('quality_log',{}).get('subscores',{}).get('factuality_effective', 0) for d in valid if d.get('quality_log',{}).get('subscores')]
    grnd = [d.get('quality_log',{}).get('subscores',{}).get('grounding_llm_raw', 0) for d in valid if d.get('quality_log',{}).get('subscores')]
    fmt = [d.get('quality_log',{}).get('subscores',{}).get('format_llm_raw', 0) for d in valid if d.get('quality_log',{}).get('subscores')]
    cons = [d.get('quality_log',{}).get('subscores',{}).get('consistency_llm_raw', 0) for d in valid if d.get('quality_log',{}).get('subscores')]
    
    if fact_base:
        print(f"\n--- Subscore Means ---")
        print(f"Factuality base (0-4):      {statistics.mean(fact_base):.3f}")
        print(f"Factuality effective (0-4):  {statistics.mean(fact_eff):.3f}")
        print(f"Grounding (0-3):             {statistics.mean(grnd):.3f}")
        print(f"Format (0-2):                {statistics.mean(fmt):.3f}")
        print(f"Consistency (0-1):           {statistics.mean(cons):.3f}")

if __name__ == '__main__':
    main()
