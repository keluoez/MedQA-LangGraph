# -*- coding: utf-8 -*-
"""
Ablation Variant 3 — w/o Correction Loop
Full Collector + Responder + ONE Verifier pass.
No Refiner, no Regenerator — the verifier scores but never triggers corrections.
"""

import json, os, sys, re, time, requests, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from experiments.shared.api_config import NVIDIA_API_KEY, NVIDIA_BASE_URL, GENERATION_MODEL

client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)


def _llm(system, user, temp=0.5, max_tok=2000, max_retry=5):
    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": user}],
                temperature=temp, max_tokens=max_tok, timeout=90)
            return resp.choices[0].message.content or ""
        except Exception as e:
            err = str(e)
            wait = (2 ** attempt) * 15  # 15, 30, 60, 120, 240s
            if attempt < max_retry - 1:
                time.sleep(wait)
            else:
                raise
    return ""


def collector(drug, protein):
    try:
        query = (f"{drug}[Title/Abstract] AND {protein}[Title/Abstract] AND "
                 f"(mechanism OR inhibit OR bind OR activate)[Title/Abstract]")
        r  = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                          params={"db":"pubmed","term":query,"retmax":5,"retmode":"json"},
                          timeout=10)
        pmids = r.json().get("esearchresult",{}).get("idlist",[])
        if pmids:
            fr = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                              params={"db":"pubmed","id":",".join(pmids),
                                      "rettype":"abstract","retmode":"text"}, timeout=15)
            time.sleep(0.4)
            return pmids, fr.text[:3000]
        return [], "No results."
    except Exception as e:
        return [], f"Error: {e}"


RESPONDER_SYS = """Output STRICTLY in 5 sections:
【Core Conclusion】【Protein Information】【Condition】【Clinical Significance】【Evidence】
Never invent PMIDs or residue numbers not in the evidence."""

VERIFIER_SYS = """Score this biomedical QA. Return ONLY JSON:
{"factuality_score":0-4,"grounding_score":0-3,"format_score":0-2,"consistency_score":0-1,"errors":[]}"""


def generate(item):
    question = item["question"]
    m = re.match(r"What is the mechanism by which (.+?) affects (.+?)\??$", question, re.IGNORECASE)
    drug    = m.group(1).strip() if m else ""
    protein = m.group(2).strip() if m else ""

    pmids, abstracts = collector(drug, protein)
    evidence = f"PMIDs: {pmids}\nAbstracts:\n{abstracts}"

    answer = _llm(RESPONDER_SYS, f"Evidence:\n{evidence}\n\nQuestion: {question}")

    # ONE verification pass — no correction triggered
    ver_raw = _llm(VERIFIER_SYS,
                   f"Drug: {drug}, Protein: {protein}\nAnswer:\n{answer}",
                   temp=0.1, max_tok=300)
    try:
        jm = re.search(r'\{[\s\S]+\}', ver_raw)
        v  = json.loads(jm.group()) if jm else {}
    except:
        v = {}
    score = (float(v.get("factuality_score",0)) + float(v.get("grounding_score",0))
             + float(v.get("format_score",0))   + float(v.get("consistency_score",0)))

    return {"question": question, "answer": answer,
            "variant": "wo_correction_loop",
            "verifier_score": round(score, 2),
            "pmids_allowed": pmids}


def run(input_file, output_file, workers=2):
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    # Resume: skip already-done items
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    done_questions = set()
    results = []
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            results = json.load(f)
            done_questions = {r["question"][:80] for r in results}
    todo = [it for it in data if it["question"][:80] not in done_questions]
    print(f"[w/o Correction Loop] {len(todo)} remaining / {len(data)} total")
    if not todo:
        print("All done!"); return results

    lock = threading.Lock()
    done_count = [len(results)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(generate, it): it for it in todo}
        for fut in as_completed(futures):
            r = fut.result()
            with lock:
                results.append(r)
                done_count[0] += 1
                # Incremental save after every result
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [{done_count[0]}/{len(data)}]", flush=True)
    print(f"Saved → {output_file} ({len(results)} results)")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   default="../00_sample_200/ablation_200_samples.json")
    ap.add_argument("--output",  default="output/wo_correction_loop_results.json")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    run(args.input, args.output, args.workers)
