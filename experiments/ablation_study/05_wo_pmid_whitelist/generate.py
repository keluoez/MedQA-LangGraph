# -*- coding: utf-8 -*-
"""
Ablation Variant 5 — w/o PMID Whitelist  (NEW — not in the original paper)
Full pipeline EXCEPT the deterministic PMID whitelist check.
The Verifier uses ONLY semantic LLM scoring — no hard PMID cross-reference.
This isolates the contribution of deterministic verification.
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
            wait = (2 ** attempt) * 15
            if attempt < max_retry - 1:
                time.sleep(wait)
            else:
                raise
    return ""


def collector(drug, protein):
    try:
        query = (f"{drug}[Title/Abstract] AND {protein}[Title/Abstract] AND "
                 f"(mechanism OR inhibit OR bind OR activate)[Title/Abstract]")
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                         params={"db":"pubmed","term":query,"retmax":5,"retmode":"json"}, timeout=10)
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
Never invent residue numbers not in the evidence."""

# ── KEY DIFFERENCE: Verifier has NO PMID whitelist enforcement ────────────────
# It can only check factuality and format via LLM judgment.
VERIFIER_NO_WHITELIST_SYS = """You are a biomedical fact-checker.
Score this QA pair on scientific quality using ONLY your semantic judgment.
Do NOT check whether PMIDs are valid — ignore citation authenticity.
Return ONLY JSON:
{"factuality_score":0-4,"grounding_score":0-3,"format_score":0-2,"consistency_score":0-1,"errors":[]}
Errors should focus on factual inaccuracies or logical issues only."""

REFINER_SYS = "Fix only the listed scientific errors. Output the full corrected answer only."
REGENERATOR_SYS = """You are a biomedical expert. The previous answer had critical errors.
Generate a completely new, accurate answer using the provided evidence.
Output in 5 sections: 【Core Conclusion】【Protein Information】【Condition】【Clinical Significance】【Evidence】"""


def generate(item):
    question = item["question"]
    m = re.match(r"What is the mechanism by which (.+?) affects (.+?)\??$", question, re.IGNORECASE)
    drug    = m.group(1).strip() if m else ""
    protein = m.group(2).strip() if m else ""

    pmids, abstracts = collector(drug, protein)
    evidence = f"PMIDs retrieved (not enforced): {pmids}\nAbstracts:\n{abstracts}"

    answer = _llm(RESPONDER_SYS, f"Evidence:\n{evidence}\n\nQuestion: {question}")
    citation_halluc_detected = 0

    for iteration in range(2):
        ver_raw = _llm(VERIFIER_NO_WHITELIST_SYS,
                       f"Answer:\n{answer}", temp=0.1, max_tok=300)
        try:
            jm = re.search(r'\{[\s\S]+\}', ver_raw)
            v  = json.loads(jm.group()) if jm else {}
        except:
            v = {}
        score  = (float(v.get("factuality_score",0)) + float(v.get("grounding_score",0))
                  + float(v.get("format_score",0))   + float(v.get("consistency_score",0)))
        errors = v.get("errors", [])

        if score >= 8.0 or not errors:
            break

        if score < 6.0:
            # Regenerate
            regen_prompt = f"Evidence:\n{evidence}\n\nPrevious errors:\n{errors}\n\nQuestion: {question}"
            answer = _llm(REGENERATOR_SYS, regen_prompt, temp=0.6)
        else:
            # Refine
            fix_prompt = f"Errors:\n{chr(10).join(errors)}\n\nAnswer:\n{answer}"
            answer = _llm(REFINER_SYS, fix_prompt, temp=0.3)

    # Detect citation hallucination POST-HOC:
    # A PMID in the answer that is NOT in the whitelist counts as hallucination
    mentioned_pmids = re.findall(r'PMID[:\s]*(\d{7,8})', answer)
    hallucinated    = [p for p in mentioned_pmids if p not in pmids]
    citation_halluc_detected = len(hallucinated) > 0

    return {"question": question, "answer": answer,
            "variant": "wo_pmid_whitelist",
            "pmids_allowed": pmids,
            "pmids_mentioned": mentioned_pmids,
            "citation_hallucination": citation_halluc_detected}


def run(input_file, output_file, workers=2):
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    done_questions = set()
    results = []
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            results = json.load(f)
            done_questions = {r["question"][:80] for r in results}
    todo = [it for it in data if it["question"][:80] not in done_questions]
    print(f"[w/o PMID Whitelist] {len(todo)} remaining / {len(data)} total")
    if not todo:
        halluc_rate = sum(1 for r in results if r.get('citation_hallucination')) / len(results) * 100
        print(f"All done! Citation Hallucination Rate: {halluc_rate:.1f}%"); return results
    lock = threading.Lock()
    done_count = [len(results)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(generate, it): it for it in todo}
        for fut in as_completed(futures):
            r = fut.result()
            with lock:
                results.append(r)
                done_count[0] += 1
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            n_h = sum(1 for rr in results if rr.get('citation_hallucination'))
            print(f"  [{done_count[0]}/{len(data)}] halluc={n_h}", flush=True)
    halluc_rate = sum(1 for r in results if r.get('citation_hallucination')) / len(results) * 100
    print(f"\nCitation Hallucination Rate (w/o whitelist): {halluc_rate:.1f}%")
    print(f"Saved → {output_file} ({len(results)} results)")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   default="../00_sample_200/ablation_200_samples.json")
    ap.add_argument("--output",  default="output/wo_pmid_whitelist_results.json")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    run(args.input, args.output, args.workers)
