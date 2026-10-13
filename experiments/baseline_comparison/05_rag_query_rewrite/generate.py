# -*- coding: utf-8 -*-
"""
Baseline 5 — Standard RAG + Query Rewrite (Stronger RAG Baseline)
Uses LLM-based query expansion with biomedical synonyms before PubMed retrieval.
Single-pass retrieval (Top-5) + LLM generation. No structure constraint.
No PMID whitelist. No correction loop.

Rationale: Addresses the lexical gap weakness of Standard RAG by expanding
drug/protein names with synonyms, alternative names, and related terms.
"""

import json, os, sys, time, argparse, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.api_config import NVIDIA_API_KEY, NVIDIA_BASE_URL, GENERATION_MODEL

client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

# ── Query Rewrite via LLM ─────────────────────────────────────────────────────
QUERY_REWRITE_SYSTEM = """You are a biomedical query expansion expert.
Given a drug name and a protein/target name, expand them into a rich PubMed search query.

Output ONLY the expanded query string (no explanation). Include:
1. Known synonyms, brand names, and alternative names for the drug
2. Gene symbols, protein names, and alternative identifiers for the target
3. Mechanism-related terms (inhibit, bind, activate, antagonist, agonist, etc.)

Format: drug_synonyms AND protein_synonyms AND mechanism_terms

Example:
Input: Aspirin, PTGS2
Output: (Aspirin OR "acetylsalicylic acid" OR ASA) AND (PTGS2 OR COX-2 OR "cyclooxygenase-2" OR "prostaglandin G/H synthase 2") AND (inhibit OR bind OR mechanism OR "mode of action" OR acetylate)"""


def rewrite_query(drug: str, protein: str) -> str:
    """Use LLM to expand drug/protein names with biomedical synonyms."""
    user_msg = f"Drug: {drug}\nProtein: {protein}"
    try:
        resp = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=300,
            timeout=30,
        )
        expanded = resp.choices[0].message.content.strip()
        # Remove any markdown formatting or quotes the LLM might add
        expanded = expanded.replace('"', '').replace("`", "")
        return expanded
    except Exception as e:
        # Fallback to original simple query
        return f"{drug}[Title/Abstract] AND {protein}[Title/Abstract] AND mechanism[Title/Abstract]"


# ── PubMed retrieval ──────────────────────────────────────────────────────────
def fetch_pubmed_abstracts(query: str, top_k: int = 5) -> str:
    """Keyword-based PubMed search with expanded query."""
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmax": top_k,
                  "retmode": "json", "sort": "relevance"}
        r = requests.get(search_url, params=params, timeout=10)
        pmids = r.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return "No PubMed results found."

        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {"db": "pubmed", "id": ",".join(pmids),
                        "rettype": "abstract", "retmode": "text"}
        fr = requests.get(fetch_url, params=fetch_params, timeout=15)
        time.sleep(0.4)
        return fr.text[:4000]
    except Exception as e:
        return f"PubMed retrieval failed: {e}"


SYSTEM_PROMPT = """You are a biomedical expert. Using ONLY the provided PubMed abstracts,
answer the question about the drug-protein interaction mechanism.
Cite the PMID(s) from the abstracts that support your answer."""


def generate_rag_qr(item: dict) -> dict:
    question = item["question"]
    drug    = item.get("drug",    question.split(" affects ")[0].replace("What is the mechanism by which ", "").strip())
    protein = item.get("protein", question.split(" affects ")[-1].replace("?", "").strip())

    # Step 1: Query Rewrite
    expanded_query = rewrite_query(drug, protein)

    # Step 2: PubMed retrieval with expanded query
    abstracts = fetch_pubmed_abstracts(expanded_query)

    # Step 3: Generate answer
    user_msg = f"PubMed Evidence:\n{abstracts}\n\nQuestion: {question}"

    try:
        resp = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.5,
            max_tokens=1500,
            timeout=90,
        )
        answer = resp.choices[0].message.content or ""
        return {
            "question":      question,
            "answer":        answer,
            "variant":       "rag_query_rewrite",
            "source":        item.get("source", ""),
            "expanded_query": expanded_query,
            "pmids_used":    abstracts[:200],
        }
    except Exception as e:
        return {"question": question, "answer": "", "variant": "rag_query_rewrite",
                "error": str(e)}


def run(input_file: str, output_file: str, count: int = 100, workers: int = 4):
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)
    data = data[:count]
    print(f"[RAG+QueryRewrite] Running on {len(data)} samples ...")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(generate_rag_qr, it): it for it in data}
        done = 0
        for f in as_completed(futures):
            done += 1
            r = f.result()
            results.append(r)
            status = "OK" if "error" not in r else f"ERR:{r.get('error','')[:40]}"
            print(f"  [{done}/{len(data)}] {status}", flush=True)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[RAG+QueryRewrite] Saved {len(results)} results -> {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="../../evaluation/our_data_100_for_eval.json")
    parser.add_argument("--output",  default="output/rag_query_rewrite_results.json")
    parser.add_argument("--count",   type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.input, args.output, args.count, args.workers)
