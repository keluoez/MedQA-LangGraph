# -*- coding: utf-8 -*-
"""
Baseline 2 — Standard RAG
Single-pass retrieval (Top-5 PMIDs via PubMed) + LLM generation.
No structure constraint. No PMID whitelist. No correction loop.
Corresponds to "Standard RAG" row in Table II.
"""

import json, os, sys, time, argparse, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.api_config import NVIDIA_API_KEY, NVIDIA_BASE_URL, GENERATION_MODEL

client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

# ── PubMed retrieval (Top-5 PMIDs) ────────────────────────────────────────────
def fetch_pubmed_abstracts(drug: str, protein: str, top_k: int = 5) -> str:
    """Keyword-based PubMed search — same depth as the Collector."""
    try:
        query = f"{drug}[Title/Abstract] AND {protein}[Title/Abstract] AND mechanism[Title/Abstract]"
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
        time.sleep(0.4)          # respect rate limit
        return fr.text[:4000]    # cap context length
    except Exception as e:
        return f"PubMed retrieval failed: {e}"


SYSTEM_PROMPT = """You are a biomedical expert. Using ONLY the provided PubMed abstracts,
answer the question about the drug-protein interaction mechanism.
Cite the PMID(s) from the abstracts that support your answer."""


def generate_standard_rag(item: dict) -> dict:
    question = item["question"]
    drug    = item.get("drug",    question.split(" affects ")[0].replace("What is the mechanism by which ", "").strip())
    protein = item.get("protein", question.split(" affects ")[-1].replace("?", "").strip())

    abstracts = fetch_pubmed_abstracts(drug, protein)
    user_msg  = f"PubMed Evidence:\n{abstracts}\n\nQuestion: {question}"

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
            "question":   question,
            "answer":     answer,
            "variant":    "standard_rag",
            "source":     item.get("source", ""),
            "pmids_used": abstracts[:200],
        }
    except Exception as e:
        return {"question": question, "answer": "", "variant": "standard_rag",
                "error": str(e)}


def run(input_file: str, output_file: str, count: int = 100, workers: int = 5):
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)
    data = data[:count]
    print(f"[Standard RAG] Running on {len(data)} samples …")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(generate_standard_rag, it): it for it in data}
        done = 0
        for f in as_completed(futures):
            done += 1
            r = f.result()
            results.append(r)
            status = "OK" if "error" not in r else "ERR"
            print(f"  [{done}/{len(data)}] {status}", flush=True)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Standard RAG] Saved {len(results)} results → {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="../../evaluation/our_data_100_for_eval.json")
    parser.add_argument("--output",  default="output/standard_rag_results.json")
    parser.add_argument("--count",   type=int, default=100)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    run(args.input, args.output, args.count, args.workers)
