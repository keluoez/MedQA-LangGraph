# -*- coding: utf-8 -*-
"""
Baseline 3 — Iterative RAG (Agentic RAG)
Uses our Collector (mechanism-oriented retrieval) + Responder (structure-constrained),
but skips the Verifier / Refiner / Regenerator entirely.
Represents "have retrieval + generation, but no closed-loop verification".
Corresponds to "Iterative RAG" row in Table II.
"""

import json, os, sys, time, argparse, re, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.api_config import NVIDIA_API_KEY, NVIDIA_BASE_URL, GENERATION_MODEL

client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

# ── Mechanism-oriented Collector (same as full pipeline) ──────────────────────
def collector(drug: str, protein: str, uniprot_id: str = "") -> str:
    """Retrieve mechanism-oriented evidence from PubMed (Top-5) + UniProt stub."""
    evidence_parts = []

    # PubMed: drug + protein + mechanistic terms
    try:
        query = (f"{drug}[Title/Abstract] AND {protein}[Title/Abstract] AND "
                 f"(mechanism OR inhibit OR bind OR activate OR \"mode of action\")[Title/Abstract]")
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": 5,
                    "retmode": "json", "sort": "relevance"},
            timeout=10)
        pmids = r.json().get("esearchresult", {}).get("idlist", [])
        if pmids:
            fr = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(pmids),
                        "rettype": "abstract", "retmode": "text"},
                timeout=15)
            evidence_parts.append(f"PubMed Abstracts (PMIDs: {', '.join(pmids)}):\n{fr.text[:3000]}")
        time.sleep(0.4)
    except Exception as e:
        evidence_parts.append(f"PubMed retrieval failed: {e}")

    return "\n\n".join(evidence_parts) if evidence_parts else "No evidence retrieved."


# ── Structure-constrained Responder (same as full pipeline) ───────────────────
RESPONDER_SYSTEM = """You are a biomedical expert generating high-fidelity mechanism QA data.
Output STRICTLY in this 5-section format:

【Core Conclusion】 — How the drug alters the protein's biochemical activity (1-2 sentences, mechanism only).
【Protein Information】 — Protein name, UniProt ID, key function.
【Condition】 — Disease or physiological context.
【Clinical Significance】 — Therapeutic implications.
【Evidence】 — Cite specific PMIDs and UniProt IDs from the provided evidence.

RULES:
- Do NOT invent numerical residue indices or Ki values unless present in the evidence.
- Every claim in 【Evidence】 must come from the provided retrieval context."""


def responder(question: str, evidence: str) -> str:
    user_msg = f"Evidence Package:\n{evidence}\n\nQuestion: {question}"
    try:
        resp = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": RESPONDER_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.5,
            max_tokens=2000,
            timeout=90,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"[ERROR] {e}"


def generate_iterative_rag(item: dict) -> dict:
    question = item["question"]
    # Best-effort drug/protein extraction from question
    m = re.match(r"What is the mechanism by which (.+?) affects (.+?)\??$", question, re.IGNORECASE)
    drug    = m.group(1).strip() if m else ""
    protein = m.group(2).strip() if m else ""

    evidence = collector(drug, protein)
    answer   = responder(question, evidence)

    return {
        "question": question,
        "answer":   answer,
        "variant":  "iterative_rag",
        "source":   item.get("source", ""),
    }


def run(input_file: str, output_file: str, count: int = 100, workers: int = 5):
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)
    data = data[:count]
    print(f"[Iterative RAG] Running on {len(data)} samples …")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(generate_iterative_rag, it): it for it in data}
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
    print(f"[Iterative RAG] Saved {len(results)} results → {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="../../evaluation/our_data_100_for_eval.json")
    parser.add_argument("--output",  default="output/iterative_rag_results.json")
    parser.add_argument("--count",   type=int, default=100)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    run(args.input, args.output, args.count, args.workers)
