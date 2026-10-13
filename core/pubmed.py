"""M4-B2（P4 抽取 + B2/B3）：PubMed / NCBI E-utilities 检索共享层。

原 4 模块各自内联的 PubMed 检索实现收敛于此；全部经由 core.http.cached_get：
- SQLite 缓存（同一查询跨运行/跨模块只请求一次）；
- NCBI 令牌桶限流（3 req/s，缓存未命中时生效）。
"""
import re
from typing import Dict, List

from core.http import cached_get

REQUESTS_HEADERS = {"User-Agent": "BioReasonPipeline/0.1 (mailto:local)"}


def fetch_pubmed_pmids(query: str, retmax: int = 6) -> List[str]:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    try:
        r = cached_get(
            url,
            params={"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json"},
            timeout=15,
            headers=REQUESTS_HEADERS,
            ncbi=True,
        )
        if r.status_code != 200:
            return []
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        return [x for x in ids if x]
    except Exception:
        return []


def fetch_pubmed_titles(pmids: List[str]) -> Dict[str, str]:
    if not pmids:
        return {}
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    try:
        r = cached_get(
            url,
            params={"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
            timeout=20,
            headers=REQUESTS_HEADERS,
            ncbi=True,
        )
        if r.status_code != 200:
            return {}
        result = r.json().get("result", {})
        out: Dict[str, str] = {}
        for pmid in pmids:
            title = result.get(pmid, {}).get("title", "")
            if title:
                out[pmid] = title
        return out
    except Exception:
        return {}


def fetch_pubmed_abstract_snippet(pmid: str, max_chars: int = 700) -> str:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    try:
        r = cached_get(
            url,
            params={"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "text"},
            timeout=20,
            headers=REQUESTS_HEADERS,
            ncbi=True,
        )
        if r.status_code != 200:
            return ""
        text = re.sub(r"\s+", " ", r.text).strip()
        if not text:
            return ""
        return text[:max_chars]
    except Exception:
        return ""


def fetch_pubmed_article_xml(pmid: str) -> str:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    try:
        r = cached_get(
            url,
            params={"db": "pubmed", "id": pmid, "retmode": "xml"},
            timeout=25,
            headers=REQUESTS_HEADERS,
            ncbi=True,
        )
        if r.status_code != 200:
            return ""
        return r.text
    except Exception:
        return ""


def extract_pubmed_title_abstract_from_xml(xml_text: str) -> Dict[str, str]:
    title_match = re.search(r"<ArticleTitle>([\s\S]*?)</ArticleTitle>", xml_text)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
    abstract_texts = re.findall(r"<AbstractText[^>]*>([\s\S]*?)</AbstractText>", xml_text)
    abstract = " ".join(re.sub(r"\s+", " ", t).strip() for t in abstract_texts if t)
    abstract = abstract.strip()
    journal_match = re.search(r"<Title>([\s\S]*?)</Title>", xml_text)
    journal = re.sub(r"\s+", " ", journal_match.group(1)).strip() if journal_match else ""
    year_match = re.search(r"<PubDate>[\s\S]*?<Year>(\d{4})</Year>", xml_text)
    year = year_match.group(1) if year_match else ""
    return {"title": title, "abstract": abstract, "journal": journal, "year": year}


def fetch_pmcid_for_pmid(pmid: str) -> str:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    try:
        r = cached_get(
            url,
            params={"dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "json"},
            timeout=20,
            headers=REQUESTS_HEADERS,
            ncbi=True,
        )
        if r.status_code != 200:
            return ""
        data = r.json()
        linksets = data.get("linksets", [])
        for ls in linksets:
            linksetdbs = ls.get("linksetdbs", [])
            for db in linksetdbs:
                if db.get("dbto") != "pmc":
                    continue
                links = db.get("links", [])
                if links:
                    return str(links[0])
        return ""
    except Exception:
        return ""


def fetch_pmc_fulltext_excerpt(pmcid: str, max_chars: int = 1500) -> str:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    try:
        r = cached_get(
            url,
            params={"db": "pmc", "id": pmcid, "retmode": "xml"},
            timeout=30,
            headers=REQUESTS_HEADERS,
            ncbi=True,
        )
        if r.status_code != 200:
            return ""
        xml_text = r.text
        body_match = re.search(r"<body[\s\S]*?>([\s\S]*?)</body>", xml_text)
        if not body_match:
            return ""
        body = body_match.group(1)
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        if not body:
            return ""
        return body[:max_chars]
    except Exception:
        return ""
