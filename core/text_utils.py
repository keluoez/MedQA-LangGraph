"""M4-B2（P4 抽取）：答案文本后处理共享工具。

- normalize_sections：按领域段落标签重组答案（标签清单由各模块传入）；
- extract_pmids：从文本中提取 PMID/PUBMED 引用编号。
"""
import re
from typing import List


def normalize_sections(answer: str, tags: List[str]) -> str:
    """按 tags 顺序抽取答案段落并重组；无任何命中时原样返回（strip 后）。"""
    chunks: List[str] = []
    for i, tag in enumerate(tags):
        start = answer.find(tag)
        if start == -1:
            continue
        end_candidates = [answer.find(t, start + len(tag)) for t in tags[i + 1 :]]
        end_candidates = [x for x in end_candidates if x != -1]
        end = min(end_candidates) if end_candidates else len(answer)
        piece = answer[start:end].strip()
        if piece:
            chunks.append(piece)
    if not chunks:
        return answer.strip()
    return "\n\n".join(chunks).strip()


def extract_pmids(text: str) -> List[str]:
    return re.findall(r"(?:PMID|PubMed)[:\s]*(\d+)", text, flags=re.IGNORECASE)
