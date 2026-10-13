# BioReasonPipeline - 基础药物-蛋白互作（当前版本）

本模块用于生成 `drug -> target protein` 机制问答样本，当前主实现是 `langgraph_pipeline.py`（LangGraph 多节点协作）。

## 当前流程

```text
Collector -> Responder -> Verifier -> (Refiner/Regenerator循环) -> Finalize
```

- Collector：收集“机制导向”证据包（ChEMBL机制字段 + PubMed机制文献 + UniProt核心功能）
- Responder：仅基于证据包生成结构化回答
- Verifier：规则校验 + LLM评审 + 子分解释
- Refiner/Regenerator：根据错误分级修正或重生成

## 核心文件

| 文件 | 作用 |
|------|------|
| `langgraph_pipeline.py` | 当前主程序（推荐） |
| `pipeline.py` | 历史版本主程序 |
| `drug_target_pairs.json` | 输入药物-靶点对 |
| `output/*.json` | 输出样本 |
| `LANGGRAPH_README.md` | LangGraph版运行说明 |
| `MULTI_AGENT_FLOW.md` | 流程图与节点说明 |

## 运行

```bash
# 生成3条样本
python langgraph_pipeline.py --count 3

# 指定输出路径
python langgraph_pipeline.py --count 5 --output output/langgraph_batch_output_eval5.json

# 设置最大修正轮数
python langgraph_pipeline.py --count 5 --max-corrections 2
```

## 输出字段

```json
{
  "question": "...",
  "answer": "...",
  "category": "Basic-Drug Interaction",
  "drug": "...",
  "protein": "...",
  "uniprot_id": "...",
  "quality_score": 0.0,
  "quality_log": {
    "last_route": "pass|refine|regenerate|stop",
    "correction_count": 0,
    "subscores": {
      "factuality_base": 0.0,
      "factuality_llm_raw": 0.0,
      "factuality_effective": 0.0,
      "grounding_llm_raw": 0.0,
      "format_llm_raw": 0.0,
      "consistency_llm_raw": 0.0
    },
    "errors": [],
    "critical_errors": []
  },
  "elapsed_seconds": 0.0
}
```

## 核心提示模板

### 1) 收集数据模板（Collector 检索模板）

Collector 使用机制关键词检索 PubMed：

```text
({drug}[Title/Abstract]) AND
(({protein}[Title/Abstract]) OR ({gene}[Title/Abstract])) AND
(mechanism[Title/Abstract] OR "mode of action"[Title/Abstract] OR inhibit*[Title/Abstract] OR bind*[Title/Abstract])
```

并补充 ChEMBL `drug_mechanism` 字段与 UniProt 核心功能信息。

### 2) 生成数据模板（Responder System Prompt）

```text
You are a biomedical domain writer creating high-quality training samples.
You must answer based only on the provided evidence package.
Do not add facts not supported by provided IDs or snippets.

Output must keep this structure:
【Core Conclusion】...
【Protein Information】...
【Condition】...
【Clinical Significance】...
【Evidence】...

Core Conclusion requirements:
- Explain ONLY how the drug changes the target protein's activity or function.
- Include the interaction type if supported (e.g., inhibitor/agonist/antagonist/binder) and the immediate biochemical consequence (e.g., blocks catalytic activity, prevents phosphorylation).
- Avoid background, definitions, and broad pharmacology in this section.
- Do not include residue positions, numeric constants, binding pocket details, or “typical” claims unless explicitly supported by the evidence package text.

Other constraints:
1) Avoid long bullet stacks. Prefer concise paragraphs.
2) Evidence must cite UniProt, DrugBank and 1-3 PMIDs from the provided list only.
3) Each PMID cited must correspond to at least one concrete mechanism statement in the answer.
4) Always include the UniProt ID and the drug name in 【Protein Information】.
5) If evidence is insufficient, explicitly state uncertainty instead of inventing details.
6) English only.
```

### 3) 验证数据模板（Verifier System Prompt）

```text
You are a strict biomedical quality auditor.
Score the answer using only the given question, pair metadata and evidence package.
Return JSON only:
{
  "factuality_score": 0-4,
  "grounding_score": 0-3,
  "format_score": 0-2,
  "consistency_score": 0-1,
  "errors": ["..."],
  "critical_errors": ["..."],
  "revision_advice": ["..."]
}

Factuality rubric (0-4):
0 = core mechanism is wrong or contradicts evidence
1 = major mechanism mismatch or mostly unsupported claims
2 = partially correct direction but key mechanism links unsupported
3 = mostly correct mechanism with minor imprecision
4 = mechanism is directly supported and internally consistent

Critical errors include unsupported evidence, wrong target/drug entity, major biomedical contradiction.
```

## 评分说明（当前实现）

- 先做规则校验得到 `factuality_base / det_grounding / det_format / det_consistency`
- 再用 LLM `factuality_score` 对 `factuality_base` 做小幅校准（±1，截断在0~4）
- 最终融合打分并附带 `subscores`，便于筛选和审计

## 依赖

- Python 3.8+
- requests
- openai
- langgraph
