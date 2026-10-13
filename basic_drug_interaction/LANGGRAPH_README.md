# LangGraph版本（basic_drug_interaction）

## 安装依赖

```bash
pip install langgraph openai requests
```

## 环境变量

```bash
set BIOREASON_API_KEY=你的密钥
set BIOREASON_BASE_URL=https://api.siliconflow.cn/v1
set BIOREASON_MODEL=Qwen/Qwen2.5-7B-Instruct
```

如果未设置 `BIOREASON_API_KEY`，脚本会自动尝试读取同目录 `pipeline.py` 的 `API_KEY` 作为兼容回退。

## 运行

```bash
# 单条
python langgraph_pipeline.py --count 1

# 多条并指定输出文件
python langgraph_pipeline.py --count 5 --output output/langgraph_batch_output_eval5.json
```

可选参数：

- `--output`：输出文件路径，默认 `output/langgraph_batch_output.json`
- `--max-corrections`：Verifier-Refiner循环最大次数，默认 `2`

## 当前证据收集策略（Collector）

- ChEMBL：读取 `drug_mechanism` 记录（action_type、mechanism_of_action、target_name、reference）
- PubMed：机制关键词检索 + 标题打分排序 + 抽取前几篇 title/abstract
- UniProt：保留蛋白核心功能字段（辅助上下文，不做冗余扩展）

PubMed 查询模板：

```text
({drug}[Title/Abstract]) AND
(({protein}[Title/Abstract]) OR ({gene}[Title/Abstract])) AND
(mechanism[Title/Abstract] OR "mode of action"[Title/Abstract] OR inhibit*[Title/Abstract] OR bind*[Title/Abstract])
```

## 当前生成与验证核心 Prompt

### Responder（生成）

- 强制5段结构：Core/Protein/Condition/Clinical/Evidence
- Core 只讲互作机制，不讲背景
- 证据引用限制：1-3条PMID，且必须来自 allowed 列表

完整模板见：
- [README.md](file:///c:/Users/JH/Desktop/bioreason_pipeline2.0/bioreason_pipeline/basic_drug_interaction/README.md)

### Verifier（验证）

- 输出四维分：factuality(0-4)、grounding(0-3)、format(0-2)、consistency(0-1)
- 含 factuality 0-4 细化 rubric
- 返回 errors/critical_errors/revision_advice

## 评分机制（当前版本）

- 规则基线：先算 `factuality_base`（由实体一致性、证据支持、critical等规则决定）
- LLM校准：用 `factuality_llm_raw` 对基线做 ±1 小幅修正
- 融合评分：再叠加 grounding/format/consistency 并扣除 error penalty
- 严重错误封顶：有 critical 时总分上限 5.9

## 产出结构

每条样本包含：

- `question`
- `answer`
- `category`
- `drug`
- `protein`
- `uniprot_id`
- `quality_score`
- `quality_log.last_route`
- `quality_log.correction_count`
- `quality_log.subscores`（含 factuality_base/factuality_effective 等）
- `quality_log.errors`
- `quality_log.critical_errors`
- `elapsed_seconds`
