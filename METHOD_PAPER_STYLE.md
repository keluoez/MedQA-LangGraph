# 基础药物-蛋白互作模块：论文写作版方法说明

## 1. 任务定义

本模块面向药物-蛋白互作机制问答数据构建。给定输入三元组 `{drug, protein, uniprot_id}`（可含 `chembl_id/gene/drugbank_id`），系统自动生成结构化问答样本，核心目标是：

- 机制准确：回答聚焦药物如何改变靶蛋白活性/功能
- 证据可溯源：机制结论可由 ChEMBL/UniProt/PubMed 证据直接支持
- 文本可读：关键信息突出，避免背景冗余

## 2. 总体框架

系统采用 LangGraph 多节点协作流程：

`Collector -> Responder -> Verifier -> (Refiner / Regenerator) -> Finalize`

其中，Verifier 与 Refiner/Regenerator 构成有限次闭环修正，避免单次生成误差直接入库。

## 3. 证据收集策略（Collector）

Collector 采用“机制导向收集”原则，仅保留与 drug-target 作用机制直接相关的信息：

- **ChEMBL**：抽取 `drug_mechanism` 记录（`action_type`, `mechanism_of_action`, `target_name`, `reference`）
- **PubMed**：以机制关键词检索候选文献并排序，保留高相关 PMID 的标题与摘要
- **UniProt**：保留蛋白核心功能字段用于上下文约束（非背景扩展）

PubMed 查询模板：

```text
({drug}[Title/Abstract]) AND
(({protein}[Title/Abstract]) OR ({gene}[Title/Abstract])) AND
(mechanism[Title/Abstract] OR "mode of action"[Title/Abstract] OR inhibit*[Title/Abstract] OR bind*[Title/Abstract])
```

## 4. 回答生成策略（Responder）

Responder 使用结构化生成约束，输出固定五段：

- `【Core Conclusion】`
- `【Protein Information】`
- `【Condition】`
- `【Clinical Significance】`
- `【Evidence】`

其中 `Core Conclusion` 采用强聚焦约束：

- 仅说明药物对目标蛋白活性/功能的直接影响
- 若证据支持，明确作用类型与即时生化后果
- 禁止无证据细节（如位点编号、口袋细节、数值常数）

## 5. 质量验证策略（Verifier）

Verifier 采用“规则判定 + LLM评审”的混合质量控制：

### 5.1 规则层（可重复）

- 结构完整性：五段标签是否齐全
- 实体一致性：drug/protein/uniprot_id 是否匹配
- 证据约束：PMID 是否来自允许集合，是否存在重复/缺失
- 细节反编造：关键机制细节是否在证据文本中可支持

### 5.2 语义评审层（可解释）

LLM 返回四维子分：

- factuality (0-4)
- grounding (0-3)
- format (0-2)
- consistency (0-1)

并输出 `errors / critical_errors / revision_advice`。

## 6. 评分融合机制

当前实现的评分机制为：

1. 规则先生成 `factuality_base`（机制正确性的基线分）  
2. 用 LLM 的 `factuality_score` 对基线做小幅校准（±1，截断到0-4）  
3. 与其余维度融合后扣除错误惩罚  
4. 若存在 critical error，最终总分封顶

该机制兼顾可复现性（规则层）与语义判断能力（LLM层），并在输出中暴露子分项用于审计。

## 7. 闭环修正策略

若样本未通过质量阈值：

- 轻中度问题：进入 Refiner，对现有答案定向修正
- 严重问题或关键矛盾：进入 Regenerator，重生成答案

修正次数受 `max_corrections` 限制，保证流程收敛。

## 8. 输出样本与可解释日志

每条样本包含：

- 基础字段：`question, answer, category, drug, protein, uniprot_id`
- 质量字段：`quality_score`
- 可解释字段：`quality_log.subscores, quality_log.errors, quality_log.critical_errors`
- 过程字段：`last_route, correction_count, elapsed_seconds`

该设计支持后续基于阈值的自动筛选与人工抽检协同。

## 9. 与代码实现映射

- 主流程编排：`langgraph_pipeline.py`
- 证据采集：`node_collect`
- 生成节点：`node_generate`
- 质检节点：`node_verify`
- 修正节点：`node_refine` / `node_regenerate`
- 最终输出：`node_finalize`

