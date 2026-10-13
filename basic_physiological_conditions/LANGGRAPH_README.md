# BioReasonPipeline - 生理条件模块 (LangGraph)

## 概述

**生理条件模块**是BioReasonPipeline的核心组件之一，专门用于生成关于"生理条件下蛋白功能"的高质量训练数据集。

该模块基于LangGraph框架构建，采用多智能体协作架构，通过收集UniProt蛋白质数据库和PubMed文献的证据信息，生成结构化的训练样本。

## 模块职责

本模块专注于回答以下类型的问题：
- "在正常体温(37°C)下，血红蛋白的功能是什么？"
- "在禁食状态下，胰高血糖素如何调节血糖？"
- "运动应激状态下，AMPK如何调控能量代谢？"

## 架构设计

### 多智能体流程

```
┌─────────────┐
│   collect   │  收集UniProt + PubMed证据
└──────┬──────┘
       │
┌──────▼──────┐
│   generate  │  生成初始回答
└──────┬──────┘
       │
┌──────▼──────┐
│   verify    │  质量验证 + 路由决策
└──────┬──────┘
       │
       ├──────────────────────┐
       │                      │
   ┌───▼───┐           ┌─────▼─────┐
   │ refine │           │ regenerate │
   └───┬───┘           └─────┬─────┘
       │                      │
       └──────────┬───────────┘
                  │
           ┌──────▼──────┐
           │  finalize   │  生成最终样本
           └──────┬──────┘
                  │
           ┌──────▼──────┐
           │     END      │
           └─────────────┘
```

### 节点说明

| 节点 | 功能 | 输入 | 输出 |
|------|------|------|------|
| **collect** | 从UniProt和PubMed收集证据 | pair (condition, protein, uniprot_id) | evidence (uniprot数据, PubMed摘要, PMID白名单) |
| **generate** | 基于证据生成结构化回答 | evidence, question | answer (初始版本) |
| **verify** | 验证质量并决策路由 | answer, evidence, question | verification (评分, 错误列表, 路由) |
| **refine** | 修复非关键错误 | verification, answer | answer (修正版本) |
| **regenerate** | 重新生成回答(处理关键错误) | answer, critical_errors | answer (重新生成版本) |
| **finalize** | 生成最终训练样本 | best_answer, scores | final_sample |

### 状态管理 (PipelineState)

```python
class PipelineState(TypedDict, total=False):
    pair: Dict[str, Any]              # 输入对: condition, protein, uniprot_id
    question: str                      # 生成的问题
    evidence: Dict[str, Any]           # 证据: uniprot, pubmed_pmids, articles
    answer: str                       # 当前回答
    best_answer: str                  # 最佳回答(修正过程中最高分)
    best_score: float                 # 最佳评分
    verification: Dict[str, Any]      # 验证结果
    route: Literal["pass","refine","regenerate","stop"]  # 路由决策
    correction_count: int             # 当前修正次数
    max_corrections: int             # 最大修正次数(默认2)
    final_sample: Dict[str, Any]      # 最终输出样本
    elapsed_seconds: float             # 处理耗时
```

## 路由决策机制

验证节点根据评分和错误情况做出路由决策：

| 条件 | 路由 | 说明 |
|------|------|------|
| score ≥ 8.0 且无关键错误 | **pass** | 直接生成最终样本 |
| score < 6.0 或有关键错误 | **regenerate** | 重新生成回答 |
| 其他情况(6.0 ≤ score < 8.0) | **refine** | 修正非关键错误 |
| 达到最大修正次数 | **stop** | 停止修正，使用最佳答案 |

## 数据收集策略

### UniProt数据

收集以下关键信息：
- **protein_name**: 蛋白质名称
- **gene**: 基因名称
- **function**: 蛋白质功能描述
- **catalytic_activity**: 催化活性信息
- **cofactor**: 辅因子信息
- **ph_optimum**: pH最适值
- **tissue_specificity**: 组织特异性
- **induction**: 诱导表达信息
- **pathway**: 代谢/信号通路

### PubMed文献

1. **智能查询策略**:
   - 基于蛋白质名称和生理条件构建查询词
   - 查询词包含: physiology, function, regulation, expression, homeostasis, pathway, mechanism

2. **PMID白名单机制**:
   - 最多保留6个最相关的PMID
   - 获取这6个PMID的详细信息(标题、摘要)
   - 回答中只允许引用白名单中的PMID

3. **PMID排名算法**:
   - 根据标题中关键词匹配度排序
   - 优先级: physiology, function > regulation, expression > homeostasis, pathway
   - 降低review类文章的排名

## 质量评分体系

### 评分构成 (满分10分)

| 维度 | 满分 | 说明 |
|------|------|------|
| **Factuality** | 4.0 | 事实准确性 |
| **Grounding** | 3.0 | 证据支持度 |
| **Format** | 2.0 | 格式规范度 |
| **Consistency** | 1.0 | 一致性(蛋白名/UniProt ID) |

### Factuality评分规则

| 条件 | 基础分 |
|------|-------|
| 存在关键错误 | 0.0 |
| 蛋白名/UniProt ID不一致 | 1.0 |
| 证据部分缺少PMID | 2.0 |
| LLM错误≥3个 | 2.0 |
| 无错误 | 4.0 |

### 错误分类

**关键错误 (Critical Errors)**:
- 使用了白名单外的PMID
- 存在不支持的残基位点详情
- 存在不支持的结合亲和力数值
- 事实性错误或矛盾

**一般错误 (Errors)**:
- 缺少必需章节(【Core Conclusion】等)
- PMID重复引用
- 非英文内容
- 证据部分缺少PMID

## 输出格式

生成的数据样本符合统一格式：

```json
{
  "question": "What is the function of Hemoglobin under Normal body temperature (37°C)?",
  "answer": "【Core Conclusion】\n...\n\n【Protein Information】\n...\n\n【Condition】\n...\n\n【Physiological Importance】\n...\n\n【Evidence】\n...",
  "category": "Basic-Physiological Conditions",
  "condition": "Normal body temperature (37°C)",
  "protein": "Hemoglobin",
  "uniprot_id": "P68871",
  "quality_score": 9.85,
  "quality_log": {
    "last_route": "pass",
    "correction_count": 2,
    "subscores": {...},
    "errors": [...],
    "critical_errors": [...]
  },
  "elapsed_seconds": 82.31
}
```

### 回答结构要求

| 章节 | 内容要求 | 示例 |
|------|---------|------|
| **【Core Conclusion】** | 核心结论，描述蛋白质在给定生理条件下的功能 | "At 37°C, hemoglobin efficiently transports oxygen..." |
| **【Protein Information】** | 蛋白质基本信息(名称+UniProt ID) | "Target Protein: Hemoglobin (UniProt: P68871)" |
| **【Condition】** | 生理条件描述 | "Physiological Condition: Normal body temperature (37°C)" |
| **【Physiological Importance】** | 生理意义/重要性 | "Maintains systemic oxygen delivery..." |
| **【Evidence】** | 证据引用(UniProt ID + PMIDs) | "UniProt: P68871; PMID:41968187" |

## 使用方法

### 命令行参数

```bash
python langgraph_pipeline.py [--count N] [--output OUTPUT_FILE] [--max-corrections N]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--count` | 处理的数据条数 | 全部 |
| `--output` | 输出文件路径 | `output/langgraph_batch_output.json` |
| `--max-corrections` | 最大修正次数 | 2 |

### 示例

```bash
# 生成3条数据
python langgraph_pipeline.py --count 3 --output output/my_results.json

# 生成全部数据
python langgraph_pipeline.py --output output/all_results.json

# 自定义修正次数
python langgraph_pipeline.py --count 5 --max-corrections 3
```

## 与其他模块的对比

| 特性 | 病理环境模块 | 生理条件模块 | 实验条件模块 |
|------|-------------|-------------|-------------|
| **问题类型** | 疾病状态下蛋白功能 | 生理条件下蛋白功能 | 实验条件对蛋白活性的影响 |
| **Condition类型** | Pathological Environment | Physiological Condition | Experimental Condition |
| **输出章节** | 【Clinical Significance】 | 【Physiological Importance】 | 【Clinical Relevance】 |
| **问题模板** | How does {disease} affect {protein}? | What is the function of {protein} under {condition}? | How does {condition} affect {protein} activity? |

## 技术特点

### 1. 确定性与灵活性结合
- 确定性检查: 格式验证、PMID白名单、实体一致性
- LLM辅助评估: 事实性、逻辑性、连贯性

### 2. 最佳答案追踪
- 修正过程中保留最高分答案
- 最终输出使用最佳版本

### 3. 智能路由
- 根据错误严重程度选择不同处理策略
- 避免过度修正或修正不足

### 4. 证据约束
- 严格限制PMID引用范围在白名单内
- 确保回答内容有充分的文献支持

## 依赖项

- `langgraph`: 状态图框架
- `openai`: LLM API调用
- `requests`: HTTP请求(UniProt/PubMed)
- `typing_extensions`: 类型提示

## 参考资料

- LangGraph文档: https://langchain-ai.github.io/langgraph/
- UniProt API: https://rest.uniprot.org/
- PubMed E-utilities: https://www.ncbi.nlm.nih.gov/home/develop/api/
