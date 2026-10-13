# BioReasonPipeline - 生理条件模块

## 模块简介

**生理条件模块**是BioReasonPipeline的三大核心模块之一，专门用于生成关于"生理条件下蛋白质功能"的高质量训练数据集。

该模块采用基于LangGraph框架的多智能体架构，通过自动化收集UniProt蛋白质数据库和PubMed生物医学文献中的证据信息，生成结构化的训练样本。

## 核心能力

本模块能够处理以下类型的生物医学问答：

- **体温与蛋白功能**: "在正常体温(37°C)下，血红蛋白如何执行氧运输功能？"
- **pH与蛋白功能**: "在正常血浆pH(7.35-7.45)下，白蛋白如何维持血浆胶体渗透压？"
- **代谢状态与蛋白功能**: "在禁食状态下，胰高血糖素如何调节血糖平衡？"
- **组织特异性**: "在不同组织中，目标蛋白的表达和功能有何差异？"

## 数据来源

### UniProt蛋白质数据库

- **蛋白质功能**: 来自UniProt的核心功能描述
- **催化活性**: 对于酶类蛋白，提供催化反应信息
- **辅因子信息**: 金属离子、维生素等辅因子依赖性
- **组织特异性**: 蛋白在不同组织的表达分布
- **诱导/抑制**: 特定生理条件下的表达调控

### PubMed文献数据库

- **高质量文献筛选**: 通过相关性排名算法选择最相关的文献
- **PMID白名单机制**: 只允许引用经过验证的文献
- **摘要信息**: 提取文献的标题和摘要作为证据

## 输出示例

```json
{
  "question": "What is the function of Hemoglobin under Normal body temperature (37°C)?",
  "answer": "【Core Conclusion】\nAt normal body temperature (37°C), hemoglobin efficiently transports oxygen from the lungs to peripheral tissues.\n\n【Protein Information】\nTarget Protein: Hemoglobin subunit beta (UniProt: P68871)\n\n【Condition】\nPhysiological Condition: Normal body temperature (37°C)\n\n【Physiological Importance】\nThis temperature optimizes hemoglobin's oxygen-binding affinity and rate, ensuring efficient oxygen transport.\n\n【Evidence】\nHemoglobin's function in oxygen transport is supported by structural studies at physiological temperature. (PMID: xxxxxx)",
  "category": "Basic-Physiological Conditions",
  "condition": "Normal body temperature (37°C)",
  "protein": "Hemoglobin",
  "uniprot_id": "P68871"
}
```

## 与其他模块的关系

BioReasonPipeline包含三个互补的数据集生成模块：

| 模块 | 输入 | 核心问题 | 输出章节 |
|------|------|---------|---------|
| **病理环境模块** | 疾病状态 | "疾病如何影响蛋白功能？" | 【Clinical Significance】 |
| **生理条件模块** | 生理状态 | "生理条件下蛋白的功能是什么？" | 【Physiological Importance】 |
| **实验条件模块** | 实验条件 | "实验条件如何影响蛋白活性？" | 【Clinical Relevance】 |

## 使用指南

### 快速开始

```bash
# 进入模块目录
cd basic_physiological_conditions

# 运行LangGraph流水线
python langgraph_pipeline.py --count 5 --output output/results.json
```

### 输入数据格式

编辑 `physiology_pairs.json` 文件，添加要处理的蛋白-条件对：

```json
[
  {
    "condition": "Normal body temperature (37°C)",
    "protein": "Hemoglobin",
    "uniprot_id": "P68871"
  }
]
```

### 输出结果

生成的数据将保存在 `output/` 目录下，包含：
- 结构化的问答对
- 质量评分
- 处理耗时
- 质量评估详情

## 质量保证

本模块通过多层次的质量控制机制确保输出数据的可靠性：

1. **证据收集**: 从权威数据库(UniProt、PubMed)获取原始证据
2. **白名单机制**: 严格限制PMID引用范围，防止虚构引用
3. **多轮修正**: 通过验证-修正循环不断提升质量
4. **评分体系**: 多维度评分(事实性、证据支持、格式规范、一致性)

## 技术架构

- **框架**: LangGraph状态图
- **智能体**: Collector、Responder、Verifier、Refiner、Regenerator、Finalize
- **LLM模型**: Qwen/Qwen2.5-7B-Instruct (SiliconFlow API)
- **API**: UniProt REST API、PubMed E-utilities

## 适用场景

本模块生成的数据适用于：

- 训练生物医学问答系统
- 构建蛋白质功能知识库
- 开发疾病机理理解模型
- 辅助生物医学文献挖掘
