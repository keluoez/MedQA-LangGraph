# BioReasonPipeline - 病理环境模块

基于LangGraph多智能体流水线的生物医药推理数据集生成系统

## 功能概述

从权威数据库（UniProt、PubMed）采集病理环境信息，生成疾病/病理状态对蛋白功能影响的数据集。

## 数据类别

**Basic-Pathological Environment**: 病理环境对蛋白功能的影响

示例问题：
- How does DNA damage in Cancer affect TP53 function?
- How does Hypoxic tumor microenvironment in Cancer affect VEGF function?

## 智能体说明

| 智能体 | 功能 | 说明 |
|--------|------|------|
| **Collector** | 知识收集 | 从UniProt/PubMed获取蛋白功能和病理机制，构建PMID白名单 |
| **Responder** | 生成回答 | 基于证据包生成结构化回答（只能引用白名单PMID） |
| **Verifier** | 质量验证 | 确定性检查 + LLM分层评分 |
| **Refiner** | 修正 | 修正Verifier发现的错误 |
| **Regenerate** | 重新生成 | 关键错误时重新生成回答 |
| **Finalize** | 格式化输出 | 清理格式、计算评分、输出JSON |

## 核心机制

### PMID白名单机制
- Collector收集6个最相关PMID作为白名单
- 只获取前3篇的详细摘要（效率优化）
- 回答只能引用白名单内的PMID
- 白名单外的PMID被标记为关键错误

### 分层评分系统
- 确定性检查：标签完整性、关键信息一致性、PMID白名单验证
- LLM评分：病理机制(0-4)、疾病关联(0-3)、证据支撑(0-2)、格式(0-2)
- 基础分由确定性检查驱动，LLM评分作为调整因子

### 验证循环逻辑

```
Verifier → 总分≥9 或 (≥8且无关键错误)? → Finalize(输出)
          │
         否 → 修正次数≥2? → Finalize(输出当前最佳)
          │
         否 → 有关键错误且<7分? → Regenerate → Verifier
          │
         否 → Refiner修正 → Verifier
```

最大修正次数：2次

## 文件说明

| 文件 | 说明 |
|------|------|
| `langgraph_pipeline.py` | ⭐ 主程序（LangGraph实现） |
| `pathology_pairs.json` | 输入数据 |
| `output/` | 生成结果目录 |
| `LANGGRAPH_README.md` | LangGraph实现详细文档 |
| `TECHNICAL_DETAILS.md` | 技术细节文档 |

## 运行方式

```bash
# 处理指定数量数据
python langgraph_pipeline.py --count N

# 指定输出文件
python langgraph_pipeline.py --count 5 --output output/my_results.json

# 设置最大修正次数（默认2次）
python langgraph_pipeline.py --count 5 --max-corrections 2
```

## 输出格式

```json
{
  "question": "How does DNA damage in Cancer affect TP53 function?",
  "answer": "【Core Conclusion】DNA damage in cancer leads to activation of TP53...\n\n【Protein Information】TP53 (UniProt: P04637)\n\n【Condition】Cancer, specifically focusing on DNA damage...\n\n【Clinical Significance】TP53 mutations are frequent in cancer...\n\n【Evidence】DNA damage triggers TP53 activation (PMID: 41934032)...",
  "category": "Basic-Pathological Environment",
  "disease": "Cancer",
  "pathology": "DNA damage",
  "protein": "TP53",
  "uniprot_id": "P04637",
  "quality_score": 9.0
}
```

## 数据库

- **UniProt** - 蛋白质功能信息、病理注释、疾病关联
- **PubMed** - 病理机制文献（智能布尔查询 + PMID排序）
