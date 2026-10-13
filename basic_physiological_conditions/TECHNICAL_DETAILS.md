# 生理条件模块 - 技术细节文档

## 1. 模块概述

本模块实现了一个基于LangGraph框架的自动化训练数据生成系统，专门用于"生理条件下蛋白质功能"这一生物学主题。系统通过多智能体协作，结合权威数据库(UniProt、PubMed)的证据收集和LLM生成，产出高质量的结构化训练样本。

### 1.1 核心目标

- **自动化**: 全程无需人工干预，自动完成数据收集、生成、验证、修正
- **高质量**: 通过多轮验证和修正循环，确保输出数据的准确性和一致性
- **可追溯**: 每个输出样本都包含完整的质量评估日志和证据来源

### 1.2 生物学背景

生理条件模块关注的是**正常生理状态下蛋白质的功能机制**，区别于：
- 病理环境模块（疾病状态）
- 实验条件模块（体外实验条件）

典型问题包括：
- 体温如何影响蛋白质构象和功能
- pH如何影响酶的催化活性
- 代谢状态如何调控激素和酶的功能

## 2. 系统架构

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────┐
│                        LangGraph Pipeline                     │
│                                                               │
│  ┌─────────┐     ┌──────────┐     ┌─────────┐              │
│  │         │────▶│          │────▶│         │              │
│  │ collect │     │ generate │     │ verify  │              │
│  │         │◀────│          │◀────│         │              │
│  └─────────┘     └──────────┘     └────┬────┘              │
│                                         │                    │
│                        ┌────────────────┼────────────────┐   │
│                        │                │                │   │
│                        ▼                ▼                ▼   │
│                   ┌─────────┐     ┌──────────┐    ┌─────────┐│
│                   │         │     │          │    │         ││
│                   │  pass   │     │  refine  │    │regenerate│
│                   │         │     │          │    │         ││
│                   └────┬────┘     └────┬─────┘    └────┬────┘│
│                        │              │               │      │
│                        │              └───────────────┘      │
│                        │                              │      │
│                        ▼                              ▼      │
│                   ┌─────────┐                    ┌─────────┐ │
│                   │         │                    │         │ │
│                   │finalize│                    │ verify  │─┘
│                   │         │                    │         │
│                   └────┬────┘                    └─────────┘ │
│                        │                                    │
│                        ▼                                    │
│                   ┌─────────┐                              │
│                   │   END   │                              │
│                   └─────────┘                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 节点功能详解

#### Collector节点

**职责**: 收集UniProt和PubMed证据

**处理流程**:

1. **UniProt数据获取**
   ```python
   uniprot_core = fetch_uniprot_core(uniprot_id)
   ```

2. **PubMed文献查询**
   ```python
   pubmed_query = (
       f"({protein}[Title/Abstract]) AND "
       f"(({gene}[Title/Abstract]) OR ({protein}[Title/Abstract])) AND "
       f"(physiology[Title/Abstract] OR function[Title/Abstract] OR "
       f"regulation[Title/Abstract] OR expression[Title/Abstract] OR "
       f"homeostasis[Title/Abstract] OR pathway[Title/Abstract])"
   )
   ```

3. **PMID排名筛选**
   - 根据标题关键词匹配度排序
   - 优先选择包含"physiology"、"function"、"regulation"的文献
   - 降低review类文献排名
   - 保留前6个最相关的PMID

4. **详细信息获取**
   - 获取每个PMID的XML格式全文
   - 解析标题、摘要、期刊、年份信息
   - 保留前3篇文章的详细信息

**输出数据**:
```python
state["evidence"] = {
    "uniprot": {
        "uniprot_id": str,
        "protein_name": str,
        "gene": str,
        "function": str,
        "catalytic_activity": str,
        "cofactor": str,
        "ph_optimum": str,
        "tissue_specificity": str,
        "subunit": str,
        "induction": str,
        "pathway": str,
        ...
    },
    "pubmed_pmids": List[str],  # 最多6个
    "pubmed_articles": List[Dict]  # 最多3个详细信息
}
```

#### Generate节点

**职责**: 基于证据生成初始回答

**输入**: evidence, question

**处理流程**:

1. 构建用户提示词，包含：
   - 问题模板
   - 蛋白质和条件信息
   - UniProt证据（功能、催化活性、辅因子等）
   - 允许引用的PMID列表
   - PubMed文章详情（标题+摘要）

2. 调用LLM生成回答

3. 后处理：
   - `normalize_sections()`: 确保章节顺序正确
   - `filter_non_english()`: 过滤非英文内容
   - `normalize_evidence_citations()`: 标准化PMID引用

**提示词设计原则**:

- **聚焦**: 明确要求只描述蛋白质在给定生理条件下的功能
- **约束**: 不能添加证据不支持的事实
- **格式**: 必须包含5个标准章节
- **简洁**: 避免长列表，使用连贯段落

#### Verify节点

**职责**: 评估回答质量并做出路由决策

**评估维度**:

| 维度 | 满分 | 评估方法 |
|------|------|----------|
| Factuality | 4.0 | LLM判断 + 基础规则 |
| Grounding | 3.0 | LLM判断 + PMID白名单检查 |
| Format | 2.0 | 确定性检查（章节完整性）|
| Consistency | 1.0 | 确定性检查（实体一致性）|

**确定性检查项目**:

```python
def deterministic_checks(state):
    # 1. 章节完整性
    missing_sections = [t for t in required_tags if t not in answer]

    # 2. 实体一致性
    if protein_name not in answer.lower():
        errors.append("protein_name_missing_or_inconsistent")
    if uniprot_id not in answer.lower():
        errors.append("uniprot_id_missing_or_inconsistent")

    # 3. PMID合法性
    unsupported_pmids = [p for p in pmids_in_answer if p not in allowed_pmids]
    if unsupported_pmids:
        critical.append(f"unsupported_pmids:{','.join(...)}")

    # 4. PMID重复
    if has_duplicates(pmids_in_answer):
        errors.append("duplicate_pmids_in_evidence")

    # 5. 非英文内容
    if contains_non_english(answer):
        errors.append("Non-English content in answer")
```

**路由决策逻辑**:

```python
if score >= 8.0 and no_critical_errors:
    route = "pass"  # 直接通过
elif correction_count >= max_corrections:
    route = "stop"  # 停止修正
elif has_critical_errors or score < 6.0:
    route = "regenerate"  # 重新生成
else:
    route = "refine"  # 修正错误
```

#### Refine节点

**职责**: 修复非关键错误

**输入**: verification.errors, verification.revision_advice

**处理流程**:

1. 提取错误列表和修正建议
2. 构建修正提示词
3. 调用LLM进行修正
4. 后处理（同Generate节点）

#### Regenerate节点

**职责**: 重新生成回答（处理关键错误）

**输入**: verification.critical_errors

**处理流程**:

1. 提取关键错误列表
2. 在提示词中添加"已知问题"警告
3. 调用LLM重新生成
4. 后处理（同Generate节点）

#### Finalize节点

**职责**: 生成最终训练样本

**输出格式**:

```python
sample = {
    "question": str,
    "answer": str,  # 使用best_answer
    "category": "Basic-Physiological Conditions",
    "condition": str,
    "protein": str,
    "uniprot_id": str,
    "quality_score": float,
    "quality_log": {
        "last_route": str,
        "correction_count": int,
        "subscores": {...},
        "errors": [...],
        "critical_errors": [...]
    },
    "elapsed_seconds": float
}
```

## 3. 评分算法详解

### 3.1 综合评分计算

```python
# 基础事实性评分
if has_critical_errors:
    factuality_base = 0.0
elif entity_inconsistency:
    factuality_base = 1.0
elif missing_pmid:
    factuality_base = 2.0
elif many_llm_errors:
    factuality_base = 2.0
elif no_errors:
    factuality_base = 4.0
else:
    factuality_base = 3.0

# LLM事实性调整
if factuality_llm >= 3.5:
    factuality_adjust = +1.0
elif factuality_llm <= 1.5:
    factuality_adjust = -1.0
else:
    factuality_adjust = 0.0

factuality_effective = clamp(factuality_base + factuality_adjust, 0.0, 4.0)

# 综合评分
blended_score = (
    factuality_effective +
    min(grounding_llm, grounding_det) +
    min(format_llm, format_det) +
    consistency
)

# 错误惩罚
error_penalty = min(1.5, 0.25 * len(det_errors) + 0.15 * len(llm_errors))
blended_score -= error_penalty

# 关键错误额外惩罚
if has_critical_errors:
    blended_score = min(blended_score, 5.9)
```

### 3.2 各维度评分标准

**Factuality (0-4)**:

| 分数 | 含义 | 典型错误 |
|------|------|----------|
| 4 | 机制直接支持且内部一致 | 无 |
| 3 | 基本正确机制，轻微不精确 | 描述略宽泛 |
| 2 | 部分正确，缺少关键机制链接 | 机制方向正确但细节缺失 |
| 1 | 主要机制不匹配或大部分不支持 | 关键步骤缺失 |
| 0 | 核心机制错误或与证据矛盾 | 事实性错误 |

**Grounding (0-3)**:

| 分数 | 含义 |
|------|------|
| 3 | 所有PMID有效且直接支持 |
| 2 | PMID有效但部分不支持 |
| 1 | 无PMID但内容合理 |
| 0 | 存在不支持的PMID |

**Format (0-2)**:

| 分数 | 含义 |
|------|------|
| 2 | 所有5个章节完整 |
| 1 | 缺少1-2个章节 |
| 0 | 缺少3个以上章节 |

**Consistency (0-1)**:

| 分数 | 含义 |
|------|------|
| 1 | 蛋白名、UniProt ID一致 |
| 0 | 存在不一致 |

## 4. PubMed查询策略

### 4.1 查询词构建

```python
# 主查询
pubmed_query = (
    f"({protein}[Title/Abstract]) AND "
    f"(({gene}[Title/Abstract]) OR ({protein}[Title/Abstract])) AND "
    f"(physiology[Title/Abstract] OR function[Title/Abstract] OR "
    f"regulation[Title/Abstract] OR expression[Title/Abstract] OR "
    f"homeostasis[Title/Abstract] OR pathway[Title/Abstract] OR "
    f"mechanism[Title/Abstract])"
)
```

**查询词设计原则**:

1. **实体聚焦**: 确保蛋白质名称和基因名称在标题/摘要中
2. **功能导向**: 使用physiology、function等关键词筛选功能相关文献
3. **机制补充**: 包含mechanism、pathway等词捕获机制研究

### 4.2 PMID排名算法

```python
def rank_pmid(pmid: str) -> float:
    title = titles.get(pmid, "").lower()
    score = 0.0

    # 实体匹配
    for token in [protein.lower(), gene.lower()]:
        if token in title:
            score += 1.0

    # 关键词加分
    if "physiology" in title or "function" in title:
        score += 0.6
    if "regulation" in title or "expression" in title:
        score += 0.4
    if "homeostasis" in title or "pathway" in title:
        score += 0.3

    # 惩罚项
    if "review" in title:
        score -= 0.2

    return score
```

## 5. UniProt数据提取

### 5.1 提取的字段

| 字段 | 说明 | 提取位置 |
|------|------|----------|
| protein_name | 蛋白质全名 | proteinDescription.recommendedName.fullName |
| gene | 基因名 | genes[0].geneName |
| function | 功能描述 | comments(FUNCTION) |
| catalytic_activity | 催化活性 | comments(CATALYTIC ACTIVITY) |
| cofactor | 辅因子 | comments(COFACTOR) |
| ph_optimum | pH/温度最适值 | comments(BIOPHYSICOCHEMICAL) |
| tissue_specificity | 组织特异性 | comments(TISSUE SPECIFICITY) |
| subunit | 亚基结构 | comments(SUBUNIT) |
| induction | 诱导表达 | comments(INDUCTION) |
| pathway | 代谢/信号通路 | comments(PATHWAY) |
| active_sites | 活性位点 | features(ACTIVE_SITE, BINDING_SITE) |

### 5.2 数据提取示例

```python
# 催化活性提取
for comment in data.get("comments", []):
    if comment.get("commentType") == "CATALYTIC ACTIVITY":
        reaction = comment.get("reaction", {})
        reaction_name = reaction.get("name", "")
        if reaction_name:
            catalytic_activity = reaction_name
        else:
            texts = comment.get("texts", [])
            if texts:
                catalytic_activity = texts[0].get("value", "")
        break

# 辅因子提取
for comment in data.get("comments", []):
    if comment.get("commentType") == "COFACTOR":
        cofactors = comment.get("cofactors", [])
        names = []
        for c in cofactors:
            name = c.get("name", {})
            if isinstance(name, dict):
                names.append(name.get("value", ""))
        cofactor_info = "; ".join(names)
        break
```

## 6. 错误处理机制

### 6.1 错误分类

**Critical Errors (关键错误)**:
- 使用白名单外的PMID
- 存在不支持的残基位点详情
- 存在不支持的结合亲和力数值
- 事实性错误或矛盾

**Non-Critical Errors (一般错误)**:
- 缺少必需章节
- PMID重复引用
- 非英文内容
- 证据部分缺少PMID

### 6.2 修正策略

| 错误类型 | 处理策略 |
|----------|----------|
| 关键错误 | 重新生成(regenerate) |
| 一般错误(少) | 修正(refine) |
| 一般错误(多) | 重新生成(regenerate) |
| 无错误 | 直接通过(pass) |

### 6.3 修正次数控制

```python
max_corrections = 2  # 默认最大2次修正

if correction_count >= max_corrections:
    route = "stop"  # 停止，使用当前最佳答案
```

## 7. 性能优化

### 7.1 API调用优化

1. **批量获取**: 使用esearch一次获取多个PMID
2. **XML缓存**: 详细信息的XML只获取一次
3. **超时控制**: 所有API调用设置合理的超时时间

### 7.2 处理时间统计

典型处理时间（单条数据）:

| 阶段 | 平均耗时 |
|------|----------|
| UniProt获取 | 0.5-1s |
| PubMed查询 | 1-2s |
| XML详情获取 | 2-3s |
| LLM生成 | 5-15s |
| LLM验证 | 3-8s |
| 总计 | 15-40s |

### 7.3 并发考虑

当前实现为串行处理，未来可考虑：
- 多个pair并行处理
- Collector和Generate节点流水线并行

## 8. 数据质量保证

### 8.1 五层质量控制

1. **证据层**: 只使用UniProt和PubMed的权威数据
2. **引用层**: 白名单机制确保PMID真实性
3. **生成层**: 提示词约束确保格式正确
4. **验证层**: 多维度评分和错误检测
5. **修正层**: 自动修正循环提升质量

### 8.2 质量监控指标

- **通过率**: pass路由占比
- **修正效率**: 达到pass所需的平均修正次数
- **评分分布**: 所有输出的质量评分分布
- **错误类型统计**: 各类错误的出现频率

## 9. 配置参数

### 9.1 可配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_corrections | 2 | 最大修正次数 |
| pubmed_retmax | 30 | PubMed初始返回数量 |
| pmid_whitelist_size | 6 | 白名单PMID数量 |
| article_detail_count | 3 | 详细信息文章数量 |
| llm_temperature | 0.2 | LLM生成温度 |
| llm_max_tokens | 1500 | LLM最大token数 |

### 9.2 环境变量

```bash
export BIOREASON_BASE_URL="https://api.siliconflow.cn/v1"
export BIOREASON_MODEL="Qwen/Qwen2.5-7B-Instruct"
export BIOREASON_API_KEY="your-api-key"
```

## 10. 扩展指南

### 10.1 添加新的数据源

1. 在`fetch_data`函数中添加新的获取逻辑
2. 在`build_generation_user_prompt`中添加数据展示
3. 在提示词中添加相应的约束

### 10.2 自定义验证规则

1. 在`deterministic_checks`中添加新的检查项
2. 在`VERIFIER_PROMPT`中添加相应的评估标准
3. 在评分计算中添加新的权重

### 10.3 修改章节结构

1. 在`normalize_sections`的tags列表中修改
2. 在所有提示词中相应修改
3. 在输出格式中相应修改
