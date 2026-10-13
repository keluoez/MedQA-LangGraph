# 病理环境模块LangGraph实现 - 技术细节文档

## 概述

本文档详细记录了病理环境模块LangGraph实现中各个智能体的prompt设计、处理逻辑和技术细节。

## 1. 整体架构设计

### 1.1 状态机流程
```
Collector → Responder → Verifier → (Refiner/Regenerate循环) → Finalize
```

路由决策：
- 总分 ≥ 9.0 → pass（直接输出）
- 总分 ≥ 8.0 且无关键错误 → pass
- 修正次数 ≥ max_corrections → stop
- 有关键错误且总分 < 7.0 → regenerate
- 其他 → refine

### 1.2 状态结构
```python
class PathologyState(TypedDict, total=False):
    pair: Dict[str, Any]           # 疾病-病理-蛋白三元组
    question: str                   # 生成的问题
    evidence: Dict[str, Any]       # 收集的证据包（含白名单PMID）
    answer: str                    # 生成的回答
    best_answer: str               # 最佳回答（历史最优）
    best_score: float              # 最佳得分
    verification: Dict[str, Any]    # 验证结果
    route: Literal["pass", "refine", "regenerate", "stop"]  # 路由决策
    correction_count: int          # 修正次数
    max_corrections: int           # 最大修正次数（默认2）
    final_sample: Dict[str, Any]   # 最终样本
    elapsed_seconds: float         # 耗时
```

## 2. Collector智能体 - 数据收集

### 2.1 核心功能
- 从UniProt获取蛋白功能和病理关联信息
- 从PubMed获取病理机制相关文献
- 构建PMID白名单和结构化的证据包

### 2.2 数据源策略

#### UniProt数据收集
```python
def fetch_uniprot_pathology(uniprot_id: str) -> Dict[str, Any]:
    # 获取的关键信息：
    # - 蛋白名称和基因信息
    # - 功能描述 (FUNCTION)
    # - 通路信息 (PATHWAY) 
    # - 病理注释 (PATHOLOGY)
    # - 疾病关联 (DISEASE) - 多个来源：
    #   - PATHOLOGY注释中的疾病信息
    #   - DISEASE注释中的疾病关联
    #   - KEYWORDS中的疾病相关标签
    # - 关键词标签
```

#### PubMed智能查询策略（基于蛋白-药物模块）

**主查询：布尔逻辑精确检索**
```python
pubmed_query = (
    f"({disease}[Title/Abstract]) AND "
    f"(({protein}[Title/Abstract]) OR ({gene}[Title/Abstract])) AND "
    f"(mechanism[Title/Abstract] OR \"mode of action\"[Title/Abstract] OR "
    f"pathophysiology[Title/Abstract] OR pathogenesis[Title/Abstract] OR "
    f"phosphorylation[Title/Abstract] OR mutation[Title/Abstract] OR "
    f"expression[Title/Abstract] OR activation[Title/Abstract] OR "
    f"inhibition[Title/Abstract])"
)
```

**备用查询：简化关键词**
```python
fallback_query = f"{disease} {protein} mechanism"
```

**查询优化策略**：
- **布尔逻辑**：使用AND/OR精确检索，提高相关性
- **备用策略**：主查询无结果时自动切换简化查询
- **超时控制**：设置合理超时时间，避免卡死
- **候选扩展**：retmax=30获取候选，再排序筛选

#### PMID智能排序算法（基于蛋白-药物模块）

```python
def rank_pmid(pmid: str) -> float:
    title = (titles.get(pmid, "") or "").lower()
    score = 0.0
    
    # 核心关键词匹配
    for t in token_pool:  # {disease, protein, gene}
        if t in title:
            score += 1.0
    
    # 机制关键词加权
    if "mechanism" in title or "mode of action" in title:
        score += 0.6
    if "pathophysiology" in title or "pathogenesis" in title:
        score += 0.5
    if "phosphorylation" in title:
        score += 0.4
    if "mutation" in title:
        score += 0.4
    if "expression" in title:
        score += 0.3
    if "activation" in title or "inhibition" in title:
        score += 0.3
    
    # 负面关键词降权
    if "review" in title:
        score -= 0.4
    if "resistance" in title:
        score -= 0.3
    
    return score
```

#### 白名单策略（基于蛋白-药物模块）

```python
# 排序后选择最相关的PMID
ranked_pmids = sorted(candidate_pmids, key=rank_pmid, reverse=True)

# 白名单：6个PMID（用于验证引用合法性）
allowed_pmids = ranked_pmids[:6]

# 详细内容：只获取前3篇的摘要（提高效率）
articles = []
for pmid in allowed_pmids[:3]:
    xml_text = fetch_pubmed_article_xml(pmid)
    # 获取标题、摘要（限制800字）、期刊、年份
    articles.append(article)

# 返回白名单和详细内容
return {
    "allowed_pmids": allowed_pmids,
    "articles": articles
}
```

### 2.3 证据包结构
```python
evidence = {
    "uniprot": {
        "status": "ok",
        "protein_name": "Cellular tumor antigen p53",
        "gene": "TP53",
        "function": "Multifunctional transcription factor...",
        "pathway": "Cell cycle arrest, DNA repair...",
        "disease_associations": ["Cancer", "Li-Fraumeni syndrome"]
    },
    "pubmed_pmids": ["41934032", "41919885", "38971234", ...],  # 白名单PMID（6个）
    "pubmed_articles": [  # 详细内容（3篇）
        {
            "pmid": "41934032",
            "title": "TP53 phosphorylation in DNA damage response",
            "abstract": "...",  # 限制800字
            "journal": "Nature",
            "year": "2024"
        }
    ],
    "disease": "Cancer",
    "pathology": "DNA damage",
    "protein": "TP53",
    "gene": "TP53"
}
```

## 3. Responder智能体 - 回答生成

### 3.1 Prompt设计哲学

**核心原则**：
1. **证据约束**：只能基于提供的证据包生成内容
2. **白名单约束**：只能引用白名单内的PMID
3. **机制聚焦**：专注于病理条件对蛋白功能的影响
4. **结构化输出**：强制使用标准化的五段式格式
5. **简洁性**：避免冗长描述，保持训练集风格

### 3.2 完整Prompt

```python
PATHOLOGY_RESPONDER_PROMPT = """You are a biomedical domain writer creating high-quality training samples about pathological conditions.
You must answer based only on the provided evidence package.
Do not add facts not supported by provided evidence.

Output must keep this structure:
【Core Conclusion】...
【Protein Information】...
【Condition】...
【Clinical Significance】...
【Evidence】...

Core Conclusion requirements:
- Explain ONLY how the pathological condition affects the target protein's activity or function.
- Include the molecular mechanism if supported (e.g., altered expression, modified activity, disrupted function).
- Focus on the protein's role in the disease context.
- Avoid background, definitions, and broad disease descriptions in this section.

Other constraints:
1) Keep answers concise and focused on the pathology-protein relationship.
2) Evidence must cite UniProt ID and 1-3 relevant PMIDs from the provided list.
3) Each PMID cited must correspond to at least one concrete mechanism statement.
4) Always include the UniProt ID and protein name in 【Protein Information】.
5) If evidence is insufficient, explicitly state uncertainty instead of inventing details.
6) English only.
"""
```

### 3.3 用户提示构建

```python
# 构建证据文本（含白名单约束）
allowed_pmids = evidence.get("pubmed_pmids", [])

evidence_text = f"""Disease: {disease}
Pathology: {pathology}
Target Protein: {protein} (UniProt: {uniprot_id})
Allowed PMIDs: {', '.join(allowed_pmids)}

PubMed Articles (focus on quality, use 1-2 most relevant):
"""

# 只使用最相关的1-2篇文献
for i, article in enumerate(pubmed_articles[:2], 1):
    evidence_text += f"\n{i}. PMID: {article.get('pmid', '')}"
    evidence_text += f"\n   Title: {article.get('title', '')}"
    if article.get("abstract"):
        abstract = article.get('abstract', '')
        if len(abstract) > 80:
            evidence_text += f"\n   Key finding: {abstract[:80]}..."
        else:
            evidence_text += f"\n   Key finding: {abstract}"

# 白名单约束提示
evidence_text += f"\n\nIMPORTANT: You can only cite PMIDs from the allowed list above."

user_prompt = f"""Generate a high-quality answer about how {pathology} in {disease} affects {protein} function.

Evidence Package:
{evidence_text}

Answer based ONLY on the provided evidence.

Structure your answer as follows:
- 【Core Conclusion】: Brief summary (1-2 sentences)
- 【Protein Information】: UniProt ID and protein name only
- 【Condition】: Brief description of the pathological condition
- 【Clinical Significance】: Clinical relevance
- 【Evidence】: Detailed logical chain with 1-2 PMIDs

Keep the answer concise and focused."""
```

### 3.4 生成参数设置
- **Temperature**: 0.4（平衡创造性和准确性）
- **Max Tokens**: 1800（控制回答长度）
- **Timeout**: 45秒（避免API超时）

## 4. Verifier智能体 - 质量验证

### 4.1 分层验证体系（确定性检查 + LLM评分）

#### 4.1.1 确定性检查（deterministic_checks）

```python
def deterministic_checks(state: PathologyState) -> Dict[str, Any]:
    pair = state["pair"]
    answer = state.get("answer", "")
    evidence = state.get("evidence", {})
    
    # 1. 必填标签检查
    required_tags = ["【Core Conclusion】", "【Protein Information】", 
                     "【Condition】", "【Clinical Significance】", "【Evidence】"]
    missing_tags = [tag for tag in required_tags if tag not in answer]
    
    # 2. 关键信息一致性检查
    errors = []
    if pair["disease"].lower() not in answer.lower():
        errors.append("disease_name_missing_or_inconsistent")
    if pair["uniprot_id"].lower() not in answer.lower():
        errors.append("uniprot_id_missing_or_inconsistent")
    if pair["protein"].lower() not in answer.lower():
        errors.append("protein_name_missing_or_inconsistent")
    
    # 3. PMID白名单验证（核心机制）
    allowed_pmids = set(evidence.get("pubmed_pmids", []))
    pmids_in_answer = re.findall(r"PMID\s*:\s*(\d+)", answer, re.IGNORECASE)
    
    unsupported_pmids = [p for p in pmids_in_answer if p not in allowed_pmids]
    if unsupported_pmids:
        critical_errors.append(f"unsupported_pmids:{','.join(sorted(set(unsupported_pmids)))}")
    
    # 4. PMID缺失检查
    if allowed_pmids and len(pmids_in_answer) == 0:
        errors.append("evidence_section_missing_pmid")
    
    # 5. PMID重复检查
    if len(pmids_in_answer) != len(set(pmids_in_answer)):
        errors.append("duplicate_pmids_in_evidence")
    
    # 6. 计算确定性检查分数
    format_score = 2 if not missing_tags else max(0, 2 - len(missing_tags))
    grounding_score = 3 if not unsupported_pmids else (1 if not pmids_in_answer else 0)
    consistency_score = 1 if key_info_consistent else 0
    
    return {
        "det_errors": errors,
        "det_critical_errors": critical_errors,
        "format_score": format_score,
        "grounding_score": grounding_score,
        "consistency_score": consistency_score,
        "pmids_in_answer": pmids_in_answer,
        "allowed_pmids": list(allowed_pmids)
    }
```

#### 4.1.2 LLM评分维度

**病理机制准确性 (0-4分)**：
- **4分**：机制直接支持且内部一致，含分子细节
- **3分**：机制基本正确，有轻微不精确
- **2分**：方向正确但关键机制链接未支持
- **1分**：主要机制不匹配或大部分声明未支持
- **0分**：核心机制错误或与证据矛盾

**疾病关联性 (0-3分)**：
- **3分**：强疾病关联性，有清晰的病理背景
- **2分**：中等疾病关联性，有具体连接
- **1分**：弱疾病关联性或通用声明
- **0分**：无疾病关联性或错误疾病背景

**证据支撑 (0-2分)**：
- **2分**：有充分的PMID和UniProt支撑
- **1分**：有基本证据支撑
- **0分**：缺乏证据支撑

**格式合规性 (0-2分)**：
- **2分**：完全符合结构化格式要求
- **0分**：格式不符合要求

### 4.2 完整Verifier Prompt

```python
PATHOLOGY_VERIFIER_PROMPT = """You are a strict biomedical quality auditor for pathological condition answers.
Score the answer using only the given question, pair metadata and evidence package.
Return JSON only:
{
  "pathology_mechanism_score": 0-4,
  "disease_relevance_score": 0-3,
  "evidence_support_score": 0-2,
  "format_score": 0-1,
  "errors": ["..."],
  "critical_errors": ["..."],
  "revision_advice": ["..."]
}

Scoring rubric (STRICT MODE - errors must impact scores):
- Pathology mechanism score (0-4):
  4 = Clear, specific mechanism with molecular details (e.g., phosphorylation, mutation sites)
  3 = Correct mechanism but lacks molecular specificity
  2 = Vague or incomplete mechanism
  1 = Incorrect or missing mechanism
  0 = No mechanism described

- Disease relevance score (0-3):
  3 = Strong, specific disease relevance with context
  2 = Moderate relevance but could be more specific
  1 = Weak or generic disease relevance
  0 = No disease relevance

- Evidence support score (0-2):
  2 = Strong evidence support with valid, specific PMIDs
  1 = Some evidence but with issues (e.g., invalid PMIDs)
  0 = No or insufficient evidence

- Format score (0-1):
  1 = Proper structure with all required sections, consistent format
  0 = Missing sections, inconsistent format, or content duplication

CRITICAL ERRORS (must reduce corresponding scores):
- Invalid PMIDs → evidence_support_score = 0
- Missing required sections → format_score = 0
- Mechanism contradicts evidence → pathology_mechanism_score ≤ 2
- Content duplication → format_score = 0
- Non-English content → format_score = 0
- Missing UniProt disease associations → disease_relevance_score ≤ 1
- Missing molecular mechanism details → pathology_mechanism_score ≤ 2

SCORING MUST REFLECT ERRORS: If errors are identified, corresponding scores must be reduced.

SPECIFIC ERROR CHECKLIST:
1. Check for non-English content (Chinese characters, etc.)
2. Verify Protein Information includes UniProt disease associations
3. Verify Core Conclusion includes specific molecular mechanisms
4. Check for content duplication between sections
5. Validate all PMIDs are 7-8 digit numbers
"""
```

### 4.3 分层评分系统（基于蛋白-药物模块）

```python
# 关键错误分类
critical_markers = (
    "unsupported", "wrong target", "wrong disease", "contradiction",
    "inconsistent id", "fabricated", "halluc", "not in the allowed",
    "not included in the allowed", "evidence does not support"
)

# 基础事实性评分（确定性检查驱动）
if all_critical:
    pathology_mechanism_base = 0.0
elif key_info_inconsistent:
    pathology_mechanism_base = 1.0
elif "evidence_section_missing_pmid" in det_errors:
    pathology_mechanism_base = 2.0
elif len(llm_errors) >= 3:
    pathology_mechanism_base = 2.0
elif no_errors_at_all:
    pathology_mechanism_base = 4.0
else:
    pathology_mechanism_base = 3.0

# LLM评分调整
if pathology_mechanism_llm >= 3.5:
    pathology_mechanism_adjust = +1.0
elif pathology_mechanism_llm <= 1.5:
    pathology_mechanism_adjust = -1.0
else:
    pathology_mechanism_adjust = 0.0

# 最终病理机制得分
pathology_mechanism_effective = max(0.0, base + adjust)

# 总分计算
total_score = (
    pathology_mechanism_effective +
    disease_relevance +
    evidence_support +
    det_checks["format_score"] +
    det_checks["consistency_score"]
)
```

### 4.4 路由决策逻辑

```python
# 路由逻辑（优先级从高到低）
if total_score >= 9.0:
    route = "pass"
elif total_score >= 8.0 and len(all_critical) == 0:
    route = "pass"
elif correction_count >= max_corrections:  # 默认2次
    route = "stop"
elif len(all_critical) > 0 and total_score < 7.0:
    route = "regenerate"
else:
    route = "refine"
```

## 5. Refiner智能体 - 修正处理

### 5.1 修正Prompt

```python
PATHOLOGY_REFINER_PROMPT = """You are a biomedical editor for pathological condition answers.
Revise the draft answer by fixing listed errors.
Keep all five required sections with the same tags.
Stay strictly within provided evidence package.
Prefer concise training-style scientific writing.
English only.
"""
```

### 5.2 修正逻辑

```python
def node_refine(state: PathologyState) -> PathologyState:
    verification = state["verification"]
    answer = state["answer"]
    pair = state["pair"]
    
    correction_count = state.get("correction_count", 0) + 1
    state["correction_count"] = correction_count
    
    errors = verification.get("errors", [])
    critical_errors = verification.get("critical_errors", [])
    revision_advice = verification.get("revision_advice", [])
    
    user_prompt = f"""Disease: {pair['disease']}
Pathology: {pair['pathology']}
Protein: {pair['protein']}

Errors to fix:
{chr(10).join(f'- {e}' for e in errors + critical_errors)}

Revision advice:
{chr(10).join(f'- {a}' for a in revision_advice)}

Original Answer:
{answer}

Fix the errors while maintaining the required format."""
    
    corrected = llm_call(client, PATHOLOGY_REFINER_PROMPT, user_prompt, temperature=0.2)
    
    if "【Core Conclusion】" in corrected:
        state["answer"] = corrected
    
    return state
```

## 6. Regenerate智能体 - 重新生成

### 6.1 触发条件
- 有关键错误且总分 < 7.0
- 核心病理机制错误
- 缺乏证据支撑

### 6.2 重新生成逻辑

```python
def node_regenerate(state: PathologyState) -> PathologyState:
    verification = state.get("verification", {})
    pair = state["pair"]
    evidence = state["evidence"]
    
    critical_errors = verification.get("critical_errors", [])
    
    # 构建用户提示（包含已知关键问题）
    user_prompt = f"""Generate a high-quality answer about how {pathology} in {disease} affects {protein} function.

Evidence Package:
{evidence_text}

Known critical issues to avoid:
""" + "\n".join(f"- {x}" for x in critical_errors)
    
    regenerated = llm_call(client, PATHOLOGY_RESPONDER_PROMPT, user_prompt, temperature=0.15)
    
    regenerated = normalize_sections(regenerated)
    regenerated = normalize_evidence_citations(regenerated)
    
    state["answer"] = regenerated
    return state
```

## 7. Finalize智能体 - 输出处理

### 7.1 格式标准化

#### 段落重排
```python
def normalize_sections(answer: str) -> str:
    tags = ["【Core Conclusion】", "【Protein Information】", "【Condition】", 
            "【Clinical Significance】", "【Evidence】"]
    chunks: List[str] = []
    
    for i, tag in enumerate(tags):
        start = answer.find(tag)
        if start == -1:
            continue
        end_candidates = [answer.find(t, start + len(tag)) for t in tags[i + 1:]]
        end_candidates = [x for x in end_candidates if x != -1]
        end = min(end_candidates) if end_candidates else len(answer)
        
        piece = answer[start:end].strip()
        if piece:
            chunks.append(piece)
    
    return "\n\n".join(chunks).strip()
```

#### 非英文内容过滤
```python
def filter_non_english_content(text: str) -> str:
    # 移除中文字符
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    # 移除其他非ASCII字符
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

#### 证据引用标准化
```python
def normalize_evidence_citations(answer: str) -> str:
    # 统一PMID格式
    answer = re.sub(r"PubMed\s*:\s*(\d+)", r"PMID:\1", answer, flags=re.IGNORECASE)
    # ... 去重和重构证据块
```

### 7.2 最终样本结构

```python
final_sample = {
    "question": "How does DNA damage in Cancer affect TP53 function?",
    "answer": "【Core Conclusion】...\n\n【Protein Information】TP53 (UniProt: P04637)\n\n【Condition】...\n\n【Clinical Significance】...\n\n【Evidence】...(PMID: 41934032)",
    "category": "Basic-Pathological Environment",
    "disease": "Cancer",
    "pathology": "DNA damage",
    "protein": "TP53",
    "uniprot_id": "P04637",
    "quality_score": 9.0,
    "quality_log": {
        "pathology_mechanism_score": 4.0,
        "disease_relevance_score": 3.0,
        "evidence_support_score": 2.0,
        "format_score": 2,
        "errors": [],
        "critical_errors": [],
        "revision_advice": [],
        "subscores": {
            "pathology_mechanism_base": 4.0,
            "pathology_mechanism_llm_raw": 4.0,
            "pathology_mechanism_effective": 4.0,
            "grounding_llm_raw": 3.0,
            "format_llm_raw": 1.0,
            "consistency_llm_raw": 2.0
        }
    },
    "elapsed_seconds": 27.8
}
```

## 8. 关键技术优化

### 8.1 PMID白名单机制（基于蛋白-药物模块）

**设计目的**：杜绝LLM编造不存在的PMID

**实现流程**：
1. Collector收集6个最相关PMID作为白名单
2. 只获取前3篇的详细摘要（效率优化）
3. Responder只能引用白名单内的PMID
4. Verifier检查引用的PMID是否在白名单内
5. 白名单外的PMID被标记为关键错误（unsupported_pmids）

**证据包结构**：
```python
evidence = {
    "pubmed_pmids": ["pmid1", "pmid2", ...],  # 白名单（6个）
    "pubmed_articles": [{...}, {...}, {...}],   # 详细内容（3篇）
}
```

### 8.2 分层评分系统（基于蛋白-药物模块）

**设计目的**：确保评分一致性，避免验证和最终化评分不一致

**实现方式**：
1. 确定性检查计算基础分（format_score, consistency_score）
2. LLM评分提供参考分
3. 基础事实性评分由确定性检查驱动
4. LLM评分作为调整因子（±1.0）
5. Finalize使用与Verifier相同的评分逻辑

### 8.3 非英文内容过滤

**设计目的**：确保生成内容为纯英文

**实现方式**：
- 正则表达式移除中文字符（\u4e00-\u9fff）
- 移除非ASCII字符
- Prompt中明确要求"English only"

### 8.4 性能优化

#### PubMed查询优化
- **布尔逻辑查询**：提高检索精确度
- **备用查询策略**：主查询无结果时自动切换
- **PMID排序筛选**：30个候选 → 排序 → 取前6个白名单
- **摘要限制**：800字上限，避免过长
- **只获取3篇详细内容**：减少API调用

#### 内存优化
- **状态管理**：LangGraph自动管理状态内存
- **分批处理**：大文件处理时建议分批

## 9. 调试和故障排除

### 9.1 常见问题

#### API调用问题
```python
# 检查API密钥
api_key = resolve_api_key()
if not api_key:
    raise RuntimeError("缺少API密钥")
```

#### PubMed超时
- 自动重试机制
- 备用查询策略（简化关键词）
- 增加超时时间

#### unsupported_pmids关键错误
- 回答引用了白名单外的PMID
- 检查Collector是否正确收集了白名单
- 确认Responder只使用白名单内的PMID

#### 评分不一致
- 确保Finalize使用与Verifier相同的评分逻辑
- 检查分层评分系统是否正确计算

### 9.2 调试模式

```python
# Verifier调试日志
print(f"  [DEBUG] 修正计数: {correction_count}/{max_corrections}")
print(f"  [DEBUG] 关键错误: {all_critical}")
print(f"  [DEBUG] 一般错误: {all_errors[:3]}")
```

## 10. 扩展开发指南

### 10.1 添加新的智能体

```python
def node_custom_agent(state: PathologyState) -> PathologyState:
    """自定义智能体"""
    # 实现自定义逻辑
    return state

# 在图中添加节点
workflow.add_node("custom", node_custom_agent)
workflow.add_edge("previous_node", "custom")
workflow.add_edge("custom", "next_node")
```

### 10.2 修改验证规则

```python
def custom_verification_rules(state: PathologyState) -> Dict[str, Any]:
    """自定义验证规则"""
    # 实现自定义验证逻辑
    return {
        "custom_score": 0-5,
        "custom_errors": []
    }
```

---

*最后更新: 2026-04-11*
*文档版本: v2.0*
