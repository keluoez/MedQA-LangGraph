# BioCraft / MedQA-LangGraph

基于 LangGraph 的多智能体生物医学 QA 数据集生成框架，配套演示平台。**无需模型微调**，通过 Agent 编排消除生物医学数据生成中的幻觉。

## 核心成果

- **数据集**：22,571 对机制型 QA（十分制评分：平均 **8.34**，Gold ≥8.0 占 **62.0%**，Bronze <6.0 仅 4.8%）
- **四领域**：Drug-Target 7,993 / Experimental 4,997 / Pathological 4,988 / Physiological 4,593
- **对外对比**：Evidence Grounding 2.77/3.0，五个对照数据集（ReasonMed / BioGraphletQA / MedQuAD / medical-o1 / ChatDoctor）均低于 1.61
- **下游效用**：PubMedQA few-shot 比 zero-shot 平均 +10.2%，ReasonMed 示例反而 −3.8%
- **可靠性工程**：77/77 单元测试全绿，离线确定性运行

> 病理域历史分数已于 2026-10 校准（修复 12 分越界口径 bug），详见 [docs/DATA_CALIBRATION.md](docs/DATA_CALIBRATION.md)。

## 三大支柱

1. **Verifier-Refiner 闭环**：Generator → Verifier → Refiner/Regenerator → 再验证，非单次生成
2. **PMID 白名单硬校验**：Python 级约束——被引 PMID 必须存在于本地知识库，否则拒绝
3. **结构化输出约束**：强制 `【Core Conclusion】…【Evidence】…` 五段式，推理与引用物理隔离

## 架构

每个领域模块是一条 6 节点 LangGraph 管线：

```
Collector（证据检索）
   → Responder（结构化生成）
   → Verifier（LLM 四维语义分 + deterministic 规则分双轨）
   → 路由：pass → Finalizer | refine → Refiner | regenerate → Regenerator | stop
   → Finalizer（最佳样本落盘）
```

**检索算法**（`rank_pmid`）：token 池命中 +1，领域关键词加权（mechanism +0.6 等），review/resistance 负权降权；白名单 top-6，摘要详情 top-3。

**评分口径**（各域有差异，完整规格见前端「机制详解」页或 `/api/pipeline/introspect`）：

| 域 | 维度 | 融合方式 |
|----|------|----------|
| Drug / Experimental | factuality(0-4) + grounding(0-3) + format(0-2) + consistency(0-1) | min(LLM, det) 融合 + error_penalty |
| Pathological | mechanism(0-4) + disease(0-3) + evidence(0-2) + format(0-1) | LLM 四维直接求和 |
| Physiological | 四维同 drug | LLM 直加，无惩罚项 |

**结构化输出双层防线**（M5-B）：
- 第一层：`parse_json_object` 后 Pydantic 校验（`core/schemas.py`），格式幻觉降级为默认分，不再击穿样本
- 第二层：instructor 生成端约束（`core/llm.py: llm_call_json`），Strict 必填模型自动重问；开关 `BIOREASON_STRUCTURED_OUTPUT`

## 仓库结构

```
├── basic_drug_interaction/        # 药物-靶点域（参考实现）
├── basic_experimental_conditions/# 实验条件域
├── basic_pathological_environment/# 病理环境域（含 recalculate_scores.py 重算脚本）
├── basic_physiological_conditions/# 生理条件域
├── complex_combination_therapy/  # 复杂联合用药域
├── core/                         # 共享库：llm / batch / pubmed / http(缓存) / schemas / json_utils
├── api/                          # FastAPI 后端（演示层）
│   ├── routes/                   #   dataset / qa / mechanism 三组端点
│   └── adapters/                 #   pipeline_bridge（防腐层）/ mock_player / 机制元数据清单
├── frontend/                     # React + Vite 前端（DatasetTab / PipelineTab / MechanismTab）
├── tests/                        # 77 个 pytest 用例（离线确定性）
├── experiments/                  # 论文实验：baseline 对比 / 消融 / 下游效用
├── evaluation/                   # 外部评估与数据集对比
├── docs/                         # DEMO_GUIDE / DATA_CALIBRATION
└── PROJECT_DOCUMENTATION.md      # 完整项目文档
```

## 快速开始

### 环境要求

- Python 3.11+（依赖：langgraph、fastapi、uvicorn、pydantic 2.10+、instructor、openai、requests、python-dotenv）
- Node.js 18+（前端）

### 配置

复制环境变量（密钥不入库，`.env` 已在 `.gitignore`）：

```bash
# .env 示例 —— 密钥必填，无默认回退
BIOREASON_API_KEY=<你的 NVIDIA API key>
BIOREASON_BASE_URL=https://integrate.api.nvidia.com/v1
BIOREASON_MODEL=meta/llama-3.2-90b-vision-instruct
```

> 管线与骨干模型解耦：任何 OpenAI 兼容端点均可（环境变量覆盖即切换）。模型名以端点实际提供的为准。

### 安装

```bash
pip install -r requirements.txt
cd frontend && npm install
```

### 运行演示平台

```bash
# 后端（127.0.0.1:8000）
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# 前端（127.0.0.1:5173）
cd frontend && npm run dev
```

打开 http://localhost:5173 —— 无 API key 也可用（mock 离线回放模式）：
- **DatasetTab**：22,571 样本浏览 / 筛选 / 详情
- **PipelineTab**：QA 生成管线实时演示（mock 回放或 live 真跑）
- **MechanismTab**：机制详解——Agent 流程、检索步骤与 rank_pmid 权重、上下文真实拼接预览、评分规则、Prompt 全文

### 运行生成管线

```bash
cd basic_drug_interaction
python langgraph_pipeline.py --limit 5          # 小批量试跑
python langgraph_pipeline.py                   # 全量（断点续传，失败样本自动重试）
```

工程特性：SQLite 检索缓存（`.cache/`）、NCBI 3 req/s 令牌桶限流、断点续传（失败样本写 `*.failures.jsonl`）、线程池并发 + 文件锁写保护。

### 运行测试

```bash
python -m pytest tests/ -v     # 77/77 全绿，离线确定性（conftest 关闭 instructor 路径）
```

## API 一览

| 端点 | 说明 |
|------|------|
| `GET /api/dataset/stats` | 数据集统计（总数 / 均分 / Gold-Silver-Bronze，按域） |
| `GET /api/dataset/samples` | 样本分页浏览（筛选 / 排序） |
| `GET /api/qa/pairs` / `scenarios` | QA 演示用的交互对与预录场景 |
| `POST /api/qa/run` | QA 演示 SSE 流（mock 回放 / live 真跑） |
| `GET /api/pipeline/introspect` | 机制自省：运行时直读 4 域真实 prompt / 评分规则 / 检索算法元数据 |

## 文档索引

| 文档 | 内容 |
|------|------|
| [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) | 完整项目文档（架构 / 模块 / API / 技术细节） |
| [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md) | 五分钟演示剧本 |
| [docs/DATA_CALIBRATION.md](docs/DATA_CALIBRATION.md) | 病理域评分口径校准说明（2026-10） |
| [paper_modification_tracking.md](paper_modification_tracking.md) | 论文修改追踪（M1-M31） |
| [HANDOVER.md](HANDOVER.md) | 交接文档 |
