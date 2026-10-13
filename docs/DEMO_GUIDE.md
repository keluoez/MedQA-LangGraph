# MedQA-LangGraph 面试演示指南

> 多智能体生物医学 QA 数据集生成框架 · 一键启动 + 5 分钟演示剧本

## 1. 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 已在 3.11 验证 |
| Node.js | 18+ | 已在 22 验证 |
| API Key | 可选 | **mock 场景零密钥可完整演示**；仅 live 真实调用需要 |

## 2. 一键启动

```powershell
# Windows（项目根目录）
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1
```

```bash
# Linux / macOS
bash scripts/start_demo.sh
```

脚本幂等：已运行的端口自动跳过；首次启动后端需构建 22,571 条数据索引，约 5-10 秒。浏览器将自动打开 `http://localhost:5173`。

手动启动备选：

```bash
# 后端（终端 1，工作目录=项目根）
pip install -r api/requirements.txt
python -m uvicorn api.main:app --port 8000

# 前端（终端 2）
cd frontend && npm install && npm run dev
```

## 3. 五分钟演示剧本（Agent 岗位面试）

### 第 1 幕 · Tab1 数据集浏览（约 1.5 分钟）——讲"结果"

- Header 徽章开场：**22,571 对 · Gold 62.0% · 平均分 8.34/10**（与论文数字一致；病理模块 2026-10 评分口径校准后的正确数字）。
- 筛选演示：领域选"药物-靶点互作"+ Gold，出现 3,802 条；点击 Imatinib → BCR-ABL 样本。
- 详情抽屉讲三件事：
  1. **五段式结构化输出**（【核心结论】【蛋白信息】【条件】【临床意义】【证据】）——推理与引用物理隔离；
  2. **证据段 PMID 内联高亮**——每条 claim 绑定可验证文献；
  3. **质量日志四维子分数**（事实性 4/4、证据溯源 3/3…）——每条样本带完整审计记录，"↻ 2 次修正"标记说明闭环迭代过。

### 第 2 幕 · Tab2 多智能体流水线（约 2 分钟）——讲"机制"

- 场景选 `first_pass`（Imatinib-BCR-ABL，2x 速度，约 12 秒），点 **▶ 运行流水线**。
- 边运行边讲解（对着动画）：
  - **Collector（蓝）**：机制导向检索——UniProt 功能 + ChEMBL 机制 + PubMed 六篇候选 → **PMID 白名单**（这就是后面防幻觉引用的确定性边界）；
  - **Responder（绿）**：只准用证据包生成，五段式硬约束；
  - **Verifier（琥珀）**：双层审计——Python 规则检查（PMID 白名单硬校验、实体一致性）+ LLM 语义评分融合，取保守小值；
  - 路由决策条亮 **PASS** → 10.00 分出库。
- 指着底部日志讲时序：`INIT → COLLECT → RESPOND → VERIFY → ROUTE → FINAL`，"每个 Agent 的输入输出全程可审计"。

### 第 3 幕 · 核心话术（约 1.5 分钟）——讲"差异化"

> **这个系统和常见方案的区别在哪？**
> - **单 Agent（ReAct）**：运动员兼裁判，验证幻觉与生成幻觉同源，自己查不出自己；
> - **高自主多 Agent（AutoGPT 类）**：planner 自己决定下一步，灵活但不可控、不可复现——生物医学领域一次幻觉引用就是事故；
> - **纯 workflow（固定 DAG 编排）**：有分工但路由静态，低分样本照样流出，没有质量闭环；
> - **本系统**：**结构确定性 + 行为概率性**——图拓扑、路由阈值、PMID 白名单全部是确定性代码，只有节点内的生成是概率的。一句话：**别人做 agent 是给模型更多自由，我们做 agent 是给模型戴上手铐**。
> - 消融实验背书：去掉闭环修正 −3.95 分、去掉 PMID 白名单 −3.47、去掉结构约束 −2.67（三独立评估器，p < 10⁻⁶）。

### 收尾（可选 30 秒）

- 点击右上角 **EN** 切换英文界面（展示双语工程）。
- 从 Tab1 详情抽屉点 **"送入流水线重跑"** 演示 Tab 联动（live 模式需配好 .env 密钥）。

## 4. 修正循环场景（强烈建议录制）

当前内置 2 个单轮场景（first_pass / insufficient_evidence，零密可用）。**面试最大亮点是修正循环**——Erlotinib-EGFR 从 5.8 分 Bronze 迭代到 8.8 分 Gold 的完整轨迹。用真实管线录制一次（需密钥）：

```bash
# 1. 配置密钥
cp .env.example .env   # 填入 BIOREASON_API_KEY

# 2. 录制修正循环案例（约 2-3 分钟真实运行）
python scripts/record_mock_runs.py --scenario-id bronze_to_gold ^
  --description "Bronze to Gold: iterative correction loop" ^
  --pair-json "{\"drug\":\"Erlotinib\",\"protein\":\"EGFR\",\"uniprot_id\":\"P00533\",\"gene\":\"EGFR\",\"chembl_id\":\"CHEMBL248\"}"

# 3. 重启 API 后，Tab2 场景下拉即出现 bronze_to_gold
```

同理可录制 `refine_loop`（轻错误路径）。录制后文件在 `api/mock/recorded_runs/`，可提交进仓库永久复用。

## 5. 常见问题

| 现象 | 说明 |
|------|------|
| DevTools 里 POST /api/qa/run 结束时有红色 `ERR_ABORTED` | SSE 流正常关闭的日志噪音，不影响功能；演示时关闭 DevTools 即可 |
| 首次打开 Tab1 空白几秒 | 后端正在加载 22,571 条数据构建内存索引，第二次请求起即飞快 |
| live 模式报错 `缺少API密钥` | `.env` 未配置 `BIOREASON_API_KEY`，或未重启后端（重启后生效） |
| 端口被占用 | 脚本自动跳过已在运行的端口；手动管理用 `uvicorn --port 8001` 改端口（同步改 `frontend/vite.config.ts` 的 proxy target） |

## 6. API 速览（面试官可能问接口设计）

```
GET  /api/dataset/samples?page=&domain=&tier=&q=&sort=   分页浏览（22,571 条）
GET  /api/dataset/samples/{id}                           样本详情（含五段式+质量日志）
GET  /api/dataset/stats                                  全局/分域统计
GET  /api/qa/scenarios                                    mock 场景列表
GET  /api/qa/pairs?q=                                     预置交互对检索
POST /api/qa/run  {mode, scenario_id|pair, speed}         SSE 流水线（事件流）
```

交互文档（可在演示时展示）：后端启动后打开 `http://127.0.0.1:8000/docs`。
