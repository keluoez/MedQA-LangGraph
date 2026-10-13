"""M5-A 机制自省 · 静态元数据清单。

纯数据文件：结构化描述 4 个领域管线的检索算法、上下文拼接、评分规则与路由策略。
不 import 任何后端领域模块 —— 运行时事实（prompt 全文 / MODEL / 真实拼接预览）
由 pipeline_bridge.introspect_pipeline() 延迟导入各模块并合并，因此 prompt 永远与代码一致。

文本口径：中文为主（演示主语言）。后端目录/规则重构时需同步本文件（防腐层约定）。
"""

# ── 全领域共享的 Agent 流程（LangGraph 节点拓扑） ─────────────────────────
AGENTS = [
    {
        "id": "collect",
        "role": "Collector",
        "desc": "证据检索：调用 UniProt / ChEMBL / PubMed 公开 API 组装证据包，产出问题模板与 PMID 白名单",
    },
    {
        "id": "generate",
        "role": "Responder",
        "desc": "结构化生成：按五段标签输出证据约束回答，仅可引用白名单 PMID",
    },
    {
        "id": "verify",
        "role": "Verifier",
        "desc": "双层审计：确定性规则检查（det）+ LLM 评分融合，产出总分与错误清单",
    },
    {
        "id": "refine",
        "role": "Refiner",
        "desc": "轻错误路径：携带错误清单与修改建议做字段级修订，回到 Verifier 复审",
    },
    {
        "id": "regenerate",
        "role": "Regenerator",
        "desc": "重错误路径：携带已知 critical 问题重启动推理，回到 Verifier 复审",
    },
    {
        "id": "finalize",
        "role": "Finalizer",
        "desc": "落盘：输出历史最佳答案 + 质量分 + 质量日志（子分数/错误/路由）",
    },
]

# ── 领域规格 ──────────────────────────────────────────────────────────────
# module / prompt_attrs / context_builder 仅供 pipeline_bridge 延迟导入使用；
# demo_pair / demo_evidence 是上下文预览的占位状态（不发起任何网络请求）。
_DOMAINS = {
    "drug": {
        "label": "Drug-Target Interaction",
        "module": "basic_drug_interaction.langgraph_pipeline",
        "prompt_attrs": {
            "responder": "RESPONDER_PROMPT",
            "verifier": "VERIFIER_PROMPT",
            "refiner": "REFINER_PROMPT",
        },
        "context_builder": "build_generation_user_prompt",
        "question_template": "What is the mechanism by which {drug} affects {protein}?",
        "sections": ["【Core Conclusion】", "【Protein Information】", "【Condition】", "【Clinical Significance】", "【Evidence】"],
        "tools": [
            {"name": "UniProt REST", "endpoint": "rest.uniprot.org/uniprotkb/{uniprot_id}.json", "note": "蛋白名 / 基因名 / 功能注释（FUNCTION）"},
            {"name": "ChEMBL", "endpoint": "www.ebi.ac.uk/chembl/api/data/…", "note": "药物记录 + mechanism_records（action_type / mechanism_of_action）"},
            {"name": "PubMed E-utilities", "endpoint": "eutils.ncbi.nlm.nih.gov (esearch / esummary / efetch)", "note": "经 core.pubmed：SQLite 检索缓存 + NCBI 3 req/s 令牌桶限流"},
        ],
        "retrieval": {
            "steps": [
                {"title": "组装 PubMed 布尔主查询", "detail": "(drug[Title/Abstract]) AND ((protein OR gene)[Title/Abstract]) AND (mechanism OR \"mode of action\" OR inhibit* OR bind*)，retmax=30"},
                {"title": "三级 fallback 降级", "detail": "候选 <3 篇时逐级放宽：① 去掉机制限定词 ② 自由词 \"drug protein mechanism\" ③ 换基因名重查"},
                {"title": "批量拉取标题", "detail": "fetch_pubmed_titles 一次取回全部候选标题，供打分使用"},
                {"title": "rank_pmid 标题打分排序", "detail": "token_pool={drug, protein, gene} 命中 +1；mechanism/mode of action +0.6；inhibit +0.4；bind +0.3；resistance −0.4；review −0.2"},
                {"title": "取白名单 top-6", "detail": "排序后前 6 个 PMID 作为 Responder 唯一可引用白名单，越界引用即判 critical"},
                {"title": "top-3 拉取摘要", "detail": "efetch 取前 3 篇 title+abstract（摘要截断 ≤1200 字符）注入证据包"},
            ],
            "rank_weights": [
                {"pattern": "drug / protein / gene 任一命中标题", "weight": 1.0},
                {"pattern": "标题含 mechanism / mode of action", "weight": 0.6},
                {"pattern": "标题含 inhibit / inhibition", "weight": 0.4},
                {"pattern": "标题含 bind / binding", "weight": 0.3},
                {"pattern": "标题含 resistance", "weight": -0.4},
                {"pattern": "标题含 review", "weight": -0.2},
            ],
            "whitelist_size": 6,
            "detail_articles": 3,
        },
        "context_blocks": [
            {"block": "Question", "source": "Collector 生成的问题模板"},
            {"block": "Drug / Protein / UniProt ID / DrugBank ID / ChEMBL ID", "source": "交互对元数据"},
            {"block": "UniProt protein name + function (≤600 字符)", "source": "Collector · UniProt"},
            {"block": "ChEMBL Drug Mechanism Records (≤3 条)", "source": "Collector · ChEMBL mechanism_records"},
            {"block": "Allowed PMIDs for citation", "source": "Collector · rank_pmid 白名单 top-6"},
            {"block": "PubMed Articles (title + abstract ≤900)", "source": "Collector · top-3 efetch"},
        ],
        "verifier_blocks": [
            {"block": "Question + 实体元数据 + Allowed PMIDs + UniProt function", "source": "状态与证据包"},
            {"block": "Answer", "source": "Responder 当前回答"},
        ],
        "scoring": {
            "dimensions": [
                {"key": "factuality", "label": "事实性", "range": "0-4", "source": "LLM 原始分 → det 错误定 base(0-4) → LLM 分 ±1 调整 → 钳制 [0,4]"},
                {"key": "grounding", "label": "证据溯源", "range": "0-3", "source": "min(LLM 分, det 分)：白名单外 PMID 直接 0，未引用 PMID 降为 1"},
                {"key": "format", "label": "格式", "range": "0-2", "source": "min(LLM 分, det 分)：缺失段标签按 2−缺失数扣减"},
                {"key": "consistency", "label": "一致性", "range": "0-1", "source": "min(LLM 分, det 分)：drug/UniProt/protein 实体名须全部出现"},
            ],
            "det_checks": [
                "五段标签完整性（缺失 → format 扣分）",
                "drug / UniProt ID / protein 实体名一致性（缺失 → consistency=0 + factuality base=1）",
                "PMID 白名单校验（越界 → grounding=0 + critical）",
                "Evidence 段缺 PMID（→ factuality base=2）",
                "残基位点 / IC50 / Ki / Kd / ATP-binding 声明无证据支持（→ critical）",
            ],
            "formula": "total = factuality_effective + min(grounding_llm, det) + min(format_llm, det) + min(consistency_llm, det) − error_penalty(0.25×det错误数 + 0.15×llm错误数, 封顶1.5)；存在 critical 时总分封顶 5.9",
            "routing": [
                {"route": "pass", "condition": "score ≥ 8 且无 critical"},
                {"route": "regenerate", "condition": "存在 critical 或 score < 6（修正次数未耗尽）"},
                {"route": "refine", "condition": "其余情况（6 ≤ score < 8）"},
                {"route": "stop", "condition": "修正次数 ≥ max_corrections（默认 2）"},
            ],
            "max_score": 10,
        },
        "demo_pair": {
            "drug": "Aspirin",
            "protein": "COX-2",
            "uniprot_id": "P23219",
            "gene": "PTGS2",
            "drugbank_id": "DB00945",
            "chembl_id": "CHEMBL135",
        },
        "demo_evidence": {
            "uniprot": {
                "protein_name": "Prostaglandin G/H synthase 2 (COX-2)",
                "gene": "PTGS2",
                "function": "Catalyzes the conversion of arachidonate to prostaglandin H2, a committed step in prostanoid biosynthesis…",
            },
            "chembl": {"pref_name": "ASPIRIN"},
            "chembl_mechanisms": [
                {
                    "action_type": "INHIBITOR",
                    "target_name": "Prostaglandin G/H synthase 2",
                    "mechanism_of_action": "Irreversible inhibition by acetylation of a serine residue near the active site",
                    "reference": "ChEMBL mechanism_records",
                }
            ],
            "pubmed_pmids": ["25277428", "12345678"],
            "pubmed_titles": {"25277428": "Aspirin acetylation of COX-2…"},
            "pubmed_articles": [
                {
                    "pmid": "25277428",
                    "title": "Aspirin acetylation of COX-2",
                    "journal": "Biochem J",
                    "year": "2014",
                    "abstract": "Aspirin selectively acetylates Ser530 of COX-2, blocking prostaglandin biosynthesis…",
                }
            ],
            "drugbank_id": "DB00945",
        },
    },
    "experimental": {
        "label": "Experimental Conditions",
        "module": "basic_experimental_conditions.langgraph_pipeline",
        "prompt_attrs": {
            "responder": "RESPONDER_PROMPT",
            "verifier": "VERIFIER_PROMPT",
            "refiner": "REFINER_PROMPT",
        },
        "context_builder": "build_generation_user_prompt",
        "question_template": "How does {condition} affect {protein} activity and function?",
        "sections": ["【Core Conclusion】", "【Protein Information】", "【Condition】", "【Clinical Relevance】", "【Evidence】"],
        "tools": [
            {"name": "UniProt REST", "endpoint": "rest.uniprot.org/uniprotkb/{uniprot_id}.json", "note": "蛋白名 / 功能 + 酶学字段：catalytic_activity / cofactor / pH optimum / active_sites"},
            {"name": "PubMed E-utilities", "endpoint": "eutils.ncbi.nlm.nih.gov (esearch / esummary / efetch)", "note": "经 core.pubmed：SQLite 检索缓存 + NCBI 3 req/s 令牌桶限流"},
        ],
        "retrieval": {
            "steps": [
                {"title": "组装 PubMed 布尔主查询", "detail": "(protein OR gene)[Title/Abstract] AND (condition[Title/Abstract]) AND (mechanism OR activity OR pH OR temperature OR denaturation OR inhibition OR catalytic OR \"active site\")，retmax=30"},
                {"title": "fallback 降级", "detail": "候选为空时放宽为自由词 \"protein condition mechanism activity\""},
                {"title": "批量拉取标题", "detail": "fetch_pubmed_titles 一次取回全部候选标题"},
                {"title": "rank_pmid 标题打分排序", "detail": "token_pool={protein, gene} + condition 中 ≥4 字符词，命中 +1；mechanism/activity +0.6；catalytic/active site +0.5；inhibit/denatur +0.4；pH/temperature +0.3；review −0.2"},
                {"title": "取白名单 top-6", "detail": "排序后前 6 个 PMID 作为唯一可引用白名单"},
                {"title": "top-3 拉取摘要", "detail": "efetch 取前 3 篇 title+abstract（摘要截断 ≤1200 字符）"},
            ],
            "rank_weights": [
                {"pattern": "protein / gene / condition 关键词命中标题", "weight": 1.0},
                {"pattern": "标题含 mechanism / activity", "weight": 0.6},
                {"pattern": "标题含 catalytic / active site", "weight": 0.5},
                {"pattern": "标题含 inhibit / denatur", "weight": 0.4},
                {"pattern": "标题含 pH / temperature（整词）", "weight": 0.3},
                {"pattern": "标题含 review", "weight": -0.2},
            ],
            "whitelist_size": 6,
            "detail_articles": 3,
        },
        "context_blocks": [
            {"block": "Question", "source": "Collector 生成的问题模板"},
            {"block": "Experimental Condition / Target Protein / UniProt ID", "source": "交互对元数据"},
            {"block": "UniProt protein name + function (≤600)", "source": "Collector · UniProt"},
            {"block": "UniProt catalytic activity / cofactor / pH optimum / active sites", "source": "Collector · UniProt 酶学注释"},
            {"block": "Allowed PMIDs for citation", "source": "Collector · rank_pmid 白名单 top-6"},
            {"block": "PubMed Articles (title + abstract ≤900)", "source": "Collector · top-3 efetch"},
        ],
        "verifier_blocks": [
            {"block": "Question + 实体元数据 + Allowed PMIDs + UniProt function", "source": "状态与证据包"},
            {"block": "Answer", "source": "Responder 当前回答"},
        ],
        "scoring": {
            "dimensions": [
                {"key": "factuality", "label": "事实性", "range": "0-4", "source": "LLM 原始分 → det 错误定 base(0-4) → LLM 分 ±1 调整 → 钳制 [0,4]"},
                {"key": "grounding", "label": "证据溯源", "range": "0-3", "source": "min(LLM 分, det 分)：白名单外 PMID 直接 0，未引用 PMID 降为 1"},
                {"key": "format", "label": "格式", "range": "0-2", "source": "min(LLM 分, det 分)：缺失段标签按 2−缺失数扣减"},
                {"key": "consistency", "label": "一致性", "range": "0-1", "source": "min(LLM 分, det 分)：实体名须全部出现"},
            ],
            "det_checks": [
                "五段标签完整性（缺失 → format 扣分）",
                "protein / UniProt ID 实体名一致性（缺失 → consistency=0 + factuality base=1）",
                "PMID 白名单校验（越界 → grounding=0 + critical）",
                "Evidence 段缺 PMID（→ factuality base=2）",
                "残基位点 / IC50 / Ki / Kd / ATP-binding 声明无证据支持（→ critical）",
            ],
            "formula": "total = factuality_effective + min(grounding_llm, det) + min(format_llm, det) + min(consistency_llm, det) − error_penalty(0.25×det错误数 + 0.15×llm错误数, 封顶1.5)；存在 critical 时总分封顶 5.9",
            "routing": [
                {"route": "pass", "condition": "score ≥ 8 且无 critical"},
                {"route": "regenerate", "condition": "存在 critical 或 score < 6（修正次数未耗尽）"},
                {"route": "refine", "condition": "其余情况（6 ≤ score < 8）"},
                {"route": "stop", "condition": "修正次数 ≥ max_corrections（默认 2）"},
            ],
            "max_score": 10,
        },
        "demo_pair": {
            "condition": "low pH",
            "protein": "Pepsin",
            "uniprot_id": "P00790",
        },
        "demo_evidence": {
            "uniprot": {
                "protein_name": "Pepsin A",
                "gene": "PGA3",
                "function": "Shows particularly broad specificity; preferentially cleaves peptide bonds between hydrophobic amino acids…",
                "catalytic_activity": "Hydrolysis of peptide bonds with broad specificity",
                "cofactor": "",
                "ph_optimum": "Optimal activity at pH 1.5-2.0",
                "active_sites": ["Asp32", "Asp215"],
            },
            "pubmed_pmids": ["12345678"],
            "pubmed_titles": {"12345678": "Pepsin activity and pH-dependent stability…"},
            "pubmed_articles": [
                {
                    "pmid": "12345678",
                    "title": "Pepsin activity and pH-dependent stability",
                    "journal": "Biochemistry",
                    "year": "1998",
                    "abstract": "Pepsin A is active in acidic conditions and irreversibly denatures above pH 6…",
                }
            ],
        },
    },
    "pathological": {
        "label": "Pathological Environment",
        "module": "basic_pathological_environment.langgraph_pipeline",
        "prompt_attrs": {
            "responder": "PATHOLOGY_RESPONDER_PROMPT",
            "verifier": "PATHOLOGY_VERIFIER_PROMPT",
            "refiner": "PATHOLOGY_REFINER_PROMPT",
        },
        "context_builder": "build_pathology_user_prompt",
        "question_template": "How does {pathology} in {disease} affect {protein} function?",
        "sections": ["【Core Conclusion】", "【Protein Information】", "【Condition】", "【Clinical Significance】", "【Evidence】"],
        "tools": [
            {"name": "UniProt REST", "endpoint": "rest.uniprot.org/uniprotkb/{uniprot_id}.json", "note": "PATHOLOGY / DISEASE 注释 + disease_associations（含 VARIANT/MUTAGENESIS 特征）+ keywords"},
            {"name": "PubMed E-utilities", "endpoint": "eutils.ncbi.nlm.nih.gov (esearch / esummary / efetch)", "note": "经 core.pubmed：SQLite 检索缓存 + NCBI 3 req/s 令牌桶限流"},
        ],
        "retrieval": {
            "steps": [
                {"title": "组装 PubMed 布尔主查询", "detail": "(disease[Title/Abstract]) AND ((protein OR gene)[Title/Abstract]) AND (mechanism OR \"mode of action\" OR pathophysiology OR pathogenesis OR phosphorylation OR mutation OR expression OR activation OR inhibition)，retmax=30"},
                {"title": "fallback 降级", "detail": "候选为空时放宽为自由词 \"disease protein mechanism\""},
                {"title": "批量拉取标题", "detail": "fetch_pubmed_titles 一次取回全部候选标题"},
                {"title": "rank_pmid 标题打分排序", "detail": "token_pool={disease, protein, gene} 命中 +1；mechanism/mode of action +0.6；pathophysiology/pathogenesis +0.5；phosphorylation +0.4；mutation +0.4；expression +0.3；activation/inhibition +0.3；review −0.4；resistance −0.3"},
                {"title": "取白名单 top-6", "detail": "排序后前 6 个 PMID 作为唯一可引用白名单"},
                {"title": "top-3 拉取摘要", "detail": "efetch 取前 3 篇 title+abstract（摘要截断 ≤800 字符）"},
            ],
            "rank_weights": [
                {"pattern": "disease / protein / gene 任一命中标题", "weight": 1.0},
                {"pattern": "标题含 mechanism / mode of action", "weight": 0.6},
                {"pattern": "标题含 pathophysiology / pathogenesis", "weight": 0.5},
                {"pattern": "标题含 phosphorylation / mutation", "weight": 0.4},
                {"pattern": "标题含 expression / activation / inhibition", "weight": 0.3},
                {"pattern": "标题含 review", "weight": -0.4},
                {"pattern": "标题含 resistance", "weight": -0.3},
            ],
            "whitelist_size": 6,
            "detail_articles": 3,
        },
        "context_blocks": [
            {"block": "生成指令：How does {pathology} in {disease} affect {protein} function", "source": "Collector 生成的问题模板"},
            {"block": "Disease / Pathology / Target Protein (UniProt) / Allowed PMIDs", "source": "交互对元数据 + 白名单"},
            {"block": "PubMed Articles (≤2 篇，title + Key finding ≤80 字符)", "source": "Collector · top-2 efetch 摘要截断"},
            {"block": "IMPORTANT: 只能引用白名单 PMID + 五段结构指令", "source": "Responder 约束提醒"},
        ],
        "verifier_blocks": [
            {"block": "Question + Disease / Pathology / Protein / UniProt ID + Allowed PMIDs + UniProt function", "source": "状态与证据包"},
            {"block": "Answer", "source": "Responder 当前回答"},
        ],
        "scoring": {
            "dimensions": [
                {"key": "pathology_mechanism", "label": "病理机制", "range": "0-4", "source": "LLM 原始分 → det 错误定 base(0-4) → LLM 分 ±1 调整 → 钳制 [0,4]（M4 修复 12 分越界）"},
                {"key": "disease_relevance", "label": "疾病相关性", "range": "0-3", "source": "LLM 评分：UniProt 疾病关联缺失时 ≤1"},
                {"key": "evidence_support", "label": "证据支撑", "range": "0-2", "source": "LLM 评分：无效 PMID 直接 0"},
                {"key": "format", "label": "格式", "range": "0-1", "source": "LLM 评分：缺段 / 内容重复 / 非英文直接 0"},
            ],
            "det_checks": [
                "五段标签完整性（SECTION_TAGS）",
                "disease / protein / UniProt ID 实体名一致性（→ factuality base=1）",
                "PMID 白名单校验（越界 → critical）",
                "Evidence 段缺 PMID（→ factuality base=2）",
                "det 明细记录于 quality_log.deterministic，不与 LLM 四维混加（M4 修复口径）",
            ],
            "formula": "total = pathology_mechanism_effective + disease_relevance + evidence_support + format（LLM 四维求和，天然 ≤10）",
            "routing": [
                {"route": "pass", "condition": "score ≥ 9，或 score ≥ 8 且无 critical"},
                {"route": "regenerate", "condition": "存在 critical 且 score < 7（修正次数未耗尽）"},
                {"route": "refine", "condition": "其余情况"},
                {"route": "stop", "condition": "修正次数 ≥ max_corrections（默认 2）"},
            ],
            "max_score": 10,
        },
        "demo_pair": {
            "disease": "Alzheimer's disease",
            "pathology": "amyloid-beta accumulation",
            "protein": "APP",
            "uniprot_id": "P05067",
        },
        "demo_evidence": {
            "uniprot": {
                "protein_name": "Amyloid-beta A4 protein",
                "function": "Functions as a cell surface receptor and performs physiological functions on the surface of neurons…",
            },
            "pubmed_pmids": ["3097456", "12345678"],
            "pubmed_titles": {"3097456": "Amyloid-beta accumulation and APP processing…"},
            "pubmed_articles": [
                {
                    "pmid": "3097456",
                    "title": "Amyloid-beta accumulation and APP processing",
                    "journal": "Nature",
                    "year": "1989",
                    "abstract": "Amyloid-beta peptides derived from APP proteolysis accumulate in Alzheimer's disease brains and disrupt neuronal function…",
                },
                {
                    "pmid": "12345678",
                    "title": "APP phosphorylation alters amyloidogenic processing",
                    "journal": "J Neurosci",
                    "year": "2005",
                    "abstract": "Phosphorylation of APP at Thr668 shifts processing toward the amyloidogenic pathway…",
                },
            ],
        },
    },
    "physiological": {
        "label": "Physiological Conditions",
        "module": "basic_physiological_conditions.langgraph_pipeline",
        "prompt_attrs": {
            "responder": "PHYSIOLOGY_RESPONDER_PROMPT",
            "verifier": "PHYSIOLOGY_VERIFIER_PROMPT",
            "refiner": "PHYSIOLOGY_REFINER_PROMPT",
        },
        "context_builder": "build_generation_user_prompt",
        "question_template": "What is the function of {protein} under {condition}?",
        "sections": ["【Core Conclusion】", "【Protein Information】", "【Condition】", "【Physiological Importance】", "【Evidence】"],
        "tools": [
            {"name": "UniProt REST", "endpoint": "rest.uniprot.org/uniprotkb/{uniprot_id}.json", "note": "扩展 10 字段：function / catalytic_activity / cofactor / biophysicochemical / tissue_specificity / subunit / induction / pathway 等"},
            {"name": "PubMed E-utilities", "endpoint": "eutils.ncbi.nlm.nih.gov (esearch / esummary / efetch)", "note": "经 core.pubmed：SQLite 检索缓存 + NCBI 3 req/s 令牌桶限流"},
        ],
        "retrieval": {
            "steps": [
                {"title": "条件关键词规范化", "detail": "_extract_condition_keywords：去掉括号补充，映射同义词（fasting→fasting, fed→postprandial, 37→temperature, 7.4→pH…）"},
                {"title": "组装 PubMed 布尔主查询", "detail": "(protein OR gene)[Title/Abstract] AND (condition 关键词 OR physiology/function/regulation/homeostasis/mechanism/pathway)，retmax=30"},
                {"title": "fallback 降级", "detail": "候选为空时放宽为自由词 \"protein condition function\""},
                {"title": "批量拉取标题 + rank_pmid 打分排序", "detail": "protein +1；gene +1；physiology/function +0.6；regulation/homeostasis +0.4；mechanism/pathway +0.3；review −0.2"},
                {"title": "取白名单 top-6", "detail": "排序后前 6 个 PMID 作为唯一可引用白名单"},
                {"title": "top-3 拉取摘要", "detail": "efetch 取前 3 篇 title+abstract（摘要截断 ≤1200 字符）"},
            ],
            "rank_weights": [
                {"pattern": "protein 命中标题", "weight": 1.0},
                {"pattern": "gene 命中标题", "weight": 1.0},
                {"pattern": "标题含 physiology / function", "weight": 0.6},
                {"pattern": "标题含 regulation / homeostasis", "weight": 0.4},
                {"pattern": "标题含 mechanism / pathway", "weight": 0.3},
                {"pattern": "标题含 review", "weight": -0.2},
            ],
            "whitelist_size": 6,
            "detail_articles": 3,
        },
        "context_blocks": [
            {"block": "Question", "source": "Collector 生成的问题模板"},
            {"block": "Protein / Physiological Condition / UniProt ID", "source": "交互对元数据"},
            {"block": "UniProt 扩展字段（10 项：function / catalytic_activity / cofactor / biophysicochemical / tissue_specificity / subunit / induction / pathway…）", "source": "Collector · UniProt"},
            {"block": "ALLOWED PMIDs", "source": "Collector · rank_pmid 白名单 top-6"},
            {"block": "PubMed Articles (title + abstract ≤700)", "source": "Collector · top-3 efetch"},
            {"block": "REMINDER：五段头标签 + 只可引用白名单 PMID", "source": "Responder 约束提醒"},
        ],
        "verifier_blocks": [
            {"block": "Question + Protein", "source": "状态元数据"},
            {"block": "Answer", "source": "Responder 当前回答"},
        ],
        "scoring": {
            "dimensions": [
                {"key": "factuality", "label": "事实性", "range": "0-4", "source": "LLM 直接评分（无 base/adjust 融合）"},
                {"key": "grounding", "label": "证据溯源", "range": "0-3", "source": "LLM 直接评分"},
                {"key": "format", "label": "格式", "range": "0-2", "source": "LLM 直接评分"},
                {"key": "consistency", "label": "一致性", "range": "0-1", "source": "LLM 直接评分"},
            ],
            "det_checks": [
                "五段标签完整性（PHYSIOLOGY_SECTIONS）",
                "PMID 白名单校验（越界 → critical）",
                "Evidence 段缺 PMID（→ det error）",
            ],
            "formula": "total = factuality + grounding + format + consistency（LLM 四维直接求和，≤10）",
            "routing": [
                {"route": "pass", "condition": "score ≥ 7 且无 critical"},
                {"route": "regenerate", "condition": "存在 critical（修正次数未耗尽）"},
                {"route": "refine", "condition": "其余情况（score < 7 且无 critical）"},
                {"route": "stop", "condition": "修正次数 ≥ max_corrections（默认 2）"},
            ],
            "max_score": 10,
        },
        "demo_pair": {
            "condition": "fasting",
            "protein": "Glycogen phosphorylase",
            "uniprot_id": "P11216",
        },
        "demo_evidence": {
            "uniprot": {
                "protein_name": "Glycogen phosphorylase, liver form",
                "gene": "PYGL",
                "function": "Phosphorylase is an important allosteric enzyme involved in glycogen catabolism…",
                "catalytic_activity": "Glycogen + phosphate = D-glucose 1-phosphate",
                "cofactor": "Pyridoxal phosphate",
                "biophysicochemical": "Activated by phosphorylation; inactive in the dephosphorylated form",
                "tissue_specificity": "Liver",
                "subunit": "",
                "induction": "",
                "pathway": "Glycogen metabolism",
            },
            "pubmed_pmids": ["12345678"],
            "pubmed_titles": {"12345678": "Hormonal regulation of glycogen phosphorylase…"},
            "pubmed_articles": [
                {
                    "pmid": "12345678",
                    "title": "Hormonal regulation of glycogen phosphorylase during fasting",
                    "journal": "J Biol Chem",
                    "year": "1990",
                    "abstract": "Glucagon elevates cAMP, activating phosphorylase kinase and converting phosphorylase b to active phosphorylase a…",
                }
            ],
        },
    },
}


def domain_specs() -> dict:
    """返回领域规格（浅拷贝顶层，防调用方意外改写模块级常量）。"""
    return dict(_DOMAINS)


def agents_flow() -> list:
    return list(AGENTS)
