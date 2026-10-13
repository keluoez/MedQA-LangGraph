# MedQA-LangGraph — Reproducible Experiments

This directory contains **fully runnable** code for all experiments reported in the paper.
Every number in every table can be reproduced by following the instructions below.

```
experiments/
├── baseline_comparison/        # Table II: vs. Vanilla LLM / Standard RAG / Iterative RAG
│   ├── 01_vanilla_llm/         # Baseline 1 — direct LLM generation, no retrieval
│   ├── 02_standard_rag/        # Baseline 2 — single-pass RAG, no verification
│   ├── 03_iterative_rag/       # Baseline 3 — multi-pass RAG, no closed-loop correction
│   ├── 04_evaluate_all/        # Score all outputs with 3 evaluators (DeepSeek/Qwen/GPT-4o)
│   └── README.md
│
├── ablation_study/             # Table IV: component ablation on 200 samples
│   ├── 00_sample_200/          # Shared: randomly sample 200 pairs from Gold data
│   ├── 01_full_system/         # Full pipeline (reference)
│   ├── 02_wo_retrieval/        # w/o Retrieval (= Vanilla LLM)
│   ├── 03_wo_correction_loop/  # w/o Correction Loop (single-pass verify, no refine)
│   ├── 04_wo_struct_constraint/# w/o Structural Constraint (free-form generation)
│   ├── 05_wo_pmid_whitelist/   # w/o PMID Whitelist (semantic-only verification)
│   ├── 06_evaluate_all/        # Score all 5 variants with DeepSeek evaluator
│   └── README.md
│
└── README.md                   # ← this file
```

## Quick Start

```bash
# 1. Set your API keys
export BIOREASON_API_KEY="your-nvidia-api-key"
export SILICONFLOW_API_KEY="your-siliconflow-key"   # for DeepSeek/Qwen eval
export OPENAI_API_KEY="your-openai-key"             # for GPT-4o eval

# 2. Run baseline comparison
cd baseline_comparison && python run_all_baselines.py --count 100

# 3. Run ablation study
cd ablation_study && python run_all_ablations.py --count 200

# 4. Evaluate & summarize
cd baseline_comparison/04_evaluate_all && python evaluate.py
cd ablation_study/06_evaluate_all && python evaluate.py
```
