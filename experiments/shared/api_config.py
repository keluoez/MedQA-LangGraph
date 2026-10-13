# -*- coding: utf-8 -*-
"""
Shared API configuration for all experiments.
Set environment variables before running any experiment script.
"""

import os

# 加载项目根目录 .env（密钥配置）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Generation backbone (NVIDIA / LLM provider) ───────────────────────────────
NVIDIA_API_KEY  = os.getenv("BIOREASON_API_KEY") or os.getenv("NVIDIA_API_KEY") or ""
NVIDIA_BASE_URL = os.getenv("BIOREASON_BASE_URL", "https://integrate.api.nvidia.com/v1")
GENERATION_MODEL = os.getenv("BIOREASON_MODEL", "meta/llama-4-maverick-17b-128e-instruct")

# ── Evaluation backbone (SiliconFlow — DeepSeek / Qwen) ───────────────────────
SILICONFLOW_API_KEY  = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V3"
QWEN_MODEL     = "Qwen/Qwen2.5-72B-Instruct"

# ── Evaluation backbone (OpenAI — GPT-4o) ─────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = "https://proaiapi.tech/v1"
GPT4O_MODEL     = "gpt-4o-2024-11-20"
