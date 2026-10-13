"""M4-B2（P4 抽取）：LLM 客户端共享层 —— 密钥解析、客户端构造、带重试的调用。

从 4 个领域模块收敛的公共实现；模块侧保留同名薄包装以绑定各自的 MODEL/BASE_URL 配置。
"""
import os
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

DEFAULT_MODEL = "meta/llama-4-maverick-17b-128e-instruct"


def resolve_api_key(primary: str = "", legacy_file: Optional[Path] = None) -> str:
    """解析 API 密钥：优先环境变量；未配置时回退读取同目录 legacy pipeline.py 中的 API_KEY。"""
    if primary:
        return primary
    if legacy_file is not None and Path(legacy_file).exists():
        text = Path(legacy_file).read_text(encoding="utf-8", errors="ignore")
        match = re.search(r'API_KEY\s*=\s*["\']([^"\']+)["\']', text)
        if match:
            return match.group(1)
    return ""


def get_client(api_key: str, base_url: str) -> OpenAI:
    if not api_key:
        raise RuntimeError(
            "缺少API密钥，请设置 BIOREASON_API_KEY / NVIDIA_API_KEY，或在同目录 legacy pipeline.py 配置 API_KEY"
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def llm_call(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1800,
    model: str = DEFAULT_MODEL,
    timeout: float = 45,
) -> str:
    """LLM 调用，带 2 次指数退避重试；仅对瞬时性错误重试（超时/连接/限流/5xx），其余立即抛出。"""
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    last_err = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return response.choices[0].message.content or ""
        except (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError) as e:
            last_err = e
            import time

            if attempt < 2:
                time.sleep(2 ** attempt)  # 1s → 2s
    raise last_err


def llm_call_json(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    response_model,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 900,
    timeout: float = 45,
):
    """M5-B 第二层：instructor 生成端结构化约束（response_model 驱动 JSON Schema + 校验失败自动重问）。

    尽力而为：BIOREASON_STRUCTURED_OUTPUT=0、instructor 未安装、端点不支持 tools 或调用异常时
    返回 None，由调用方回退第一层（parse + Pydantic 校验 + 降级）路径，绝不阻断业务。
    """
    if os.getenv("BIOREASON_STRUCTURED_OUTPUT", "1") == "0":
        return None
    try:
        import instructor
    except ImportError:
        return None
    try:
        wrapped = instructor.from_openai(client)
        obj = wrapped.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=1,
        )
        return response_model.model_validate(obj).model_dump()
    except Exception:
        return None
