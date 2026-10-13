"""M5-A 机制自省接口：静态元数据 + 运行时 prompt/模型/上下文预览。

只读、无网络、无 LLM 调用 —— 模块加载级 import 即可完成，离线演示安全。
"""
from fastapi import APIRouter

from ..adapters import pipeline_bridge

router = APIRouter(prefix="/api/pipeline", tags=["mechanism"])


@router.get("/introspect")
def introspect():
    """返回 4 领域的检索算法 / 上下文拼接 / 评分规则 / prompt 全文。"""
    return pipeline_bridge.introspect_pipeline()
