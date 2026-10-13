"""测试全局夹具：离线环境约定。

M5-B：统一关闭 instructor 结构化抢救路径（BIOREASON_STRUCTURED_OUTPUT=0），
保证所有 verifier 单测走确定性的第一层（parse + Pydantic 校验 + 降级）路径，
不会在离线环境意外发起真实 LLM 调用。需要测 instructor 路径时在用例内 monkeypatch.setenv 覆盖。
"""
import os

os.environ["BIOREASON_STRUCTURED_OUTPUT"] = "0"
