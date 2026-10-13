r"""M4-B2（P4 抽取）：LLM 输出 JSON 解析。

提取 LLM 输出中首个可解析的平衡 JSON 对象（原 4 模块各自内联实现的收敛版）。

相比旧版贪婪/内联正则，可正确处理：
- ```json 代码围栏包裹；
- 对象前后夹杂自然语言说明文本；
- verifier 输出含嵌套结构（旧版 `\{[^}]+\}` 会匹配失败导致全 0 分降级）；
- 输出含多个 JSON 块时取首个合法块（旧版首尾拼接会跨块截取导致解析失败）。
彻底解析失败返回空 dict，由调用方按既定降级策略处理。
"""
import json
import re
from typing import Any, Dict


def parse_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1)
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    start = None  # 该平衡块不合法，继续扫描下一个
    return {}
