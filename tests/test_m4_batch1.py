"""M4 第一批重构（P3/P5/P6）的离线单元测试，无需 API key、无网络调用。

- P3: run_batch 续传只统计成功记录；失败样本写入 .failures.jsonl，重启后自动重试
- P5: parse_json_object 兼容围栏 / 前后缀文本 / 多 JSON 块 / 嵌套结构
- P6: llm_call 对瞬时错误指数退避重试；非重试错误立即抛出；重试耗尽后抛最后错误
"""
import json
import sys
from pathlib import Path

import httpx
import openai as openai_sdk
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basic_drug_interaction import langgraph_pipeline as drug
from basic_experimental_conditions import langgraph_pipeline as exp
from basic_pathological_environment import langgraph_pipeline as patho
from basic_physiological_conditions import langgraph_pipeline as phys

ALL_MODULES = [drug, exp, patho, phys]


# ---------- P5: parse_json_object ----------

@pytest.mark.parametrize("mod", ALL_MODULES, ids=["drug", "exp", "patho", "phys"])
def test_parse_plain_object(mod):
    assert mod.parse_json_object('{"factuality_score": 4.0}') == {"factuality_score": 4.0}


@pytest.mark.parametrize("mod", ALL_MODULES, ids=["drug", "exp", "patho", "phys"])
def test_parse_json_fence(mod):
    text = '```json\n{"factuality_score": 4.0, "grounding_score": 3}\n```'
    assert mod.parse_json_object(text)["grounding_score"] == 3


@pytest.mark.parametrize("mod", ALL_MODULES, ids=["drug", "exp", "patho", "phys"])
def test_parse_surrounding_text(mod):
    text = 'Here is my evaluation:\n{"factuality_score": 3.5}\nHope this helps.'
    assert mod.parse_json_object(text)["factuality_score"] == 3.5


@pytest.mark.parametrize("mod", ALL_MODULES, ids=["drug", "exp", "patho", "phys"])
def test_parse_first_valid_block_among_multiple(mod):
    # 首块含尾逗号不合法，旧版贪婪正则会把两块首尾拼接导致整体解析失败
    text = '{"bad": 1,} some text {"good": 2}'
    assert mod.parse_json_object(text) == {"good": 2}


@pytest.mark.parametrize("mod", ALL_MODULES, ids=["drug", "exp", "patho", "phys"])
def test_parse_nested_object(mod):
    # pathological 旧版内联正则 \{[^}]+\} 无法匹配嵌套结构
    text = '{"scores": {"factuality": 4}, "errors": []}'
    assert mod.parse_json_object(text)["scores"]["factuality"] == 4


@pytest.mark.parametrize("mod", ALL_MODULES, ids=["drug", "exp", "patho", "phys"])
def test_parse_brace_inside_string_value(mod):
    text = '{"note": "see {PMID:12345} for details", "score": 2}'
    assert mod.parse_json_object(text)["score"] == 2


@pytest.mark.parametrize("mod", ALL_MODULES, ids=["drug", "exp", "patho", "phys"])
def test_parse_garbage_returns_empty(mod):
    assert mod.parse_json_object("no json here") == {}
    assert mod.parse_json_object("") == {}


# ---------- P6: llm_call retry ----------

class _FakeCompletions:
    def __init__(self, script):
        self.script = list(script)  # 元素为 Exception 或伪响应

    def create(self, **kwargs):
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, script):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions(script)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [type("C", (), {})()]
        self.choices[0].message = type("M", (), {})()
        self.choices[0].message.content = content


def _mk_request():
    return httpx.Request("POST", "http://test.local")


def _mk_rate_limit():
    return openai_sdk.RateLimitError(
        "rate limited", response=httpx.Response(429, request=_mk_request()), body=None
    )


def _mk_timeout():
    return openai_sdk.APITimeoutError(request=_mk_request())


@pytest.mark.parametrize("mod", ALL_MODULES, ids=["drug", "exp", "patho", "phys"])
def test_llm_call_retries_transient_and_succeeds(mod, monkeypatch):
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    client = _FakeClient([_mk_rate_limit(), _mk_timeout(), _FakeResponse("ok")])
    assert mod.llm_call(client, "sys", "user") == "ok"


@pytest.mark.parametrize("mod", ALL_MODULES, ids=["drug", "exp", "patho", "phys"])
def test_llm_call_no_retry_on_auth_error(mod, monkeypatch):
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    err = openai_sdk.AuthenticationError(
        "bad key", response=httpx.Response(401, request=_mk_request()), body=None
    )
    client = _FakeClient([err])
    with pytest.raises(openai_sdk.AuthenticationError):
        mod.llm_call(client, "sys", "user")
    assert len(client.chat.completions.script) == 0  # 恰好调用 1 次后 pop 空，未重试


@pytest.mark.parametrize("mod", ALL_MODULES, ids=["drug", "exp", "patho", "phys"])
def test_llm_call_raises_after_exhausting_retries(mod, monkeypatch):
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    client = _FakeClient([_mk_timeout(), _mk_timeout(), _mk_timeout()])
    with pytest.raises(openai_sdk.APITimeoutError):
        mod.llm_call(client, "sys", "user")
    assert len(client.chat.completions.script) == 0  # 恰好 3 次调用后放弃


# ---------- P3: run_batch resume + failures ----------

def test_run_batch_resume_skips_invalid_and_records_failures(tmp_path, monkeypatch):
    out = tmp_path / "out.jsonl"
    # 预置历史输出：1 条成功 + 1 条 error 行 + 1 条空 answer 行 + 1 条坏 JSON 行
    out.write_text(
        json.dumps({"drug": "D1", "protein": "P1", "answer": "a", "quality_score": 8.5}) + "\n",
        encoding="utf-8",
    )
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"drug": "D2", "protein": "P2", "error": "boom"}) + "\n")
        f.write(json.dumps({"drug": "D3", "protein": "P3", "answer": "", "quality_score": 0}) + "\n")
        f.write("not json\n")

    pairs = [
        {"drug": "D1", "protein": "P1"},
        {"drug": "D2", "protein": "P2"},
        {"drug": "D3", "protein": "P3"},
    ]

    monkeypatch.setattr(drug, "build_graph", lambda: None)
    calls = []

    def fake_run_one(app, pair, max_corrections=2):
        calls.append(pair["drug"])
        if pair["drug"] == "D2":
            raise RuntimeError("boom")  # D2 本轮依然失败
        return {"drug": pair["drug"], "protein": pair["protein"], "answer": "ok", "quality_score": 9.0}

    monkeypatch.setattr(drug, "run_one", fake_run_one)

    drug.run_batch(pairs, out)

    # D1 成功被跳过；error 行的 D2、空 answer 的 D3、坏 JSON 不计入已完成 → 全部重试
    assert sorted(calls) == ["D2", "D3"]

    main_lines = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.strip() != "not json"
    ]
    drugs_in_main = [r["drug"] for r in main_lines if "drug" in r]
    assert "D1" in drugs_in_main and "D3" in drugs_in_main  # D3 本轮成功落库
    # 历史遗留的 error 行仍保留在主文件（不做清理），但本轮成功的 D1/D3 不含 error
    assert all("error" not in r for r in main_lines if r.get("drug") in ("D1", "D3"))

    # D2 失败写入独立 failures 文件
    failure_lines = [
        json.loads(line)
        for line in (tmp_path / "out.failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(failure_lines) == 1
    assert failure_lines[0]["drug"] == "D2" and "boom" in failure_lines[0]["error"]

    # 模拟二次重启：D3 已成功应被跳过，D2 仍失败 → 自动重试
    calls.clear()
    drug.run_batch(pairs, out)
    assert sorted(calls) == ["D2"]


# ---------- Demo 回归：防腐层 import 链不受影响 ----------

def test_pipeline_bridge_importable():
    from api.adapters import pipeline_bridge

    assert callable(pipeline_bridge.build_graph)
