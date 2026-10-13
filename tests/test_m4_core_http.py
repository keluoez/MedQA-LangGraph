"""M4-B2（B2/B3）离线单元测试：core.http 的 SQLite 检索缓存 + NCBI 令牌桶限流。

- B2：cached_get 同一 (url, params) 只发一次真实请求，命中缓存返回 CachedResponse；
  非 200（如 404）属确定性结果也缓存；网络异常不缓存；BIOREASON_NO_CACHE=1 禁用。
- B3：ncbi=True 且缓存未命中时先过 3 req/s 令牌桶；命中缓存则快速返回不限流。
- 全程离线：requests.get 以桩替身，令牌桶用受控假时钟驱动（不真实等待）；
  缓存文件重定向到 tmp_path，不污染项目根 .cache/。
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import http as core_http
from core.http import CachedResponse, SqliteCache, TokenBucket, _cache_key, cached_get


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------

class FakeResponse:
    """模拟 requests.Response 的最小接口（status_code/text/json）。"""

    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.text = json.dumps(payload if payload is not None else {"ok": True})

    def json(self):
        return json.loads(self.text)


class FakeClock:
    """受控时钟：monotonic 只随 sleep 推进，令牌桶测试确定且零真实等待。"""

    def __init__(self, start=1000.0):
        self.now = float(start)
        self.sleep_calls = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleep_calls.append(seconds)
        self.now += seconds


class GetStub:
    """requests.get 替身：记录调用并按队列返回响应/抛异常；队列耗尽后重复最后一项。"""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture()
def isolated_cache(tmp_path, monkeypatch):
    """每个用例独立的缓存环境：重定向缓存路径 + 重置模块级单例 + 清除 NO_CACHE。"""
    monkeypatch.setattr(core_http, "_CACHE_PATH", tmp_path / "http_cache.sqlite3")
    monkeypatch.setattr(core_http, "_cache", None)
    monkeypatch.delenv("BIOREASON_NO_CACHE", raising=False)
    return tmp_path


@pytest.fixture()
def install_get(monkeypatch):
    def _install(stub):
        monkeypatch.setattr(core_http.requests, "get", stub)
        return stub
    return _install


@pytest.fixture()
def spy_bucket(monkeypatch):
    """将 NCBI_BUCKET 替换为记录 acquire 调用的探针。"""
    calls = []
    monkeypatch.setattr(
        core_http, "NCBI_BUCKET", SimpleNamespace(acquire=lambda amount=1.0: calls.append(amount))
    )
    return calls


# ---------------------------------------------------------------------------
# B2 存储层：SqliteCache / _cache_key
# ---------------------------------------------------------------------------

def test_sqlite_cache_roundtrip_and_overwrite(tmp_path):
    cache = SqliteCache(tmp_path / "c1.sqlite3")
    assert cache.get("missing") is None
    cache.set("k", 200, "hello")
    assert cache.get("k") == (200, "hello")
    cache.set("k", 404, "gone")  # 同 key 覆盖
    assert cache.get("k") == (404, "gone")


def test_sqlite_cache_persists_across_restart(tmp_path):
    """B2 核心价值：重启进程后缓存仍命中，避免重复网络请求。"""
    path = tmp_path / "c2.sqlite3"
    SqliteCache(path).set("k", 200, "persisted")
    assert SqliteCache(path).get("k") == (200, "persisted")


def test_cache_key_ignores_param_order():
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    a = _cache_key(url, {"db": "pubmed", "term": "x", "retmax": 6})
    b = _cache_key(url, {"retmax": 6, "term": "x", "db": "pubmed"})
    assert a == b
    assert _cache_key("https://x/1", None) != _cache_key("https://x/2", None)


# ---------------------------------------------------------------------------
# B3：TokenBucket 限流
# ---------------------------------------------------------------------------

def test_token_bucket_burst_then_wait(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(core_http.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(core_http.time, "sleep", clock.sleep)
    # rate=4.0 使单次等待恰为 0.25（二进制精确），避免 1/3 类增量在受控时钟下
    # 因浮点亚 ULP 亏损被时钟吸收而无法补足令牌（真实单调钟不受此影响）
    bucket = TokenBucket(rate=4.0, capacity=4.0)
    for _ in range(4):  # 桶容量 4：前 4 个请求瞬时通过
        bucket.acquire(1.0)
    assert clock.sleep_calls == []
    start = clock.now
    bucket.acquire(1.0)  # 第 5 个请求需等待 0.25 秒补足令牌
    assert clock.now - start == 0.25
    assert clock.sleep_calls == [0.25]


def test_token_bucket_sustained_rate(monkeypatch):
    """8 个请求 @4 req/s：4 个突发 + 4 个排队，总等待 = (8-4)/4 = 1.0 秒。"""
    clock = FakeClock()
    monkeypatch.setattr(core_http.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(core_http.time, "sleep", clock.sleep)
    bucket = TokenBucket(rate=4.0, capacity=4.0)
    start = clock.now
    for _ in range(8):
        bucket.acquire(1.0)
    assert clock.now - start == 1.0


def test_ncbi_bucket_matches_official_policy():
    """NCBI E-utilities 官方政策：3 req/s，桶容量 3（允许瞬时突发 3）。"""
    assert core_http.NCBI_BUCKET.rate == 3.0
    assert core_http.NCBI_BUCKET.capacity == 3.0


# ---------------------------------------------------------------------------
# B2：cached_get 缓存行为
# ---------------------------------------------------------------------------

def test_cached_get_fetches_once_then_hits_cache(isolated_cache, install_get):
    stub = install_get(GetStub(FakeResponse(200, {"ok": 1})))
    r1 = cached_get("https://rest.uniprot.org/uniprotkb/P12345.json", timeout=20)
    r2 = cached_get("https://rest.uniprot.org/uniprotkb/P12345.json", timeout=20)
    assert len(stub.calls) == 1                    # 只发一次真实请求
    assert r2.status_code == 200 and r2.json() == {"ok": 1}
    assert isinstance(r2, CachedResponse)          # 第二次由缓存供给
    assert stub.calls[0]["timeout"] == 20


def test_cached_get_param_order_same_key(isolated_cache, install_get):
    stub = install_get(GetStub(FakeResponse()))
    cached_get("https://x/api", params={"a": 1, "b": 2})
    cached_get("https://x/api", params={"b": 2, "a": 1})  # 参数序不同、字典等价 → 同 key
    assert len(stub.calls) == 1


def test_cached_get_caches_non_200(isolated_cache, install_get):
    """非 200（如 404）是确定性结果，同样缓存以避免重复请求。"""
    stub = install_get(GetStub(FakeResponse(404)))
    r1 = cached_get("https://x/missing")
    r2 = cached_get("https://x/missing")
    assert len(stub.calls) == 1
    assert r2.status_code == 404


def test_cached_get_env_disables_cache(isolated_cache, monkeypatch, install_get):
    monkeypatch.setenv("BIOREASON_NO_CACHE", "1")
    stub = install_get(GetStub(FakeResponse()))
    r1 = cached_get("https://x/api")
    r2 = cached_get("https://x/api")
    assert len(stub.calls) == 2                    # 每次都走网络
    assert not isinstance(r2, CachedResponse)     # 返回真实响应对象


def test_cached_get_error_not_cached(isolated_cache, install_get):
    """网络异常不缓存，下次调用可重试成功。"""
    stub = install_get(GetStub(requests.ConnectionError("boom"), FakeResponse(200, {"ok": 1})))
    with pytest.raises(requests.ConnectionError):
        cached_get("https://x/flaky")
    r = cached_get("https://x/flaky")
    assert r.status_code == 200 and r.json() == {"ok": 1}


# ---------------------------------------------------------------------------
# B3：cached_get 与令牌桶协同
# ---------------------------------------------------------------------------

def test_cached_get_ncbi_miss_goes_through_bucket(isolated_cache, install_get, spy_bucket):
    """缓存未命中且 ncbi=True：真实请求前必须经过令牌桶。"""
    stub = install_get(GetStub(FakeResponse()))
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": "aspirin"}
    cached_get(url, params=params, ncbi=True)
    assert spy_bucket == [1.0]


def test_cached_get_ncbi_hit_skips_bucket(isolated_cache, install_get, spy_bucket):
    """缓存命中：快速返回，不触发限流也不发网络请求。"""
    stub = install_get(GetStub(FakeResponse()))
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {"db": "pubmed", "id": "12345678"}
    cached_get(url, params=params, ncbi=True)
    cached_get(url, params=params, ncbi=True)
    assert spy_bucket == [1.0]                    # 仅未命中的第一次走了桶
    assert len(stub.calls) == 1


def test_cached_get_non_ncbi_skips_bucket(isolated_cache, install_get, spy_bucket):
    """UniProt/ChEMBL 等非 NCBI 源不做 3 req/s 限流（各自服务无此政策）。"""
    install_get(GetStub(FakeResponse()))
    cached_get("https://rest.uniprot.org/uniprotkb/P05067.json", timeout=20)
    assert spy_bucket == []
