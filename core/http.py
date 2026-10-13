"""M4-B2（B2/B3）：HTTP 检索基础设施 —— SQLite 响应缓存 + NCBI 令牌桶限流。

- cached_get：透明替换 requests.get；缓存命中不发网络请求，未命中才真实请求。
- ncbi=True 的请求先过 3 req/s 令牌桶（NCBI E-utilities 官方速率政策），线程安全。
- 缓存文件位于项目根 .cache/http_cache.sqlite3；设置环境变量 BIOREASON_NO_CACHE=1 可禁用。
- 仅缓存 HTTP 响应（含 404 等非 200 状态，属确定性结果）；网络异常不缓存、原样抛出。
"""
import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CACHE_PATH = _PROJECT_ROOT / ".cache" / "http_cache.sqlite3"


class SqliteCache:
    """线程安全的 SQLite 键值缓存（key -> (status_code, body_text)）。"""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS http_cache ("
            "key TEXT PRIMARY KEY, status INTEGER, body TEXT, created_at REAL)"
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[Tuple[int, str]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT status, body FROM http_cache WHERE key = ?", (key,)
            ).fetchone()
        return (int(row[0]), str(row[1])) if row is not None else None

    def set(self, key: str, status: int, body: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO http_cache (key, status, body, created_at) VALUES (?, ?, ?, ?)",
                (key, int(status), body, time.time()),
            )
            self._conn.commit()


class TokenBucket:
    """阻塞式令牌桶限流器。rate=每秒补充令牌数，capacity=桶容量（允许的瞬时突发）。"""

    def __init__(self, rate: float, capacity: float):
        self.rate = float(rate)
        self.capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, amount: float = 1.0) -> None:
        with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                time.sleep((amount - self._tokens) / self.rate)


# NCBI E-utilities 政策：3 req/s（https://ncbiinsights.ncbi.nlm.nih.gov/new-entrez-web-service-e-utilities/）
NCBI_BUCKET = TokenBucket(rate=3.0, capacity=3.0)


class CachedResponse:
    """缓存命中时返回的轻量响应对象，兼容 requests.Response 的 status_code/text/json() 用法。"""

    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    def json(self) -> Any:
        return json.loads(self.text)


_cache: Optional[SqliteCache] = None
_cache_lock = threading.Lock()


def _get_cache() -> Optional[SqliteCache]:
    global _cache
    if os.getenv("BIOREASON_NO_CACHE") == "1":
        return None
    with _cache_lock:
        if _cache is None:
            _cache = SqliteCache(_CACHE_PATH)
        return _cache


def _cache_key(url: str, params: Optional[dict]) -> str:
    raw = json.dumps({"url": url, "params": params or {}}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cached_get(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = 15,
    ncbi: bool = False,
):
    """带 SQLite 缓存的 HTTP GET。

    - 同一 (url, params) 的响应只请求一次，后续运行/重启直接读缓存（B2）；
    - ncbi=True 时，缓存未命中的真实请求先经过 3 req/s 令牌桶（B3）；
    - 返回对象支持 .status_code / .text / .json()，可直接替换原 requests.get 调用。
    """
    key = _cache_key(url, params)
    cache = _get_cache()
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return CachedResponse(hit[0], hit[1])
    if ncbi:
        NCBI_BUCKET.acquire()
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    if cache is not None:
        cache.set(key, resp.status_code, resp.text)
    return resp
