"""The only OpenRouter caller. Every completion is cached in `ada_cache.llm_cache`.

    cached_chat(role, messages, tools=None, **params) -> ChatResult

`role` is a key of harness/models.json (CONTRACTS.md §7). The cache key is
sha256 of canonical JSON {model_id, messages, tools, params}. LLM_CACHE_MODE=record
(default) calls the API on a miss; replay_only raises CacheMiss instead. Transient
failures retry after 1/2/4/8 s (Retry-After wins when sent); if the primary model
still fails, the role's `alt` model is tried once through the same loop.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from pymongo import ASCENDING
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from core.contracts import ChatResult, LlmCacheEntry
from core.db import ADA_CACHE, db

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_PATH = Path(__file__).resolve().parent.parent / "harness" / "models.json"
CACHE_COLLECTION = "llm_cache"
MODES = ("record", "replay_only")
TIMEOUT_S = 90.0  # per request; the controller is a slow reasoning model
DEFAULT_MAX_TOKENS = 4096
BACKOFF_S = (1, 2, 4, 8)  # sleeps between the 5 attempts
MAX_RETRY_AFTER_S = 60.0
MAX_CONCURRENCY = 4

# Test hooks: tests monkeypatch these; None means build the real one on first use.
_http: httpx.Client | None = None
_cache: Collection | None = None
_sleep = time.sleep

_semaphore = threading.BoundedSemaphore(MAX_CONCURRENCY)
_indexed: set[str] = set()
_index_lock = threading.Lock()


class LLMError(RuntimeError):
    """The primary and alt models both failed (or auth failed)."""


class CacheMiss(LLMError):
    """LLM_CACHE_MODE=replay_only and no cached response exists."""


class _Retryable(Exception):
    def __init__(self, reason: str, retry_after: float | None = None):
        super().__init__(reason)
        self.retry_after = retry_after


class _Fatal(Exception):
    pass


# --- config -----------------------------------------------------------------

@lru_cache(maxsize=1)
def models() -> dict[str, Any]:
    return json.loads(MODELS_PATH.read_text())


def _mode() -> str:
    load_dotenv()
    mode = os.environ.get("LLM_CACHE_MODE") or "record"
    if mode not in MODES:
        raise ValueError(f"LLM_CACHE_MODE={mode!r}; expected one of {MODES}")
    return mode


def _client() -> httpx.Client:
    global _http
    if _http is None:
        _http = httpx.Client(timeout=TIMEOUT_S)
    return _http


def _collection() -> Collection:
    coll = _cache if _cache is not None else db(ADA_CACHE)[CACHE_COLLECTION]
    key = coll.full_name
    if key not in _indexed:
        with _index_lock:
            if key not in _indexed:
                coll.create_index(
                    [("model_id", ASCENDING), ("prompt_hash", ASCENDING)], unique=True
                )
                _indexed.add(key)
    return coll


# --- keys and cost ----------------------------------------------------------

def canonical_request(
    model_id: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    return {"model_id": model_id, "messages": messages, "tools": tools, "params": params}


def prompt_hash(request: dict[str, Any]) -> str:
    blob = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def cost_usd(spec: dict[str, Any], usage: dict[str, Any]) -> float:
    tokens_in = usage.get("prompt_tokens") or 0
    tokens_out = usage.get("completion_tokens") or 0
    usd = (tokens_in * spec["usd_per_mtok_in"] + tokens_out * spec["usd_per_mtok_out"]) / 1e6
    return round(usd, 6)


# --- HTTP -------------------------------------------------------------------

def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        secs = float(raw)
    except ValueError:
        try:
            secs = (parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError):
            return None
    return min(max(secs, 0.0), MAX_RETRY_AFTER_S)


def _post_once(body: dict[str, Any]) -> dict[str, Any]:
    load_dotenv()
    headers = {"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}"}
    try:
        with _semaphore:
            resp = _client().post(OPENROUTER_URL, json=body, headers=headers, timeout=TIMEOUT_S)
    except httpx.TransportError as e:  # includes timeouts
        raise _Retryable(f"{type(e).__name__}: {e}") from e
    if resp.status_code == 401:
        raise LLMError(f"OpenRouter 401: {resp.text[:200]}")
    if resp.status_code == 429 or resp.status_code >= 500:
        raise _Retryable(f"HTTP {resp.status_code}", _retry_after(resp))
    if resp.status_code >= 400:
        raise _Fatal(f"HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    if data.get("error") or not data.get("choices"):
        raise _Retryable(f"bad body: {str(data.get('error') or data)[:200]}")
    return data


def _post_with_retries(body: dict[str, Any]) -> dict[str, Any]:
    for attempt in range(len(BACKOFF_S) + 1):
        try:
            return _post_once(body)
        except _Retryable as e:
            if attempt == len(BACKOFF_S):
                raise _Fatal(f"gave up after {attempt + 1} attempts: {e}") from e
            _sleep(e.retry_after if e.retry_after is not None else BACKOFF_S[attempt])
    raise AssertionError("unreachable")


# --- public -----------------------------------------------------------------

def _result(role: str, spec: dict[str, Any], key: str, response: dict[str, Any],
            cached: bool) -> ChatResult:
    usage = response.get("usage") or {}
    return ChatResult(
        role=role, model_id=spec["id"], prompt_hash=key,
        message=response["choices"][0]["message"], usage=usage,
        cost_usd=cost_usd(spec, usage), cached=cached,
    )


def cached_chat(
    role: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
    **params: Any,
) -> ChatResult:
    table = models()
    if role not in table:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(table)}")
    primary = table[role]
    params = dict(params)
    if "temperature" not in params and primary.get("temperature") is not None:
        params["temperature"] = primary["temperature"]
    params.setdefault("max_tokens", DEFAULT_MAX_TOKENS)

    specs = [primary] + ([primary["alt"]] if primary.get("alt") else [])
    candidates = []
    for spec in specs:
        request = canonical_request(spec["id"], messages, tools, params)
        candidates.append((spec, request, prompt_hash(request)))

    coll = _collection()
    for spec, _, key in candidates:
        hit = coll.find_one({"model_id": spec["id"], "prompt_hash": key})
        if hit is not None:
            return _result(role, spec, key, hit["response"], cached=True)
    if _mode() == "replay_only":
        raise CacheMiss(f"{role}: no cached response for {[k for _, _, k in candidates]}")

    errors = []
    for spec, request, key in candidates:
        body = {"model": spec["id"], "messages": messages, **params}
        if tools:
            body["tools"] = tools
        try:
            response = _post_with_retries(body)
        except _Fatal as e:
            errors.append(f"{spec['id']}: {e}")
            continue
        entry = LlmCacheEntry(
            model_id=spec["id"], prompt_hash=key, request=request, response=response,
            usage=response.get("usage") or {}, ts=datetime.now(timezone.utc),
        )
        try:
            coll.insert_one(entry.model_dump(exclude={"id"}))
        except DuplicateKeyError:
            pass  # an identical concurrent call stored it first
        return _result(role, spec, key, response, cached=False)
    raise LLMError(f"{role}: all models failed: {errors}")
