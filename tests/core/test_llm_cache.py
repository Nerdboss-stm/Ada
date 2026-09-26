import json

import httpx
import pytest

from core import llm
from core.db import ADA_TEST, db

COLL = "llm_cache_test"
MSGS = [{"role": "user", "content": "walk"}]


def ok_body(model: str, tokens_in: int = 1000, tokens_out: int = 200) -> dict:
    return {
        "id": "gen-1", "model": model,
        "choices": [{"message": {"role": "assistant", "content": "gait"}}],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out,
                  "total_tokens": tokens_in + tokens_out},
    }


class FakeOpenRouter:
    """Serves queued responses; default is a 200 for whatever model was asked."""

    def __init__(self):
        self.requests: list[dict] = []
        self.queue: list[httpx.Response | Exception] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.queue:
            item = self.queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return httpx.Response(200, json=ok_body(body["model"]))


@pytest.fixture
def api(monkeypatch):
    coll = db(ADA_TEST)[COLL]
    coll.drop()
    fake = FakeOpenRouter()
    sleeps: list[float] = []
    monkeypatch.setattr(llm, "_http", httpx.Client(transport=httpx.MockTransport(fake)))
    monkeypatch.setattr(llm, "_cache", coll)
    monkeypatch.setattr(llm, "_sleep", sleeps.append)
    monkeypatch.setattr(llm, "_indexed", set())
    monkeypatch.setenv("LLM_CACHE_MODE", "record")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    fake.sleeps = sleeps
    fake.coll = coll
    yield fake
    coll.drop()


def test_second_identical_call_makes_zero_http(api):
    first = llm.cached_chat("agent_v0", MSGS)
    assert len(api.requests) == 1 and not first.cached
    reordered = [{"content": "walk", "role": "user"}]
    second = llm.cached_chat("agent_v0", reordered)
    assert len(api.requests) == 1
    assert second.cached and second.prompt_hash == first.prompt_hash
    assert second.message == first.message and second.cost_usd == first.cost_usd


def test_request_defaults_and_key_sensitivity(api):
    llm.cached_chat("agent_v0", MSGS)
    body = api.requests[0]
    assert body["model"] == llm.models()["agent_v0"]["id"]
    assert body["temperature"] == 0 and body["max_tokens"] == 4096
    assert "tools" not in body
    llm.cached_chat("agent_v0", MSGS, max_tokens=512)
    assert len(api.requests) == 2 and api.requests[1]["max_tokens"] == 512
    llm.cached_chat("controller", MSGS)  # temperature null => not sent
    assert "temperature" not in api.requests[2]


def test_replay_only_miss_raises_without_http(api, monkeypatch):
    monkeypatch.setenv("LLM_CACHE_MODE", "replay_only")
    with pytest.raises(llm.CacheMiss):
        llm.cached_chat("judge", MSGS)
    assert api.requests == []


def test_replay_only_hit_returns_cached(api, monkeypatch):
    recorded = llm.cached_chat("judge", MSGS)
    monkeypatch.setenv("LLM_CACHE_MODE", "replay_only")
    replayed = llm.cached_chat("judge", MSGS)
    assert replayed.cached and replayed.message == recorded.message
    assert len(api.requests) == 1


@pytest.mark.parametrize("role", ["agent_v0", "judge", "frontier"])
def test_cost_matches_models_json(api, role):
    spec = llm.models()[role]
    api.queue.append(httpx.Response(200, json=ok_body(spec["id"], 123_456, 7_890)))
    res = llm.cached_chat(role, MSGS)
    expected = round((123_456 * spec["usd_per_mtok_in"] + 7_890 * spec["usd_per_mtok_out"]) / 1e6, 6)
    assert res.cost_usd == expected
    assert res.usage["prompt_tokens"] == 123_456


def test_retry_after_is_honored(api):
    api.queue += [httpx.Response(429, headers={"Retry-After": "3"}), httpx.Response(503)]
    res = llm.cached_chat("agent_v0", MSGS)
    assert not res.cached and len(api.requests) == 3
    assert api.sleeps == [3.0, 2]


def test_timeout_is_retried(api):
    api.queue.append(httpx.ReadTimeout("slow"))
    res = llm.cached_chat("controller", MSGS)
    assert res.model_id == llm.models()["controller"]["id"]
    assert api.sleeps == [1]


def test_alt_fallback_after_repeated_failure(api):
    api.queue += [httpx.Response(500)] * 5
    res = llm.cached_chat("agent_v0", MSGS)
    alt = llm.models()["agent_v0"]["alt"]
    assert res.model_id == alt["id"]
    assert [r["model"] for r in api.requests[-1:]] == [alt["id"]]
    assert api.sleeps == [1, 2, 4, 8]
    assert res.cost_usd == round((1000 * alt["usd_per_mtok_in"] + 200 * alt["usd_per_mtok_out"]) / 1e6, 6)
    again = llm.cached_chat("agent_v0", MSGS)  # alt answer is reused, no new HTTP
    assert again.cached and len(api.requests) == 6


def test_both_models_fail_raises(api):
    api.queue += [httpx.Response(400, text="bad")] * 2
    with pytest.raises(llm.LLMError):
        llm.cached_chat("judge", MSGS)
    assert api.coll.count_documents({}) == 0


def test_unique_index_and_document_shape(api):
    llm.cached_chat("agent_v0", MSGS, tools=[{"type": "function", "function": {"name": "read_task"}}])
    idx = api.coll.index_information()
    assert any(v.get("unique") and v["key"] == [("model_id", 1), ("prompt_hash", 1)] for v in idx.values())
    doc = api.coll.find_one()
    assert set(doc) == {"_id", "model_id", "prompt_hash", "request", "response", "usage", "ts"}
    assert doc["request"]["tools"][0]["function"]["name"] == "read_task"
    assert api.requests[0]["tools"] == doc["request"]["tools"]


def test_unknown_role_rejected(api):
    with pytest.raises(ValueError):
        llm.cached_chat("nobody", MSGS)
