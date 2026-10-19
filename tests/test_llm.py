import pytest

from retailmind.generate.llm import (
    DiskCache,
    LLMError,
    OpenAIChatClient,
    ScriptedClient,
    _cache_key,
    make_client,
)


def test_scripted_client_returns_replies_then_default_and_records_calls():
    client = ScriptedClient(["first", "second"], default="fallback")
    msgs = [{"role": "user", "content": "hi"}]
    assert client.complete(msgs, temperature=0.0, max_tokens=10) == "first"
    assert client.complete(msgs, temperature=0.0, max_tokens=10) == "second"
    assert client.complete(msgs, temperature=0.0, max_tokens=10) == "fallback"
    assert len(client.calls) == 3 and client.calls[0] == msgs


def test_cache_key_depends_on_every_part():
    base = dict(
        model="m", messages=[{"role": "user", "content": "a"}], temperature=0.0, max_tokens=8
    )
    k = _cache_key(**base)
    assert k == _cache_key(**base)
    assert k != _cache_key(**{**base, "temperature": 0.5})
    assert k != _cache_key(**{**base, "model": "other"})


def test_disk_cache_round_trip(tmp_path):
    cache = DiskCache(tmp_path / "c")
    assert cache.get("k") is None
    cache.put("k", {"model": "m"}, "hello")
    assert cache.get("k") == "hello"


class _FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def _patch_requests(monkeypatch, responses):
    calls = {"n": 0, "posts": []}
    import requests

    def fake_post(url, json, headers, timeout):
        calls["posts"].append({"url": url, "json": json, "headers": headers})
        result = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(requests, "post", fake_post)
    return calls


def _ok(text):
    return _FakeResponse(200, {"choices": [{"message": {"content": text}}]})


def test_client_posts_openai_format_and_parses_reply(monkeypatch):
    calls = _patch_requests(monkeypatch, [_ok(" hello ")])
    client = OpenAIChatClient("llama3.1", "http://host:11434/v1", api_key="ollama")
    out = client.complete([{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=16)
    assert out == "hello"
    post = calls["posts"][0]
    assert post["url"] == "http://host:11434/v1/chat/completions"
    assert post["json"]["model"] == "llama3.1" and post["json"]["stream"] is False
    assert post["headers"]["Authorization"] == "Bearer ollama"


def test_client_caches_and_does_not_call_twice(monkeypatch, tmp_path):
    calls = _patch_requests(monkeypatch, [_ok("cached")])
    client = OpenAIChatClient("m", "http://h/v1", cache_dir=tmp_path / "c")
    args = ([{"role": "user", "content": "q"}],)
    assert client.complete(*args, temperature=0.0, max_tokens=8) == "cached"
    assert client.complete(*args, temperature=0.0, max_tokens=8) == "cached"
    assert calls["n"] == 1  # second call served from disk


def test_client_retries_transient_errors_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls = _patch_requests(monkeypatch, [_FakeResponse(503, text="busy"), _ok("recovered")])
    client = OpenAIChatClient("m", "http://h/v1", max_retries=2)
    assert (
        client.complete([{"role": "user", "content": "q"}], temperature=0, max_tokens=8)
        == "recovered"
    )
    assert calls["n"] == 2


def test_client_does_not_retry_a_permanent_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls = _patch_requests(monkeypatch, [_FakeResponse(401, text="bad key")])
    client = OpenAIChatClient("m", "http://h/v1", max_retries=3)
    with pytest.raises(LLMError, match="HTTP 401"):
        client.complete([{"role": "user", "content": "q"}], temperature=0, max_tokens=8)
    assert calls["n"] == 1


def test_missing_model_or_base_url_fails():
    with pytest.raises(LLMError):
        OpenAIChatClient("", "http://h/v1")
    with pytest.raises(LLMError):
        OpenAIChatClient("m", "")


def test_make_client_reads_env_over_config(monkeypatch):
    from retailmind.config import LLMConfig

    monkeypatch.setenv("LLM_BASE_URL", "http://env/v1")
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    client = make_client(LLMConfig(model="cfg-model", base_url="http://cfg/v1"))
    assert client.url == "http://env/v1/chat/completions" and client.model == "env-model"
    assert client.api_key == "secret"
