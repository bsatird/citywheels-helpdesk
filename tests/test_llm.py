"""Tests for the provider client: request shape, parsing and retry behaviour.

No network is involved: a fake session returns canned responses, so the retry
path (429 with ``Retry-After``, 5xx, connection errors) is tested for free.
"""

from __future__ import annotations

import json

import pytest
import requests

from citywheels import llm
from citywheels.llm import ChatResult, LLMClient, LLMError, LLMSettings

BASE_URL = "https://example.test/v1"


def settings(**overrides) -> LLMSettings:
    values = {
        "provider": "gemini",
        "model": "test-model",
        "base_url": BASE_URL,
        "api_key": "test-key",
        "temperature": 0.0,
    }
    values.update(overrides)
    return LLMSettings(**values)


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = "", headers=None):
        self.status_code = status_code
        self.payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    """Returns the scripted responses in order and records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, headers=None, data=None, timeout=None):
        self.calls.append(
            {
                "url": url,
                "headers": headers or {},
                "payload": json.loads(data),
                "timeout": timeout,
            }
        )
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def answer_turn(content: str = "hello") -> FakeResponse:
    return FakeResponse(payload={"choices": [{"message": {"content": content}}]})


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Record backoff instead of waiting for it."""
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


def client(*responses, **kwargs):
    session = FakeSession(*responses)
    kwargs.setdefault("min_interval", 0.0)
    return LLMClient(settings(), session=session, **kwargs), session


# --------------------------------------------------------------------------- #
# request shape and parsing
# --------------------------------------------------------------------------- #


def test_chat_posts_an_openai_shaped_request(no_sleeping):
    client_, session = client(answer_turn("hi"))
    result = client_.chat([{"role": "user", "content": "hello"}], [{"type": "function"}])

    assert result.content == "hi"
    assert not result.has_tool_calls
    call = session.calls[0]
    assert call["url"] == f"{BASE_URL}/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["payload"]["model"] == "test-model"
    assert call["payload"]["tool_choice"] == "auto"
    assert call["payload"]["messages"] == [{"role": "user", "content": "hello"}]


def test_chat_omits_tools_when_none_are_given(no_sleeping):
    client_, session = client(answer_turn())
    client_.chat([{"role": "user", "content": "hello"}])
    assert "tools" not in session.calls[0]["payload"]


def test_null_content_becomes_an_empty_string(no_sleeping):
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "abc",
                            "function": {
                                "name": "get_booking",
                                "arguments": '{"booking_id": "CW-1002"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    client_, _ = client(FakeResponse(payload=payload))
    result = client_.chat([{"role": "user", "content": "where is CW-1002?"}])

    assert isinstance(result, ChatResult)
    assert result.content == ""
    assert result.has_tool_calls
    call = result.tool_calls[0]
    assert (call.id, call.name) == ("abc", "get_booking")
    assert call.parsed_arguments() == {"booking_id": "CW-1002"}


def test_tool_call_without_id_or_arguments_is_normalised(no_sleeping):
    payload = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "get_current_time"}}],
                }
            }
        ]
    }
    client_, _ = client(FakeResponse(payload=payload))
    call = client_.chat([{"role": "user", "content": "what time is it?"}]).tool_calls[0]

    assert call.id == "call_0"
    assert call.arguments == "{}"
    assert call.parsed_arguments() == {}

def test_the_gemini_thought_signature_is_kept_on_the_tool_call(no_sleeping):
    """Gemini 3.x rejects the next request when the signature is dropped."""
    signature = {"google": {"thought_signature": "abc123"}}
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "extra_content": signature,
                            "function": {"name": "get_booking", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "get_current_time", "arguments": "{}"},
                        },
                    ],
                }
            }
        ]
    }
    client_, _ = client(FakeResponse(payload=payload))
    calls = client_.chat([{"role": "user", "content": "hi"}]).tool_calls

    assert calls[0].extra == signature
    # providers that send no extras get an empty dict, not a missing attribute
    assert calls[1].extra == {}




def test_unexpected_body_raises_a_clear_error(no_sleeping):
    client_, _ = client(FakeResponse(payload={"nope": True}))
    with pytest.raises(LLMError, match="Unexpected response from the provider"):
        client_.chat([{"role": "user", "content": "hello"}])


def test_non_json_body_raises(no_sleeping):
    client_, _ = client(FakeResponse(text="<html>gateway</html>", payload=ValueError("no json")))
    with pytest.raises(LLMError, match="did not return JSON"):
        client_.chat([{"role": "user", "content": "hello"}])



# --------------------------------------------------------------------------- #
# retries and configuration
# --------------------------------------------------------------------------- #


def test_429_is_retried_and_retry_after_is_honoured(no_sleeping):
    client_, session = client(
        FakeResponse(429, text="slow down", headers={"Retry-After": "7"}),
        answer_turn("ok"),
    )
    assert client_.chat([{"role": "user", "content": "hello"}]).content == "ok"
    assert len(session.calls) == 2
    assert no_sleeping == [7.0]  # the provider asked for more than the backoff


def test_server_errors_are_retried(no_sleeping):
    client_, session = client(FakeResponse(503, text="upstream"), answer_turn("ok"))
    assert client_.chat([{"role": "user", "content": "hello"}]).content == "ok"
    assert len(session.calls) == 2
    assert no_sleeping == [0.6]


def test_connection_errors_are_retried(no_sleeping):
    client_, session = client(requests.ConnectionError("boom"), answer_turn("ok"))
    assert client_.chat([{"role": "user", "content": "hello"}]).content == "ok"
    assert len(session.calls) == 2


def test_the_client_gives_up_after_the_last_attempt(no_sleeping):
    client_, session = client(
        *[FakeResponse(500, text="down")] * 5,
        max_retries=3,
    )
    with pytest.raises(LLMError, match="after 4 attempts"):
        client_.chat([{"role": "user", "content": "hello"}])
    assert len(session.calls) == 4
    assert no_sleeping == [0.6, 1.2, 2.4]


def test_client_errors_are_not_retried(no_sleeping):
    client_, session = client(FakeResponse(401, text="bad key"))
    with pytest.raises(LLMError, match="HTTP 401"):
        client_.chat([{"role": "user", "content": "hello"}])
    assert len(session.calls) == 1
    assert no_sleeping == []


def test_retry_after_is_read_and_capped():
    assert LLMClient._retry_after(FakeResponse(headers={"Retry-After": "12"})) == 12.0
    assert LLMClient._retry_after(FakeResponse(headers={"Retry-After": "3600"})) == 60.0
    assert LLMClient._retry_after(FakeResponse(headers={"Retry-After": "soon"})) is None
    assert LLMClient._retry_after(FakeResponse()) is None


def test_retry_delay_is_read_from_a_gemini_error_body():
    """Gemini has no Retry-After header; the hint lives in the JSON payload."""
    body = {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota.",
            "details": [{"@type": "google.rpc.RetryInfo", "retryDelay": "23s"}],
        }
    }
    response = FakeResponse(429, payload=body, text=json.dumps(body))
    assert LLMClient._retry_after(response) == 23.0


def test_retry_delay_is_read_from_a_gemini_message():
    text = '{"error": {"message": "Quota exceeded. Please retry in 41.605723618s."}}'
    assert LLMClient._retry_after(FakeResponse(429, text=text)) == pytest.approx(41.6057, abs=1e-3)


def test_a_429_with_a_body_hint_waits_that_long(no_sleeping):
    body = {"error": {"details": [{"retryDelay": "23s"}]}}
    client_, _ = client(
        FakeResponse(429, payload=body, text=json.dumps(body)),
        answer_turn("ok"),
    )
    assert client_.chat([{"role": "user", "content": "hello"}]).content == "ok"
    assert no_sleeping == [23.0]


def test_min_interval_spaces_two_requests(monkeypatch):
    sleeps: list[float] = []
    clock = {"now": 100.0}
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        llm.time,
        "sleep",
        lambda seconds: (sleeps.append(seconds), clock.update(now=clock["now"] + seconds)),
    )

    client_, session = client(answer_turn("a"), answer_turn("b"), min_interval=5.0)
    client_.chat([{"role": "user", "content": "first"}])
    client_.chat([{"role": "user", "content": "second"}])

    assert len(session.calls) == 2
    assert sleeps == [5.0]


def test_min_interval_comes_from_the_environment(monkeypatch):
    session = FakeSession()
    monkeypatch.setenv("LLM_MIN_INTERVAL", "12")
    assert LLMClient(settings(), session=session).min_interval == 12.0
    monkeypatch.setenv("LLM_MIN_INTERVAL", "nonsense")
    assert LLMClient(settings(), session=session).min_interval == 0.0


def test_backoff_grows_and_is_capped():
    client_, _ = client(max_retries=0)
    delays = [client_._backoff(attempt) for attempt in range(7)]
    assert delays[:3] == [0.6, 1.2, 2.4]
    assert delays == sorted(delays)
    assert delays[-1] == 20.0  # capped instead of growing without bound
    assert client_._backoff(9) == 20.0
    assert client_._backoff(0, 9.0) == 9.0


def test_settings_come_from_the_environment(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "preset-key")

    cfg = LLMSettings.from_env()
    assert (cfg.provider, cfg.model, cfg.api_key) == (
        "gemini",
        "gemini-3.5-flash",
        "preset-key",
    )
    assert cfg.base_url.endswith("/v1beta/openai")


def test_llm_api_key_wins_over_the_preset_variable(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "preset-key")
    monkeypatch.setenv("LLM_API_KEY", "explicit-key")
    assert LLMSettings.from_env().api_key == "explicit-key"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nonsense")
    with pytest.raises(LLMError, match="Unknown LLM_PROVIDER"):
        LLMSettings.from_env()


def test_missing_key_is_reported_with_a_hint(monkeypatch):
    for name in ("LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    cfg = LLMSettings.from_env()
    assert cfg.api_key is None

    with pytest.raises(LLMError, match="No API key found"):
        LLMClient(cfg, session=FakeSession()).chat([{"role": "user", "content": "hi"}])


def test_temperature_is_read_and_falls_back_to_zero(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    assert LLMSettings.from_env().temperature == 0.3
    monkeypatch.setenv("LLM_TEMPERATURE", "hot")
    assert LLMSettings.from_env().temperature == 0.0


def test_ollama_needs_no_key(no_sleeping):
    session = FakeSession(answer_turn("ok"))
    local = LLMSettings("ollama", "qwen2.5:7b", "http://localhost:11434/v1", None)
    result = LLMClient(local, session=session).chat([{"role": "user", "content": "hi"}])

    assert result.content == "ok"
    assert "Authorization" not in session.calls[0]["headers"]
