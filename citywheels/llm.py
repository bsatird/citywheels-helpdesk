"""A small OpenAI-compatible chat client.

Gemini (AI Studio), GitHub Models, OpenAI and a local Ollama all speak the same
``POST /chat/completions`` protocol with ``tools``, so one ~120-line client covers
every free option of the assignment. ``LLM_PROVIDER`` picks the preset; every value
can be overridden from the environment.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PROVIDER = "gemini"

PRESETS: dict[str, dict[str, Any]] = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        # Gemini 3.x needs the thought signature echoed back with every tool call;
        # gemini-2.5-flash-lite is retired, so pin a current flash model.
        "model": "gemini-3.5-flash",
        "key_vars": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "docs": "https://aistudio.google.com/apikey (free tier)",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_vars": ("OPENAI_API_KEY",),
        "docs": "https://platform.openai.com/api-keys (paid)",
    },
    "github": {
        "base_url": "https://models.github.ai/inference",
        "model": "openai/gpt-4o-mini",
        "key_vars": ("GITHUB_TOKEN", "GITHUB_MODELS_TOKEN"),
        "docs": "https://github.com/settings/tokens (free with a GitHub account)",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b",
        "key_vars": (),
        "docs": "https://ollama.com (local, no key)",
    },
}


class LLMError(RuntimeError):
    """Raised when the provider cannot be reached or returns an unusable answer."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = "{}"
    #: Provider extras that must be echoed back with the call. Gemini 3.x puts a
    #: ``thought_signature`` here and rejects the next request without it.
    extra: dict = field(default_factory=dict)

    def parsed_arguments(self) -> dict:
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


@dataclass
class ChatResult:
    """One model turn: either text, or tool calls (or both)."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def _first_env(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


@dataclass
class LLMSettings:
    provider: str
    model: str
    base_url: str
    api_key: str | None
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "LLMSettings":
        provider = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
        preset = PRESETS.get(provider)
        if preset is None:
            raise LLMError(
                f"Unknown LLM_PROVIDER={provider!r}. "
                f"Choose one of: {', '.join(sorted(PRESETS))}."
            )
        api_key = os.getenv("LLM_API_KEY") or _first_env(preset["key_vars"])
        base_url = os.getenv("LLM_BASE_URL") or preset["base_url"]
        model = os.getenv("LLM_MODEL") or preset["model"]
        try:
            temperature = float(os.getenv("LLM_TEMPERATURE") or 0.0)
        except ValueError:
            temperature = 0.0
        return cls(provider, model, base_url, api_key, temperature)

    def missing_key_message(self) -> str:
        preset = PRESETS[self.provider]
        return (
            f"No API key found for provider {self.provider!r}. Set "
            f"{' or '.join(preset['key_vars']) or 'LLM_API_KEY'} in your .env file. "
            f"Get a key here: {preset['docs']}. See .env.example."
        )


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


_DELAY_IN_TEXT = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_DELAY_IN_JSON = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)\s*s?"', re.IGNORECASE)


def _seconds(value: Any) -> float | None:
    """``7``, ``"7"`` and ``"7s"`` all mean seven seconds."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(float(value), 0.0)
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*s?\s*", value)
    return float(match.group(1)) if match else None


class LLMClient:
    """OpenAI-compatible chat client with retries on rate limits and 5xx errors."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        session: requests.Session | None = None,
        min_interval: float | None = None,
    ) -> None:
        self.settings = settings or LLMSettings.from_env()
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        # Free tiers allow a handful of requests per minute; spacing the calls out
        # is cheaper than hammering the endpoint until it answers 429.
        self.min_interval = (
            _float_env("LLM_MIN_INTERVAL", 0.0) if min_interval is None else float(min_interval)
        )
        self._last_request_at = 0.0

    # -- request plumbing -------------------------------------------------
    @property
    def endpoint(self) -> str:
        return f"{self.settings.base_url.rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    @staticmethod
    def _to_message(payload: dict) -> dict:
        """Our messages are already OpenAI-shaped; only guard the null content."""
        message = {key: value for key, value in payload.items() if value is not None}
        if "content" not in message:
            message["content"] = ""
        return message

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResult:
        """Send one turn to the model and normalise the answer."""
        if not self.settings.api_key and self.settings.provider != "ollama":
            raise LLMError(self.settings.missing_key_message())

        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [self._to_message(m) for m in messages],
            "temperature": self.settings.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        data = self._post(payload)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response from the provider: {data!r}") from exc

        calls = [
            ToolCall(
                id=call.get("id") or f"call_{index}",
                name=(call.get("function") or {}).get("name", ""),
                arguments=(call.get("function") or {}).get("arguments") or "{}",
                extra=call.get("extra_content") or {},
            )
            for index, call in enumerate(message.get("tool_calls") or [])
        ]
        return ChatResult(
            content=(message.get("content") or "").strip(),
            tool_calls=calls,
            raw=data,
        )

    def _post(self, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            retry_after: float | None = None
            self._throttle()
            try:
                response = self.session.post(
                    self.endpoint,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
            else:
                if response.status_code < 400:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise LLMError(
                            f"The provider did not return JSON: {response.text[:300]}"
                        ) from exc
                if response.status_code in (408, 409, 429) or response.status_code >= 500:
                    last_error = LLMError(
                        f"HTTP {response.status_code}: {response.text[:300]}"
                    )
                    retry_after = self._retry_after(response)
                else:
                    raise LLMError(
                        f"HTTP {response.status_code} from {self.endpoint}: "
                        f"{response.text[:500]}"
                    )
            if attempt < self.max_retries:
                time.sleep(self._backoff(attempt, retry_after))
        raise LLMError(
            f"Could not reach the model after {self.max_retries + 1} attempts: {last_error}"
        )

    @staticmethod
    def _retry_after(response: Any) -> float | None:
        """The provider's own "wait this long" hint, in seconds.

        OpenAI sends a ``Retry-After`` header; Gemini sends nothing in the headers
        and puts the delay in the JSON body instead
        (``{"error": {"details": [{"retryDelay": "23s"}]}}`` and a "Please retry in
        23.6s." message), so both places are checked.
        """
        header = getattr(response, "headers", None) or {}
        from_header = _seconds(header.get("Retry-After"))
        if from_header is not None:
            return min(from_header, 60.0)

        body = getattr(response, "text", "") or ""
        if hasattr(response, "json"):
            try:
                body += json.dumps(response.json())
            except Exception:  # noqa: BLE001 - the body may not be JSON at all
                pass
        for pattern in (_DELAY_IN_JSON, _DELAY_IN_TEXT):
            match = pattern.search(body)
            if match:
                return min(float(match.group(1)), 60.0)
        return None

    def _throttle(self) -> None:
        """Keep at least ``min_interval`` seconds between two requests."""
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        if self._last_request_at:
            wait = self.min_interval - (now - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _backoff(self, attempt: int, retry_after: float | None = None) -> float:
        """Exponential backoff, never shorter than what the provider asked for.

        Free tiers answer 429 quickly and ask you to wait a few seconds; ignoring
        that hint just burns the remaining attempts.
        """
        delay = min(0.6 * (2**attempt), 20.0)
        if retry_after is not None:
            delay = max(delay, retry_after)
        return delay


def build_client() -> LLMClient:
    """Client configured from the environment (used by the CLI and the evaluation)."""
    return LLMClient(LLMSettings.from_env())

