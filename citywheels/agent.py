"""The agent loop: model -> tool -> model -> ... -> final answer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

from .config import SUPPORT_EMAIL, SUPPORT_PHONE
from .guardrails import unsupported_amounts
from .llm import ChatResult, ToolCall
from .prompts import SYSTEM_PROMPT
from .tools.registry import TOOL_SPECS, execute_tool

MAX_ITERATIONS = 6

EMPTY_ANSWER_FALLBACK = (
    "Sorry, I could not put an answer together for that. Please contact the "
    f"CityWheels staff at {SUPPORT_EMAIL} or {SUPPORT_PHONE}."
)

MAX_ITERATIONS_FALLBACK = (
    "Sorry, I could not finish checking this. Please contact the CityWheels staff "
    f"at {SUPPORT_EMAIL} or {SUPPORT_PHONE} and they will help you."
)


class ChatModel(Protocol):
    """Anything with the same ``chat`` signature as :class:`citywheels.llm.LLMClient`."""

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResult:
        ...


@dataclass
class ToolInvocation:
    """One executed tool call, kept for the trace and the evaluation."""

    name: str
    arguments: dict
    result: str
    error: str | None = None

    @property
    def payload(self) -> dict:
        try:
            value = json.loads(self.result)
        except (json.JSONDecodeError, TypeError):
            return {}
        return value if isinstance(value, dict) else {}

    def short(self, limit: int = 160) -> str:
        text = " ".join(self.result.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class AgentResult:
    """What happened during one question."""

    question: str
    answer: str
    tool_calls: list[str] = field(default_factory=list)
    invocations: list[ToolInvocation] = field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = "final_answer"
    history: list[dict] = field(default_factory=list)
    unsupported_amounts: list[float] = field(default_factory=list)

    @property
    def used_tools(self) -> list[str]:
        return list(dict.fromkeys(self.tool_calls))


class HelpDeskAgent:
    """Runs the tool-calling loop until the model answers or the budget is gone."""

    def __init__(
        self,
        llm: ChatModel,
        *,
        tool_executor: Callable[[str, object], str] = execute_tool,
        tool_specs: list[dict] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.llm = llm
        self.tool_executor = tool_executor
        self.tool_specs = TOOL_SPECS if tool_specs is None else tool_specs
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

    def run(self, question: str, history: Iterable[dict] | None = None) -> AgentResult:
        """Answer one question. ``history`` is the transcript of earlier turns."""
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(dict(message) for message in history or [])
        messages.append({"role": "user", "content": question})

        invocations: list[ToolInvocation] = []
        tool_calls: list[str] = []
        answer: str | None = None
        stopped_reason = "max_iterations"
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            reply = self.llm.chat(messages, self.tool_specs)

            if not reply.has_tool_calls:
                text = (reply.content or "").strip()
                answer = text or EMPTY_ANSWER_FALLBACK
                stopped_reason = "final_answer" if text else "empty_response"
                # Keep the answer in the transcript so follow-up questions in the
                # REPL can refer to it.
                messages.append({"role": "assistant", "content": answer})
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [
                        _tool_call_payload(call) for call in reply.tool_calls
                    ],
                }
            )
            for call in reply.tool_calls:
                raw = self.tool_executor(call.name, call.arguments)
                tool_calls.append(call.name)
                invocation = ToolInvocation(
                    name=call.name,
                    arguments=call.parsed_arguments(),
                    result=raw,
                    error=_error_of(raw),
                )
                invocations.append(invocation)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": raw}
                )

        if answer is None:
            answer = MAX_ITERATIONS_FALLBACK

        return AgentResult(
            question=question,
            answer=answer,
            tool_calls=tool_calls,
            invocations=invocations,
            iterations=iterations,
            stopped_reason=stopped_reason,
            history=messages[1:],
            unsupported_amounts=unsupported_amounts(
                answer, [invocation.result for invocation in invocations]
            ),
        )


def _tool_call_payload(call: ToolCall) -> dict:
    """Shape one tool call for the assistant message we send back.

    ``extra_content`` (Gemini's ``thought_signature``) has to travel with the call
    or Gemini 3.x answers HTTP 400; other providers do not return it and must not
    receive an unknown field.
    """
    payload = {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments},
    }
    if call.extra:
        payload["extra_content"] = call.extra
    return payload


def _error_of(raw_result: str) -> str | None:
    try:
        payload = json.loads(raw_result)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(payload, dict):
        error = payload.get("error")
        return str(error) if error else None
    return None
