"""A fake model for tests and demos: no network, no API key.

``HelpDeskAgent`` only needs an object with ``chat(messages, tools) -> ChatResult``,
so a scripted list of turns is enough to exercise the whole tool-calling loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import ChatResult, ToolCall


@dataclass
class FakeLLM:
    """Replays a fixed script of :class:`ChatResult` turns."""

    turns: list[ChatResult] = field(default_factory=list)
    #: Recorded calls, so tests can assert what the model was shown.
    requests: list[dict] = field(default_factory=list)
    #: When True, the last scripted turn is reused if the script runs out.
    repeat_last: bool = False

    def __post_init__(self) -> None:
        self.turns = list(self.turns)
        self.requests = []
        self._index = 0

    @property
    def calls(self) -> int:
        """How many times the agent asked for a turn."""
        return len(self.requests)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResult:
        self.requests.append(
            {"messages": [json.loads(json.dumps(m)) for m in messages], "tools": tools}
        )
        if self._index >= len(self.turns):
            if self.repeat_last and self.turns:
                return self.turns[-1]
            raise AssertionError(
                f"FakeLLM ran out of scripted turns after {self._index} call(s)"
            )
        turn = self.turns[self._index]
        self._index += 1
        return turn


def text_turn(content: str) -> ChatResult:
    """A plain final answer."""
    return ChatResult(content=content)


def tool_turn(*calls: tuple[str, dict] | ToolCall, content: str = "") -> ChatResult:
    """A turn in which the model asks for one or more tools.

    ``tool_turn(("get_booking", {"booking_id": "CW-1002"}))``
    """
    tool_calls: list[ToolCall] = []
    for position, call in enumerate(calls, start=1):
        if isinstance(call, ToolCall):
            tool_calls.append(call)
            continue
        name, arguments = call
        tool_calls.append(
            ToolCall(
                id=f"call_{position}",
                name=name,
                arguments=json.dumps(arguments),
            )
        )
    return ChatResult(content=content, tool_calls=tool_calls)
