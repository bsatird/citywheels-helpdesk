"""Guardrails: catch answers that quote money the tools never produced.

The model is told to copy amounts from ``calculate_refund``/``search_policies``, but
instructions alone are not a check. After every answer we extract the euro amounts
the model wrote and compare them with every number that came back from a tool -
both JSON numbers and amounts written in policy prose. The result is reported
(``AgentResult.unsupported_amounts``), printed by the CLI with ``--verbose`` and
counted by the evaluation.

This is a warning, not a block: the customer still gets an answer, but we can see
how often the model drifts.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

_NUMBER = r"\d{1,3}(?:[ \u00a0]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"

_MONEY_BEFORE = re.compile(rf"€\s*({_NUMBER})")
_MONEY_AFTER = re.compile(rf"({_NUMBER})\s*(?:€|eur\b|euros?\b|euro\b)", re.IGNORECASE)

_TOLERANCE = 0.011


def _to_float(raw: str) -> float | None:
    cleaned = raw.replace(" ", "").replace("\u00a0", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def money_in_text(text: str) -> list[float]:
    """All euro amounts written in ``text`` (deduplicated, in order)."""
    found: list[float] = []
    for pattern in (_MONEY_BEFORE, _MONEY_AFTER):
        for match in pattern.finditer(text or ""):
            value = _to_float(match.group(1))
            if value is not None and value not in found:
                found.append(value)
    return found


def _collect_numbers(value: Any, into: set[float], depth: int = 0) -> None:
    if depth > 8:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        into.add(round(float(value), 2))
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in "{[":
            try:
                _collect_numbers(json.loads(stripped), into, depth + 1)
                return
            except json.JSONDecodeError:
                pass
        # Prose carries money too: ``search_policies`` returns the policy text, so
        # "| Electric bike | 9 EUR | 40 EUR |" must support an answer that says
        # "40 EUR". Only amounts written with a currency count, never bare numbers
        # such as "24 hours" or "15%".
        into.update(round(amount, 2) for amount in money_in_text(stripped))
    elif isinstance(value, dict):
        for item in value.values():
            _collect_numbers(item, into, depth + 1)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_numbers(item, into, depth + 1)


def numbers_in_tool_outputs(payloads: Iterable[Any]) -> set[float]:
    """Every number that appeared in tool results, including nested JSON and prose."""
    numbers: set[float] = set()
    for payload in payloads:
        _collect_numbers(payload, numbers)
    return numbers


def _is_supported(amount: float, numbers: set[float]) -> bool:
    return any(abs(amount - number) <= _TOLERANCE for number in numbers)


def unsupported_amounts(answer: str, payloads: Iterable[Any]) -> list[float]:
    """Euro amounts in ``answer`` that no tool result can justify."""
    numbers = numbers_in_tool_outputs(payloads)
    return [amount for amount in money_in_text(answer) if not _is_supported(amount, numbers)]
