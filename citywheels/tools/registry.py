"""Tool registry: the schemas given to the model and the dispatcher that runs them.

The dispatcher never raises. Every failure (unknown tool, broken JSON arguments,
unexpected exception inside a tool) becomes a JSON payload with an ``error`` key,
which the model can read and report -- so a bad booking number cannot crash the
program.
"""

from __future__ import annotations

import json
import traceback
from datetime import timedelta
from typing import Any, Callable

from ..config import resolve_now
from .bookings import get_booking
from .policies import search_policies
from .refunds import calculate_refund


def _current_time() -> dict:
    """The fixed "now" of the shop, so relative questions are answerable."""
    moment = resolve_now()
    return {
        "now": moment.isoformat(timespec="seconds"),
        "timezone": "Europe/Warsaw (UTC+02:00 in June)",
        "note": (
            "This fixed date is what every tool uses. Do not guess another date. "
            "Cancellation and refund numbers always come from calculate_refund."
        ),
    }


def _calculate_refund(booking_id: str, cancel_in_hours: float | None = None) -> dict:
    """Adapter that keeps the hypothetical scenario out of the core refund rules."""
    if cancel_in_hours in (None, ""):
        return calculate_refund(booking_id)
    try:
        hours = float(cancel_in_hours)
    except (TypeError, ValueError):
        return calculate_refund(booking_id)
    return calculate_refund(booking_id, now=resolve_now() + timedelta(hours=hours))


TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_policies",
            "description": (
                "Search the CityWheels policy documents (prices, add-ons, deposits, "
                "late return fees, cancellation and refund rules, opening hours, age "
                "limits, payment, helmets, accidents, contact details). Use this for "
                "every general question about the shop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, in a few keywords.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many passages to return (1-5). Default 3.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_booking",
            "description": (
                "Look up one booking by its booking number, for example CW-1002. "
                "Returns the booking details, or found=false when the number is "
                "unknown. Never guess booking details without calling this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {
                        "type": "string",
                        "description": (
                            "Booking number such as CW-1002. Digits only also work."
                        ),
                    }
                },
                "required": ["booking_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": (
                "Calculate, in code, how much money the customer gets back if they "
                "cancel this booking, following the CityWheels cancellation rules. "
                "Always use this tool for any refund amount and repeat its numbers "
                "exactly. Do not calculate refunds yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {
                        "type": "string",
                        "description": "Booking number such as CW-1002.",
                    },
                    "cancel_in_hours": {
                        "type": "number",
                        "description": (
                            "Optional hypothetical: how many hours from now the "
                            "customer would cancel. Only for 'what if I cancel "
                            "tomorrow / in two days' questions. Omit for a "
                            "cancellation now."
                        ),
                    },
                },
                "required": ["booking_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "The current date and time used by the shop (fixed, so answers are "
                "repeatable). Use it for questions like 'is my booking tomorrow'."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


TOOL_IMPLEMENTATIONS: dict[str, Callable[..., dict]] = {
    "search_policies": search_policies,
    "get_booking": get_booking,
    "calculate_refund": _calculate_refund,
    "get_current_time": _current_time,
}

TOOL_NAMES: tuple[str, ...] = tuple(TOOL_IMPLEMENTATIONS)


def _as_arguments(arguments: Any) -> dict:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, dict):
        return arguments
    parsed = json.loads(arguments)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def execute_tool(name: str, arguments: Any = None) -> str:
    """Run one tool call and return its result as a JSON string. Never raises."""
    implementation = TOOL_IMPLEMENTATIONS.get(name)
    if implementation is None:
        return json.dumps(
            {
                "error": "unknown_tool",
                "message": (
                    f"There is no tool called {name!r}. Available tools: "
                    + ", ".join(TOOL_NAMES)
                    + "."
                ),
            }
        )
    try:
        kwargs = _as_arguments(arguments)
    except (json.JSONDecodeError, ValueError) as exc:
        return json.dumps(
            {
                "error": "invalid_arguments",
                "tool": name,
                "message": f"Could not read the arguments for {name}: {exc}",
            }
        )
    try:
        return json.dumps(implementation(**kwargs), ensure_ascii=False, default=str)
    except TypeError as exc:
        return json.dumps(
            {
                "error": "invalid_arguments",
                "tool": name,
                "message": f"Wrong arguments for {name}: {exc}",
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        return json.dumps(
            {
                "error": "tool_failed",
                "tool": name,
                "message": f"{type(exc).__name__}: {exc}",
            }
        )

