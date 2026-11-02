"""Tool schemas and the dispatcher: it must never raise, whatever the model sends."""

from __future__ import annotations

import json

import pytest

from citywheels.tools import registry
from citywheels.tools.registry import TOOL_NAMES, TOOL_SPECS, execute_tool


def test_the_four_tools_are_registered():
    assert set(TOOL_NAMES) == {
        "search_policies",
        "get_booking",
        "calculate_refund",
        "get_current_time",
    }
    assert len(TOOL_SPECS) == len(TOOL_NAMES)


def test_every_schema_is_well_formed():
    for spec in TOOL_SPECS:
        assert spec["type"] == "function"
        function = spec["function"]
        name = function["name"]
        assert name in registry.TOOL_IMPLEMENTATIONS
        assert len(function["description"]) > 20
        parameters = function["parameters"]
        assert parameters["type"] == "object"
        assert isinstance(parameters.get("properties"), dict)
        for required in parameters.get("required", []):
            assert required in parameters["properties"]


def test_current_time_is_the_fixed_now():
    payload = json.loads(execute_tool("get_current_time"))
    assert payload["now"] == "2026-06-10T09:00:00+02:00"


def test_arguments_may_be_a_json_string():
    payload = json.loads(execute_tool("get_booking", '{"booking_id": "1002"}'))
    assert payload["found"] is True
    assert payload["booking_id"] == "CW-1002"


def test_missing_arguments_are_treated_as_empty():
    assert json.loads(execute_tool("get_current_time", None))["now"]


def test_unknown_tool_is_reported_as_json():
    payload = json.loads(execute_tool("drop_database", "{}"))
    assert payload["error"] == "unknown_tool"
    assert "search_policies" in payload["message"]


def test_broken_json_arguments_are_reported():
    payload = json.loads(execute_tool("get_booking", '{"booking_id": '))
    assert payload["error"] == "invalid_arguments"


def test_wrong_argument_names_are_reported():
    payload = json.loads(execute_tool("get_booking", {"booking": "CW-1001"}))
    assert payload["error"] == "invalid_arguments"
    assert "get_booking" in payload["message"]


def test_a_crashing_tool_does_not_escape(monkeypatch):
    def boom(**_kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(registry.TOOL_IMPLEMENTATIONS, "boom", boom)
    payload = json.loads(execute_tool("boom", {}))
    assert payload["error"] == "tool_failed"
    assert "kaboom" in payload["message"]


# --- the hypothetical "what if I cancel later" clock ------------------------
def test_hypothetical_cancellation_moves_the_clock():
    # CW-1001 starts 2026-06-13 10:00, which is 73 h after "now"
    now = json.loads(execute_tool("calculate_refund", {"booking_id": "CW-1001"}))
    assert now["hours_before_start"] == 73.0
    assert now["refund_tier"] == "full"

    later = json.loads(
        execute_tool("calculate_refund", {"booking_id": "CW-1001", "cancel_in_hours": 50})
    )
    assert later["hours_before_start"] == 23.0
    assert later["refund_tier"] == "none"


@pytest.mark.parametrize("value", ["soon", None, ""])
def test_unusable_cancel_in_hours_falls_back_to_now(value):
    payload = json.loads(
        execute_tool("calculate_refund", {"booking_id": "CW-1001", "cancel_in_hours": value})
    )
    assert payload["hours_before_start"] == 73.0
