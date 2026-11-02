"""The guardrail that catches euro amounts no tool ever produced."""

from __future__ import annotations

import json

import pytest

from citywheels.guardrails import (
    money_in_text,
    numbers_in_tool_outputs,
    unsupported_amounts,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("You get 340 EUR back.", [340.0]),
        ("Total: €1 234.50 today", [1234.5]),
        ("The refund is 73 euro", [73.0]),
        ("It costs 4,50 €", [4.5]),
        ("340 EUR and again 340 EUR", [340.0]),
        ("no money here", []),
        ("", []),
        (None, []),
    ],
)
def test_money_in_text(text, expected):
    assert money_in_text(text) == expected


def test_amounts_copied_from_a_tool_result_are_supported():
    tool = json.dumps({"total_refund_eur": 340.0, "deposit_refund_eur": 300.0})
    answer = "You get 340 EUR back, including the 300 EUR deposit."
    assert unsupported_amounts(answer, [tool]) == []


def test_invented_amounts_are_flagged():
    tool = json.dumps({"total_refund_eur": 340.0})
    assert unsupported_amounts("You get 999 EUR back.", [tool]) == [999.0]


def test_comma_decimals_match_the_tool_numbers():
    tool = json.dumps({"total_refund_eur": 73.0})
    assert unsupported_amounts("That is 73,00 EUR", [tool]) == []


def test_numbers_inside_a_json_string_are_collected():
    payloads = [json.dumps({"a": {"b": [1, 2.5]}}), '{"c": 3}']
    assert numbers_in_tool_outputs(payloads) == {1.0, 2.5, 3.0}


def test_booleans_and_text_are_not_amounts():
    assert numbers_in_tool_outputs([{"found": True, "message": "no numbers"}]) == set()


def test_amounts_written_in_policy_prose_are_collected():
    """``search_policies`` returns the policy text, so its prices are prose."""
    payload = {
        "found": True,
        "results": [
            {
                "source": "data/policies/prices.md#bike-rental",
                "text": "| Electric bike (`e-bike`) | 9 EUR | 40 EUR |\n"
                "A day is up to 24 hours. Rentals of 7 days get 15% off.",
            }
        ],
    }
    assert numbers_in_tool_outputs([payload]) == {9.0, 40.0}


def test_amounts_quoted_from_policy_prose_are_supported():
    payload = {"text": "The deposit for an electric bike is 150 EUR."}
    assert unsupported_amounts("It is 150 EUR.", [payload]) == []


def test_amount_without_any_tool_output_is_flagged():
    assert unsupported_amounts("That will be 40 EUR.", []) == [40.0]
