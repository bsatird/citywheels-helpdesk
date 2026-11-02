"""``get_booking`` and the booking-number normaliser."""

from __future__ import annotations

import json

import pytest

from citywheels.tools.bookings import (
    find_booking,
    get_booking,
    load_bookings,
    normalize_booking_id,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CW-1002", "CW-1002"),
        ("cw-1002", "CW-1002"),
        ("1002", "CW-1002"),
        (" cw 1002 ", "CW-1002"),
        ("CW_1002", "CW-1002"),
        ("cw1002", "CW-1002"),
        ("CW-001002", "CW-1002"),
    ],
)
def test_booking_numbers_are_normalised(raw, expected):
    assert normalize_booking_id(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_empty_input_has_no_booking_number(raw):
    assert normalize_booking_id(raw) == ""


def test_unparseable_input_is_echoed_back():
    assert normalize_booking_id("hello") == "HELLO"


def test_get_booking_returns_the_real_booking():
    result = get_booking("cw 1004")
    assert result["found"] is True
    assert result["booking_id"] == "CW-1004"
    booking = result["booking"]
    assert booking["bike_type"] == "city"
    assert booking["add_ons"] == ["child_seat"]
    assert booking["paid_with"] == "voucher"


def test_unknown_booking_is_reported_not_raised():
    result = get_booking("CW-9999")
    assert result["found"] is False
    assert result["error"] == "not_found"
    assert "CW-9999" in result["message"]
    assert "not found" in result["message"].lower()


def test_missing_booking_number_asks_for_one():
    result = get_booking("")
    assert result["found"] is False
    assert result["error"] == "missing_booking_id"
    assert result["booking_id"] is None


def test_get_booking_accepts_its_own_booking_list(make_booking):
    bookings = [make_booking(id="CW-9001", status="in_progress")]
    result = get_booking("9001", bookings=bookings)
    assert result["found"] is True
    assert result["booking"]["status"] == "in_progress"


def test_find_booking_returns_none_for_an_unknown_id():
    assert find_booking("CW-0000") is None


def test_every_booking_in_the_file_can_be_read_back():
    for booking in load_bookings():
        found = get_booking(booking["id"])["booking"]
        assert found == booking


def test_result_is_json_serialisable():
    json.dumps(get_booking("CW-1001"))
    json.dumps(get_booking("CW-9999"))
