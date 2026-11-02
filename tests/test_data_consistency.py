"""The bookings in ``data/`` must agree with the published price list.

These tests protect the ground truth the agent is graded against: if someone edits
``data/bookings.json`` or ``data/policies/prices.md``, the mismatch shows up here
instead of in the evaluation.
"""

from __future__ import annotations

import math

import pytest

from citywheels.config import parse_datetime
from citywheels.tools.bookings import load_bookings

# data/policies/prices.md
HOURLY = {"city": 4, "e-bike": 9, "kids": 2}
DAILY = {"city": 20, "e-bike": 40, "kids": 10}
DEPOSIT = {"city": 50, "e-bike": 150, "kids": 30}
CHILD_SEAT_PER_DAY = 3
WEEK_DISCOUNT = 0.15

BOOKINGS = load_bookings()
IDS = [booking["id"] for booking in BOOKINGS]


def rental_hours(booking: dict) -> float:
    start = parse_datetime(booking["start"])
    end = parse_datetime(booking["end"])
    return (end - start).total_seconds() / 3600.0


def expected_rental_price(booking: dict) -> float:
    hours = rental_hours(booking)
    days = max(1, math.ceil(hours / 24))
    bike_type = booking["bike_type"]
    if hours > 6:
        per_bike = days * DAILY[bike_type]
        if days >= 7:
            per_bike *= 1 - WEEK_DISCOUNT
    else:
        per_bike = math.ceil(hours) * HOURLY[bike_type]
    total = booking["bikes"] * per_bike
    if "child_seat" in booking["add_ons"]:
        total += CHILD_SEAT_PER_DAY * days
    return round(total, 2)


def test_the_expected_bookings_are_there():
    assert IDS == [f"CW-{number}" for number in range(1001, 1011)]


@pytest.mark.parametrize("booking", BOOKINGS, ids=IDS)
def test_rental_price_matches_the_price_list(booking):
    assert booking["rental_price_eur"] == expected_rental_price(booking)


@pytest.mark.parametrize("booking", BOOKINGS, ids=IDS)
def test_deposit_matches_the_deposit_table(booking):
    assert booking["deposit_eur"] == DEPOSIT[booking["bike_type"]] * booking["bikes"]


@pytest.mark.parametrize("booking", BOOKINGS, ids=IDS)
def test_bookings_have_the_fields_the_tools_need(booking):
    required = {
        "id",
        "status",
        "bike_type",
        "bikes",
        "start",
        "end",
        "add_ons",
        "rental_price_eur",
        "booking_fee_eur",
        "deposit_eur",
        "paid_with",
    }
    assert required <= set(booking)
    assert booking["status"] in {"confirmed", "in_progress", "completed", "cancelled"}
    assert booking["bike_type"] in HOURLY
    assert booking["paid_with"] in {"card", "voucher"}
    assert booking["booking_fee_eur"] in {0, 2}
    assert set(booking["add_ons"]) <= {"child_seat", "extra_lock", "helmet"}
    assert parse_datetime(booking["end"]) > parse_datetime(booking["start"])


def test_ids_are_unique_and_upper_case():
    assert len(set(IDS)) == len(IDS)
    assert all(booking_id == booking_id.upper() for booking_id in IDS)
