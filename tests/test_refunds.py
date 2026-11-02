"""``calculate_refund`` -- one test per rule of data/policies/cancellations.md."""

from __future__ import annotations

import pytest

from citywheels.config import parse_datetime
from citywheels.tools.refunds import calculate_refund, refund_rate

# 2026-06-10 09:00 +02:00 is "now", so these starts are exactly N hours away.
START_48H = "2026-06-12T09:00:00+02:00"
START_24H = "2026-06-11T09:00:00+02:00"
START_49H = "2026-06-12T10:00:00+02:00"
START_47H = "2026-06-12T08:00:00+02:00"
START_25H = "2026-06-11T10:00:00+02:00"
START_23H = "2026-06-11T08:00:00+02:00"
START_6H = "2026-06-10T15:00:00+02:00"
START_IN_PAST = "2026-06-10T08:00:00+02:00"


# --- the three refund tiers -------------------------------------------------
def test_full_refund_at_exactly_48_hours(make_booking, now):
    result = calculate_refund("CW-9001", now, bookings=[make_booking(start=START_48H)])
    assert result["refund_tier"] == "full"
    assert result["hours_before_start"] == 48.0
    assert result["rental_refund_eur"] == 40.0


def test_full_refund_above_48_hours(make_booking, now):
    result = calculate_refund("CW-9001", now, bookings=[make_booking(start=START_49H)])
    assert result["refund_tier"] == "full"
    assert result["rental_refund_eur"] == 40.0


def test_half_refund_at_exactly_24_hours(make_booking, now):
    result = calculate_refund("CW-9001", now, bookings=[make_booking(start=START_24H)])
    assert result["refund_tier"] == "half"
    assert result["hours_before_start"] == 24.0
    assert result["rental_refund_eur"] == 20.0


def test_half_refund_between_24_and_48_hours(make_booking, now):
    for start in (START_47H, START_25H):
        result = calculate_refund("CW-9001", now, bookings=[make_booking(start=start)])
        assert result["refund_tier"] == "half"
        assert result["rental_refund_eur"] == 20.0


def test_no_rental_refund_below_24_hours(make_booking, now):
    for start in (START_23H, START_6H):
        result = calculate_refund("CW-9001", now, bookings=[make_booking(start=start)])
        assert result["refund_tier"] == "none"
        assert result["rental_refund_eur"] == 0.0


def test_no_rental_refund_when_the_rental_already_started(make_booking, now):
    result = calculate_refund("CW-9001", now, bookings=[make_booking(start=START_IN_PAST)])
    assert result["refund_tier"] == "none"
    assert result["hours_before_start"] < 0
    assert result["rental_refund_eur"] == 0.0


@pytest.mark.parametrize(
    ("hours", "tier"),
    [
        (100, "full"),
        (48, "full"),
        (47.99, "half"),
        (24, "half"),
        (23.99, "none"),
        (-5, "none"),
    ],
)
def test_refund_rate_boundaries(hours, tier):
    assert refund_rate(hours)[0] == tier


# --- the fixed parts of the rule --------------------------------------------
def test_booking_fee_is_never_refunded(make_booking, now):
    booking = make_booking(start=START_48H, booking_fee_eur=2.0)
    result = calculate_refund("CW-9001", now, bookings=[booking])
    assert result["booking_fee_refund_eur"] == 0.0
    assert result["booking_fee_eur"] == 2.0
    # 40 rental + 50 deposit; the 2 EUR fee is not part of it
    assert result["total_refund_eur"] == 90.0


def test_deposit_is_always_refunded_in_full(make_booking, now):
    booking = make_booking(start=START_23H, deposit_eur=300.0)
    result = calculate_refund("CW-9001", now, bookings=[booking])
    assert result["deposit_refund_eur"] == 300.0
    assert result["total_refund_eur"] == 300.0  # 0% rental + full deposit


def test_total_is_rental_refund_plus_deposit(make_booking, now):
    booking = make_booking(start=START_25H, rental_price_eur=80.0, deposit_eur=300.0)
    result = calculate_refund("CW-9001", now, bookings=[booking])
    assert result["rental_refund_eur"] == 40.0
    assert result["total_refund_eur"] == 340.0


def test_half_refund_rounds_to_cents(make_booking, now):
    booking = make_booking(start=START_24H, rental_price_eur=23.01)
    result = calculate_refund("CW-9001", now, bookings=[booking])
    # 23.01 / 2 = 11.505 -> half-up -> 11.51
    assert result["rental_refund_eur"] == 11.51


# --- statuses that cannot be cancelled --------------------------------------
@pytest.mark.parametrize("status", ["in_progress", "completed", "cancelled"])
def test_non_cancellable_statuses_get_nothing(make_booking, now, status):
    booking = make_booking(status=status, start=START_48H)
    result = calculate_refund("CW-9001", now, bookings=[booking])
    assert result["found"] is True
    assert result["cancellable"] is False
    assert result["total_refund_eur"] == 0.0
    assert result["deposit_refund_eur"] == 0.0
    assert status in result["explanation"]
    assert "refund_method" not in result


# --- how the money comes back ------------------------------------------------
def test_card_booking_is_refunded_to_the_card(make_booking, now):
    result = calculate_refund(
        "CW-9001", now, bookings=[make_booking(start=START_48H, paid_with="card")]
    )
    assert result["refund_method"] == "card"
    assert "card" in result["refund_method_note"].lower()


def test_voucher_booking_is_refunded_as_a_voucher(make_booking, now):
    result = calculate_refund(
        "CW-9001", now, bookings=[make_booking(start=START_48H, paid_with="voucher")]
    )
    assert result["refund_method"] == "voucher"
    assert "voucher" in result["refund_method_note"].lower()


# --- bad input never crashes -------------------------------------------------
@pytest.mark.parametrize("booking_id", ["CW-9999", "", None, "not-a-booking"])
def test_unknown_booking_returns_not_found(booking_id):
    result = calculate_refund(booking_id)
    assert result["found"] is False
    assert result["total_refund_eur"] == 0.0
    assert result["cancellable"] is False
    assert result["explanation"]


# --- against the real booking file ------------------------------------------
@pytest.mark.parametrize(
    ("booking_id", "total", "method"),
    [
        ("CW-1001", 90.0, "card"),  # 73 h before the start -> 100%
        ("CW-1002", 340.0, "card"),  # 33 h -> 50%
        ("CW-1003", 50.0, "card"),  # 6 h -> 0% + deposit
        ("CW-1004", 73.0, "voucher"),  # voucher booking
        ("CW-1007", 186.0, "card"),  # exactly 48 h -> 100%
        ("CW-1009", 169.0, "card"),  # 121 h -> 100%
        ("CW-1010", 170.0, "card"),  # exactly 24 h -> 50%
    ],
)
def test_real_bookings_match_the_expected_amounts(booking_id, total, method):
    result = calculate_refund(booking_id)
    assert result["found"] is True
    assert result["total_refund_eur"] == total
    assert result["refund_method"] == method


@pytest.mark.parametrize("booking_id", ["CW-1005", "CW-1006", "CW-1008"])
def test_real_bookings_that_cannot_be_cancelled(booking_id):
    result = calculate_refund(booking_id)
    assert result["cancellable"] is False
    assert result["total_refund_eur"] == 0.0


def test_now_comes_from_the_environment_by_default(monkeypatch):
    monkeypatch.setenv("CITYWHEELS_NOW", "2026-06-11T09:00:00+02:00")
    # one day later CW-1002 (starts 2026-06-11 18:00) has only 9 h left
    result = calculate_refund("CW-1002")
    assert result["hours_before_start"] == 9.0
    assert result["refund_tier"] == "none"


def test_accepts_a_string_for_now(make_booking):
    booking = make_booking(start=START_48H)
    result = calculate_refund("CW-9001", "2026-06-10T09:00:00+02:00", bookings=[booking])
    assert result["refund_tier"] == "full"
    assert result["now"] == parse_datetime("2026-06-10T09:00:00+02:00").isoformat(
        timespec="seconds"
    )


