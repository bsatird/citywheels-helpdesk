"""Shared fixtures.

The tools take ``now`` as an argument, so the tests never depend on the wall clock
or on the machine's time zone.
"""

from __future__ import annotations

import pytest

from citywheels.config import parse_datetime

#: The "current date" from data/README.md.
FIXED_NOW = "2026-06-10T09:00:00+02:00"


@pytest.fixture(autouse=True)
def _no_ambient_now(monkeypatch):
    """Make sure a CITYWHEELS_NOW from the shell never leaks into the tests."""
    monkeypatch.delenv("CITYWHEELS_NOW", raising=False)


@pytest.fixture
def now():
    return parse_datetime(FIXED_NOW)


@pytest.fixture
def make_booking():
    """Build a booking dict with sensible defaults (exactly 48 h before ``now``)."""

    def _make(**overrides):
        booking = {
            "id": "CW-9001",
            "status": "confirmed",
            "bike_type": "city",
            "bikes": 1,
            "start": "2026-06-12T09:00:00+02:00",
            "end": "2026-06-12T18:00:00+02:00",
            "add_ons": [],
            "rental_price_eur": 40.0,
            "booking_fee_eur": 2.0,
            "deposit_eur": 50.0,
            "paid_with": "card",
        }
        booking.update(overrides)
        return booking

    return _make
