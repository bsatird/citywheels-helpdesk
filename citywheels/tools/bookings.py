"""Lookups in ``data/bookings.json``.

The tool never raises for a bad input: an unknown or malformed booking number
comes back as a ``found: false`` payload that the model can report to the
customer (this is one of the assignment's behaviour requirements).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .. import config

_ID_RE = re.compile(r"^(CW-?)?0*(\d{1,6})", re.IGNORECASE)


def normalize_booking_id(raw: object) -> str:
    """Turn user input such as ``" cw 1002 "`` or ``"1002"`` into ``"CW-1002"``.

    Anything that cannot be understood is returned upper-cased and trimmed, so
    the caller can still echo it back in the "not found" message.
    """
    text = str(raw or "").strip().upper()
    text = re.sub(r"[\s_]+", "-", text)
    text = text.strip("-")
    match = _ID_RE.match(text)
    if match:
        return f"CW-{int(match.group(2)):04d}"
    return text


def load_bookings(path: Path | str | None = None) -> list[dict]:
    """Read all bookings from disk."""
    booking_path = Path(path) if path is not None else config.BOOKINGS_PATH
    with booking_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{booking_path} must contain a JSON list of bookings")
    return data


def find_booking(
    booking_id: object, bookings: list[dict] | None = None
) -> dict | None:
    """Return the booking with that id, or ``None``."""
    target = normalize_booking_id(booking_id)
    if not target:
        return None
    source = bookings if bookings is not None else load_bookings()
    for booking in source:
        if str(booking.get("id", "")).upper() == target:
            return booking
    return None


def get_booking(booking_id: object, *, bookings: list[dict] | None = None) -> dict:
    """Tool implementation: booking details or a clear "not found" result."""
    normalized = normalize_booking_id(booking_id)
    if not normalized:
        return {
            "found": False,
            "booking_id": None,
            "error": "missing_booking_id",
            "message": (
                "No booking number was given. Ask the customer for their booking "
                f"number (it looks like CW-1001) or send them to {config.SUPPORT_EMAIL}."
            ),
        }

    booking = find_booking(normalized, bookings=bookings)
    if booking is None:
        return {
            "found": False,
            "booking_id": normalized,
            "error": "not_found",
            "message": (
                f"Booking {normalized} was not found in the CityWheels system. "
                "Say that you could not find it, do not guess any details, and ask "
                "the customer to check the number or contact the staff at "
                f"{config.SUPPORT_EMAIL} / {config.SUPPORT_PHONE}."
            ),
        }

    return {
        "found": True,
        "booking_id": booking["id"],
        "booking": booking,
    }
