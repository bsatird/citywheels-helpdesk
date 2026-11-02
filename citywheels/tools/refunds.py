"""``calculate_refund`` -- the cancellation money is computed here, never by the model.

Rules (``data/policies/cancellations.md``):

* 48 hours or more before the start -> 100% of the rental price,
* less than 48 but at least 24 hours   -> 50%,
* less than 24 hours before the start  -> 0%.

The booking fee is never refunded, the deposit is always refunded in full, and
``in_progress`` / ``completed`` / ``cancelled`` bookings cannot be cancelled at all.

Boundaries are inclusive on the lower end: exactly 48 h is a full refund, exactly
24 h is a half refund. Money is handled with :class:`~decimal.Decimal` and rounded
half-up to cents.
"""

from __future__ import annotations

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

from ..config import SUPPORT_EMAIL, SUPPORT_PHONE, parse_datetime, resolve_now
from .bookings import find_booking, normalize_booking_id

FULL_REFUND_MIN_HOURS = 48
HALF_REFUND_MIN_HOURS = 24
CANCELLABLE_STATUS = "confirmed"
NON_CANCELLABLE_STATUSES = ("in_progress", "completed", "cancelled")

POLICY_REF = "data/policies/cancellations.md#cancelling-a-booking"
POLICY_REF_MONEY = "data/policies/cancellations.md#how-the-money-comes-back"

_CENT = Decimal("0.01")


def money(value: object) -> Decimal:
    """Convert a value to a Decimal rounded half-up to cents."""
    return Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP)


def refund_rate(hours_before_start: float) -> tuple[str, Decimal]:
    """Map "hours until the rental starts" to a refund tier and rate.

    ``hours_before_start`` may be negative (the rental already started).
    """
    if hours_before_start >= FULL_REFUND_MIN_HOURS:
        return "full", Decimal("1")
    if hours_before_start >= HALF_REFUND_MIN_HOURS:
        return "half", Decimal("0.5")
    return "none", Decimal("0")


def hours_before_start(booking: dict, now: datetime) -> float:
    """Hours between ``now`` and the start of the booking (negative once started)."""
    start = parse_datetime(booking["start"])
    return (start - now).total_seconds() / 3600.0


def _format_hours(hours: float) -> str:
    if hours < 0:
        return f"{abs(hours):.1f} hours after the start"
    return f"{hours:.1f} hours before the start"


def _not_found(booking_id: object) -> dict:
    normalized = normalize_booking_id(booking_id)
    return {
        "found": False,
        "booking_id": normalized or None,
        "cancellable": False,
        "rental_refund_eur": 0.0,
        "booking_fee_refund_eur": 0.0,
        "deposit_refund_eur": 0.0,
        "total_refund_eur": 0.0,
        "error": "not_found",
        "explanation": (
            f"Booking {normalized or '(none)'} was not found, so no refund can be "
            "calculated. Do not invent an amount; tell the customer the booking was "
            f"not found and to contact the staff at {SUPPORT_EMAIL} / {SUPPORT_PHONE}."
        ),
    }


def calculate_refund(
    booking_id: object,
    now: str | datetime | None = None,
    *,
    bookings: list[dict] | None = None,
) -> dict:
    """Calculate what the customer gets back for cancelling ``booking_id`` now.

    Parameters
    ----------
    booking_id:
        Booking number, normalised before lookup (``"1002"`` also works).
    now:
        Override for the current moment (tests and the evaluation use this).
        Defaults to :func:`citywheels.config.resolve_now`.
    bookings:
        Optional pre-loaded booking list (tests pass their own fixtures).
    """
    booking = find_booking(booking_id, bookings=bookings)
    if booking is None:
        return _not_found(booking_id)

    moment = parse_datetime(now) if isinstance(now, str) else (now or resolve_now())
    status = str(booking.get("status", "")).lower()

    if status != CANCELLABLE_STATUS:
        return {
            "found": True,
            "booking_id": booking["id"],
            "cancellable": False,
            "status": status,
            "rental_refund_eur": 0.0,
            "booking_fee_refund_eur": 0.0,
            "deposit_refund_eur": 0.0,
            "total_refund_eur": 0.0,
            "policy_refs": [POLICY_REF],
            "explanation": (
                f"Booking {booking['id']} has the status '{status}'. Only bookings "
                "with the status 'confirmed' can be cancelled; for this booking "
                "nothing is refunded. Do not offer a refund, explain the status and "
                f"tell the customer to contact the staff at {SUPPORT_EMAIL} / "
                f"{SUPPORT_PHONE} if they think this is wrong."
            ),
        }

    hours = hours_before_start(booking, moment)
    tier, rate = refund_rate(hours)

    rental_price = money(booking.get("rental_price_eur", 0))
    rental_refund = money(rental_price * rate)
    booking_fee = money(booking.get("booking_fee_eur", 0))
    booking_fee_refund = money(0)  # never refunded
    deposit = money(booking.get("deposit_eur", 0))
    deposit_refund = deposit  # always refunded in full
    total = money(rental_refund + booking_fee_refund + deposit_refund)

    paid_with = str(booking.get("paid_with", "card")).lower()
    if paid_with == "voucher":
        method = "voucher"
        method_note = (
            "Paid with a voucher, so the refund comes back as a new voucher of the "
            "same value, not as money on a card."
        )
    else:
        method = "card"
        method_note = (
            "Paid by card, so the money is refunded to the same card within "
            "5 working days."
        )

    if tier == "full":
        tier_note = (
            f"Cancelling {_format_hours(hours)} is 48 hours or more before the start, "
            "so 100% of the rental price is refunded."
        )
    elif tier == "half":
        tier_note = (
            f"Cancelling {_format_hours(hours)} is less than 48 but at least 24 hours "
            "before the start, so 50% of the rental price is refunded."
        )
    else:
        tier_note = (
            f"Cancelling {_format_hours(hours)} is less than 24 hours before the start, "
            "so the rental price is not refunded."
        )

    explanation = " ".join(
        [
            f"Booking {booking['id']} (status: {status}) can be cancelled.",
            tier_note,
            f"Rental price {rental_price} EUR -> refund {rental_refund} EUR.",
            f"Booking fee {booking_fee} EUR is never refunded (0 EUR).",
            f"Deposit {deposit} EUR is refunded in full ({deposit_refund} EUR).",
            f"Total refund: {total} EUR. {method_note}",
            "State these amounts exactly as given and do not recalculate them.",
        ]
    )

    return {
        "found": True,
        "booking_id": booking["id"],
        "cancellable": True,
        "status": status,
        "now": moment.isoformat(timespec="seconds"),
        "booking_start": booking.get("start"),
        "hours_before_start": round(hours, 2),
        "refund_tier": tier,
        "refund_rate": float(rate),
        "rental_price_eur": float(rental_price),
        "rental_refund_eur": float(rental_refund),
        "booking_fee_eur": float(booking_fee),
        "booking_fee_refund_eur": float(booking_fee_refund),
        "deposit_eur": float(deposit),
        "deposit_refund_eur": float(deposit_refund),
        "total_refund_eur": float(total),
        "refund_method": method,
        "refund_method_note": method_note,
        "policy_refs": [POLICY_REF, POLICY_REF_MONEY],
        "explanation": explanation,
    }

