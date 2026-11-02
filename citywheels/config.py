"""Runtime configuration: paths, the fixed "current date" and the contact details.

Everything that a test or the evaluation might want to change is read from the
environment, so the tools stay deterministic:

* ``CITYWHEELS_NOW``       -- the "current" date/time (default 2026-06-10T09:00:00+02:00)
* ``CITYWHEELS_DATA_DIR``  -- where ``bookings.json`` and ``policies/`` live
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# <repo>/citywheels/config.py -> <repo>
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("CITYWHEELS_DATA_DIR") or PROJECT_ROOT / "data")
POLICIES_DIR = DATA_DIR / "policies"
BOOKINGS_PATH = DATA_DIR / "bookings.json"

#: The bookings are set in June 2026, so Europe/Warsaw is on CEST (UTC+2).
#: ``zoneinfo`` needs the IANA database, which is not bundled on Windows; fall
#: back to a fixed offset so the agent behaves the same everywhere.
try:  # pragma: no cover - depends on the host
    from zoneinfo import ZoneInfo

    SHOP_TZ: timezone | ZoneInfo = ZoneInfo("Europe/Warsaw")
except Exception:  # pragma: no cover - Windows without tzdata
    SHOP_TZ = timezone(timedelta(hours=2), "Europe/Warsaw")

FIXED_NOW = "2026-06-10T09:00:00+02:00"

SUPPORT_EMAIL = "help@citywheels.example"
SUPPORT_PHONE = "+48 000 000 000"


def parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 string into a timezone-aware datetime.

    Naive values are assumed to be shop-local time. A trailing ``Z`` is accepted
    because ``datetime.fromisoformat`` only understands it from Python 3.11.
    """
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SHOP_TZ)
    return parsed


def resolve_now(value: str | None = None) -> datetime:
    """Return the "current" moment used by every time-dependent tool.

    An explicit ``value`` wins over the ``CITYWHEELS_NOW`` variable, which wins
    over :data:`FIXED_NOW`. The result is always timezone aware.
    """
    if value is not None:
        raw = value
    else:
        raw = os.getenv("CITYWHEELS_NOW") or FIXED_NOW
    return parse_datetime(raw).astimezone(SHOP_TZ)
