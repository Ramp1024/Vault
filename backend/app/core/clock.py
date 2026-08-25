from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.core.config import settings


def local_timezone() -> ZoneInfo:
    """Return the configured application timezone, falling back to UTC.

    An unknown or misconfigured ``APP_TIMEZONE`` must never break retrieval, so
    an unresolved zone degrades to UTC rather than raising.
    """
    try:
        return ZoneInfo(settings.APP_TIMEZONE)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def today() -> date:
    """Return the current date in the configured timezone.

    This is the single reference point for resolving relative dates so the
    server's container timezone (typically UTC) never shifts "today" away from
    the user's wall clock.
    """
    return datetime.now(local_timezone()).date()


def local_day_bounds_to_utc(low_day: date, high_day: date) -> tuple[str, str]:
    """Convert an inclusive local-day span to UTC RFC3339 ``[start, end]`` instants.

    A local calendar day maps to a UTC instant window offset by the timezone, so
    a note stored as a UTC timestamp is matched against the day the user actually
    means. E.g. in Asia/Kolkata (+5:30) the day ``2026-07-13`` becomes
    ``2026-07-12T18:30:00Z .. 2026-07-13T18:29:59Z``. Used to translate a
    human/local date range into bounds comparable to UTC-stored timestamps.
    """
    tz = local_timezone()
    start = datetime.combine(low_day, time.min, tzinfo=tz).astimezone(timezone.utc)
    end = datetime.combine(high_day, time.max, tzinfo=tz).astimezone(timezone.utc)
    return start.isoformat(), end.isoformat()


def to_local_day(value: str) -> date | None:
    """Return the local calendar day of a stored ISO date/datetime string, or None.

    A timezone-aware datetime (e.g. ``2026-07-12T18:30:00Z``) is converted into
    the configured timezone before its day is taken, so a note edited late at
    night in UTC lands on the correct local day. A plain date string
    (``2026-07-13``) or naive datetime is returned by its own day unchanged.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.date()
    return parsed.astimezone(local_timezone()).date()
