from __future__ import annotations

from datetime import date, datetime
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
