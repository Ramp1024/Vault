from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import dateparser

from app.core.clock import local_timezone
from app.core.config import settings

# Whole-period units the LLM may request. Boundary policy is owned HERE, never by
# the LLM, so it stays consistent: weeks run Monday-Sunday and months run from the
# 1st to the last day.
_RANGE_UNITS = frozenset({"week", "month"})


def _week_span(anchor: date) -> tuple[date, date]:
    """Return the inclusive [Monday, Sunday] of the week containing ``anchor``."""
    monday = anchor - timedelta(days=anchor.weekday())
    return monday, monday + timedelta(days=6)


def _month_span(anchor: date) -> tuple[date, date]:
    """Return the inclusive [first, last] day of the month containing ``anchor``."""
    first = anchor.replace(day=1)
    if first.month == 12:
        next_first = first.replace(year=first.year + 1, month=1)
    else:
        next_first = first.replace(month=first.month + 1)
    return first, next_first - timedelta(days=1)


def _snap_to_unit(anchor: date, unit: str) -> tuple[date, date] | None:
    """Expand a single date into the whole-period span for ``unit``, or None."""
    if unit == "week":
        return _week_span(anchor)
    if unit == "month":
        return _month_span(anchor)
    return None


def _parse_anchor(anchor: str, today: date) -> date | None:
    """Resolve a verbatim date phrase to a single date via dateparser.

    Anchored to ``today`` in the configured timezone so relative phrases match
    the user's wall clock. Returns None when the phrase cannot be resolved — the
    LLM never computes dates, dateparser owns all phrase-to-date resolution.
    """
    parsed = dateparser.parse(
        anchor,
        languages=["en"],
        settings={
            "RELATIVE_BASE": datetime.combine(today, time.min),
            "TIMEZONE": settings.APP_TIMEZONE,
            "PREFER_DATES_FROM": "past",
            "RETURN_AS_TIMEZONE_AWARE": False,
        },
    )
    return parsed.date() if parsed is not None else None


def _utc_bounds(low_day: date, high_day: date) -> tuple[str, str]:
    """Convert an inclusive day span to UTC RFC3339 [start, end] instants."""
    tz = local_timezone()
    start = datetime.combine(low_day, time.min, tzinfo=tz)
    end = datetime.combine(high_day, time.max, tzinfo=tz)
    return (
        start.astimezone(timezone.utc).isoformat(),
        end.astimezone(timezone.utc).isoformat(),
    )


def resolve_temporal_descriptor(
    descriptor: object, today: date, query: str
) -> tuple[str, str] | None:
    """Resolve an LLM temporal descriptor to UTC [low, high] bounds, or None.

    The descriptor is ``{"kind": "single"|"range", "unit": "week"|"month",
    "anchor": "<verbatim phrase>"}``: the LLM only classifies structure and copies
    the phrase, dateparser resolves the phrase to one date, and this module owns
    the range boundary policy. Anything malformed (missing/hallucinated anchor,
    unparseable phrase, unknown range unit) returns None so retrieval fails safe
    to plain semantic search.
    """
    if not isinstance(descriptor, dict):
        return None

    anchor = descriptor.get("anchor")
    if not isinstance(anchor, str) or not anchor.strip():
        return None

    # Anti-hallucination: the anchor must be copied from the user's question.
    normalized_anchor = " ".join(anchor.split()).lower()
    if normalized_anchor not in " ".join(query.split()).lower():
        return None

    day = _parse_anchor(anchor.strip(), today)
    if day is None:
        return None

    if str(descriptor.get("kind") or "single").strip().lower() == "range":
        span = _snap_to_unit(day, str(descriptor.get("unit") or "").strip().lower())
        if span is None:
            return None
    else:
        span = (day, day)

    return _utc_bounds(*span)
