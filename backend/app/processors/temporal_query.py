from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta

import dateparser

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


def resolve_temporal_descriptor(
    descriptor: object, today: date, query: str
) -> tuple[str, str] | None:
    """Resolve a temporal descriptor to inclusive LOCAL date-only ``[low, high]``.

    The descriptor is ``{"kind": "single"|"range", "unit": "week"|"month",
    "anchor": "<verbatim phrase>"}``: only structure is classified and the phrase
    copied, dateparser resolves the phrase to one date, and this module owns the
    range boundary policy. The result is a pair of ``YYYY-MM-DD`` local days — the
    single human-meaningful representation used throughout; the storage/timezone
    translation is applied later by the filter builder/matcher. Anything malformed
    (missing/hallucinated anchor, unparseable phrase, unknown range unit) returns
    None so retrieval fails safe to plain semantic search.
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

    return _day_bounds(*span)


def _day_bounds(low_day: date, high_day: date) -> tuple[str, str]:
    """Return an inclusive local-day span as ``YYYY-MM-DD`` date-only strings."""
    return low_day.isoformat(), high_day.isoformat()


_MONTH = (
    r"(?:january|february|march|april|may|june|july|august|september|october"
    r"|november|december)"
)

# Deterministic temporal phrasings, tried in order. Each yields (unit, anchor,
# month_end): the anchor is a phrase dateparser can resolve; month_end snaps a
# month anchor to its final day before taking the week.
_TEMPORAL_PATTERNS: tuple[tuple[re.Pattern[str], str, bool], ...] = (
    (re.compile(rf"\blast week of ({_MONTH})\b"), "week", True),
    (re.compile(rf"\bfirst week of ({_MONTH})\b"), "week", False),
    (re.compile(r"\bweek of ([a-z]+(?:\s+\d{1,2})?(?:,?\s+\d{4})?)"), "week", False),
    (re.compile(r"\b(?:in|during|for|throughout)\s+(" + _MONTH + r")\b"), "month", False),
    (re.compile(r"\b(last week|this week|past week)\b"), "week", False),
    (re.compile(r"\b(last month|this month|past month)\b"), "month", False),
    (re.compile(r"\b(yesterday|today)\b"), "day", False),
)


def resolve_range(
    anchor: str, unit: str, today: date, *, month_end: bool = False
) -> tuple[str, str] | None:
    """Resolve a verbatim anchor phrase + unit to inclusive local-day bounds.

    dateparser owns phrase-to-date resolution; boundary policy (week = Mon–Sun,
    month = 1st–last) lives here. ``month_end`` snaps a month anchor to its final
    day first, so "last week of June" resolves to the week containing June's last
    day. Returns None when the phrase cannot be resolved.
    """
    day = _parse_anchor(anchor.strip(), today)
    if day is None:
        return None
    if month_end:
        day = _month_span(day)[1]
    if unit == "week":
        span = _week_span(day)
    elif unit == "month":
        span = _month_span(day)
    else:
        span = (day, day)
    return _day_bounds(*span)


def detect_temporal_range(query: str, today: date) -> tuple[str, str] | None:
    """Deterministically detect a temporal expression and resolve its day bounds.

    Recognizes the common natural phrasings ("week of July 13", "last week of
    June", "in August", "last week/month", "yesterday/today") without any LLM,
    returning inclusive local-day ``[low, high]`` bounds or None. No date
    arithmetic is hardcoded — dateparser resolves anchors and this module owns
    only week/month boundary policy.
    """
    normalized = " ".join(query.split()).lower()
    for pattern, unit, month_end in _TEMPORAL_PATTERNS:
        match = pattern.search(normalized)
        if match is None:
            continue
        anchor = match.group(1)
        bounds = resolve_range(anchor, unit, today, month_end=month_end)
        if bounds is not None:
            return bounds
    return None
