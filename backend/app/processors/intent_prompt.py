from __future__ import annotations

import json
from datetime import date

from app.models.metadata_schema import MetadataSchema
from app.models.prompt import Prompt

_SYSTEM = (
    "You are a query compiler for a personal knowledge base. Your only job is to "
    "translate a user's natural-language question into a structured search "
    "request. You do NOT answer the question, retrieve documents, or explain "
    "anything.\n"
    "\n"
    "You are given a metadata schema describing the ONLY filterable fields that "
    "exist. Follow these rules strictly:\n"
    "1. Rewrite the user's question into a concise semantic_query capturing what "
    "they are looking for. Remove any part that is expressed as a metadata "
    "filter so it is not duplicated.\n"
    "2. Infer metadata filters ONLY from fields present in the schema. Never "
    "invent field names, and never use an operator a field does not list.\n"
    "3. Every value must match the field's type: booleans are true/false, dates "
    "are ISO 'YYYY-MM-DD' strings, numbers are numeric, strings are text.\n"
    "4. AUTHORING ACTIVITY OVER TIME: When the user asks about WHEN THEY wrote, "
    "edited, created, noted, journaled, logged, added, or did something (e.g. "
    "'what did I write yesterday', 'notes from last week', 'what did I do day "
    "before yesterday'), DO NOT resolve the date and DO NOT emit a date filter. "
    "Instead populate the `temporal` object. Copy the date expression VERBATIM "
    "from the user's question into `anchor` (e.g. 'yesterday', 'day before "
    "yesterday', 'last week', 'aug 11'); never compute, shift, or reword it. Set "
    "`kind` to 'range' when they mean a whole week or month ('this week', 'last "
    "month') and put the matching `unit` ('week' or 'month'); otherwise set "
    "`kind` to 'single'. When you populate `temporal`, do not also add a date "
    "filter for the same time expression.\n"
    "5. If the question is NOT about authoring activity over time, set "
    "`temporal` to null and rely on filters and semantic_query as usual.\n"
    "6. If no filter clearly applies, return an empty filters array. When in "
    "doubt, prefer fewer filters and rely on the semantic_query.\n"
    "7. Respond with a SINGLE valid JSON object and nothing else."
)

_OUTPUT_CONTRACT = (
    "Output JSON shape:\n"
    "{\n"
    '  "semantic_query": "<rewritten query text>",\n'
    '  "filters": [\n'
    '    {"field": "<field name>", "operator": "<operator>", "value": <value>}\n'
    "  ],\n"
    '  "temporal": {"kind": "single|range", "unit": "week|month", '
    '"anchor": "<verbatim date phrase>"}  // or null when not applicable\n'
    "}"
)


def build_intent_prompt(
    query: str, schema: MetadataSchema, *, today: date | None = None
) -> Prompt:
    """Build the intent-analysis prompt from the user query and the schema.

    The prompt is connector-agnostic: everything source-specific is expressed
    through the supplied :class:`MetadataSchema`, so new connectors change the
    available fields without any change to this template.

    ``today`` is passed only as context; the model never resolves dates itself —
    it copies authoring-time expressions verbatim into ``temporal.anchor`` for
    downstream resolution.
    """
    reference_date = (today or date.today()).isoformat()
    schema_json = json.dumps(schema.to_dict(), indent=2, ensure_ascii=False)
    user = (
        f"Current date (context only; do not resolve dates yourself): "
        f"{reference_date}\n\n"
        "Available metadata schema (the only fields and operators you may use):\n"
        f"{schema_json}\n\n"
        f"{_OUTPUT_CONTRACT}\n\n"
        f'User question: "{query.strip()}"\n\n'
        "Return only the JSON object."
    )
    return Prompt(system=_SYSTEM, user=user)
