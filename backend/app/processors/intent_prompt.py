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
    "4. Resolve every date to an absolute ISO 'YYYY-MM-DD' value using the "
    "current date provided below as the reference point. Infer the year when it "
    "is omitted (e.g. 'Jul 20' -> the closest 20 July on or before the current "
    "date) and resolve relative expressions such as 'last week' or 'yesterday'.\n"
    "5. For dates, choose the operator by scope: a SINGLE specific day uses '=' "
    "with that day's date; only use 'between' for an explicit span of days (a "
    "week, a month, a date range), providing value as a two-element array "
    "[low, high]. Use '<' / '>' / '<=' / '>=' for open-ended 'before'/'after'.\n"
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
    "  ]\n"
    "}"
)


def build_intent_prompt(
    query: str, schema: MetadataSchema, *, today: date | None = None
) -> Prompt:
    """Build the intent-analysis prompt from the user query and the schema.

    The prompt is connector-agnostic: everything source-specific is expressed
    through the supplied :class:`MetadataSchema`, so new connectors change the
    available fields without any change to this template.

    ``today`` anchors relative and year-less dates (e.g. "Jul 20", "last week")
    to an absolute reference, defaulting to the current date.
    """
    reference_date = (today or date.today()).isoformat()
    schema_json = json.dumps(schema.to_dict(), indent=2, ensure_ascii=False)
    user = (
        f"Current date (reference for resolving dates): {reference_date}\n\n"
        "Available metadata schema (the only fields and operators you may use):\n"
        f"{schema_json}\n\n"
        f"{_OUTPUT_CONTRACT}\n\n"
        f'User question: "{query.strip()}"\n\n'
        "Return only the JSON object."
    )
    return Prompt(system=_SYSTEM, user=user)
