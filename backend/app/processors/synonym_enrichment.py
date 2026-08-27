"""Offline, LLM-assisted synonym enrichment for enumerable metadata values.

This is where the LLM earns its keep in the current architecture: **offline**, on
a closed input, to make the *deterministic* analyzer better — never at query time
where free-form generation over-filters.

For each enumerable field it asks the LLM for a few natural-language synonyms of
each known value (e.g. ``Done`` -> "completed", "finished"). The synonyms are
validated (deduped, stripped, and rejected if they collide with another value of
the same field) and stored on the schema as ``value_aliases``. At query time the
deterministic ``ConstraintExtractor`` matches these aliases to the canonical
value — with a required field cue — so a paraphrased query ("what tasks have I
completed") resolves to ``status=Done`` without any live LLM call.

The LLM only proposes synonyms for values we already know; it never invents
fields, values, or filters, so this cannot reintroduce the over-filtering that
free-form filter generation caused.

Run:  python -m app.processors.synonym_enrichment
"""

from __future__ import annotations

import json
import logging
import re

from app.models.metadata_schema import FieldType, MetadataField, MetadataSchema
from app.models.prompt import Prompt
from app.services.llm import LLM

logger = logging.getLogger(__name__)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_MAX_ALIASES = 4


def enrich_schema_with_aliases(
    schema: MetadataSchema, llm: LLM, *, max_aliases: int = _MAX_ALIASES
) -> MetadataSchema:
    """Return a copy of ``schema`` with LLM-generated ``value_aliases`` filled in.

    Only enumerable string fields are enriched; every other field is returned
    unchanged. Any LLM/parse failure for a field leaves that field without
    aliases rather than aborting the whole schema.
    """
    fields: list[MetadataField] = []
    for field in schema:
        if not (field.type is FieldType.STRING and field.is_enumerable):
            fields.append(field)
            continue
        aliases = _aliases_for_field(field, llm, max_aliases)
        fields.append(
            MetadataField(
                name=field.name,
                type=field.type,
                operators=field.operators,
                multi=field.multi,
                description=field.description,
                allowed_values=field.allowed_values,
                temporal_role=field.temporal_role,
                value_aliases=aliases,
            )
        )
    return MetadataSchema.from_fields(fields)


def _aliases_for_field(
    field: MetadataField, llm: LLM, max_aliases: int
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    try:
        raw = llm.generate(_build_prompt(field, max_aliases))
    except Exception:  # pragma: no cover - backend dependent
        logger.warning("Alias generation failed for %s", field.name, exc_info=True)
        return ()
    parsed = _parse(raw)
    if not parsed:
        return ()
    return _sanitize(field, parsed, max_aliases)


def _sanitize(
    field: MetadataField, proposed: dict, max_aliases: int
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Clean proposed synonyms: dedupe, drop collisions with other values."""
    value_words = {v.strip().casefold() for v in field.allowed_values}
    pairs: list[tuple[str, tuple[str, ...]]] = []
    for value in field.allowed_values:
        raw_aliases = proposed.get(value) or proposed.get(value.casefold()) or []
        if not isinstance(raw_aliases, list):
            continue
        seen: set[str] = set()
        kept: list[str] = []
        for alias in raw_aliases:
            if not isinstance(alias, str):
                continue
            text = " ".join(alias.split()).strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            # Reject an alias that is (or names) another value of this field.
            if key in value_words and key != value.casefold():
                continue
            if key == value.casefold():
                continue
            seen.add(key)
            kept.append(text)
            if len(kept) >= max_aliases:
                break
        if kept:
            pairs.append((value, tuple(kept)))
    return tuple(pairs)


def _parse(raw: str) -> dict | None:
    if not raw:
        return None
    match = _JSON_OBJECT.search(raw)
    if match is None:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _build_prompt(field: MetadataField, max_aliases: int) -> Prompt:
    values = ", ".join(field.allowed_values)
    desc = f" ({field.description})" if field.description else ""
    user = (
        f"The metadata field '{field.name}'{desc} takes these exact values:\n"
        f"{values}\n\n"
        f"For each value, list up to {max_aliases} natural-language words or short "
        "phrases a person might type meaning that value (synonyms only — do not "
        "invent new values or repeat the value itself).\n\n"
        'Respond as JSON mapping each exact value to a list of synonyms, e.g. '
        '{"Done": ["completed", "finished"]}.'
    )
    system = (
        "You generate concise natural-language synonyms for known metadata values. "
        "Output only JSON."
    )
    return Prompt(system=system, user=user)


def main() -> None:
    from app.services.llm import build_intent_llm
    from app.services.metadata_schema_store import MetadataSchemaStore

    store = MetadataSchemaStore()
    schema = store.load()
    if not schema:
        print("No persisted schema to enrich.")
        return
    enriched = enrich_schema_with_aliases(schema, build_intent_llm())
    store.save(enriched)
    print(f"Wrote aliases to {store.path}")
    for field in enriched:
        if field.value_aliases:
            for value, aliases in field.value_aliases:
                print(f"  {field.name}: {value} <- {list(aliases)}")


if __name__ == "__main__":
    main()
