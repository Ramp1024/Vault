from __future__ import annotations

import json
import logging
import re

from app.models.metadata_schema import MetadataSchema
from app.models.search_request import SearchRequest
from app.models.filter import Filter, Operator
from app.core.clock import today as current_date
from app.core.config import settings
from app.processors.filter_validator import FilterValidator
from app.processors.intent_prompt import build_intent_prompt
from app.processors.query_analyzer import QueryAnalyzer
from app.processors.temporal_query import resolve_temporal_descriptor
from app.services.llm import LLM

logger = logging.getLogger(__name__)

# Extract the first JSON object from a model response, tolerating markdown code
# fences or surrounding prose the model may add despite instructions.
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class LLMIntentAnalyzer(QueryAnalyzer):
    """Schema-aware query understanding powered by an LLM.

    Acts purely as a *query compiler*: given the user query and a
    :class:`MetadataSchema`, it asks the LLM for a rewritten semantic query and
    candidate metadata filters, then validates those filters against the schema
    before producing a :class:`SearchRequest`. It never retrieves documents,
    builds backend filters, or reasons about retrieval internals.

    Failure is always safe: any LLM/JSON error degrades to a plain semantic
    request over the original query, so retrieval keeps working even when the
    model is unavailable or returns malformed output.
    """

    def __init__(
        self,
        llm: LLM,
        schema: MetadataSchema,
        *,
        default_top_k: int = 5,
        validator: FilterValidator | None = None,
    ) -> None:
        self.llm = llm
        self.schema = schema
        self.default_top_k = default_top_k
        self.validator = validator or FilterValidator(schema)

    def analyze(self, query: str) -> SearchRequest:
        normalized = " ".join(query.split()).strip()

        # With no schema there is nothing to infer; fall back to a semantic-only
        # request without spending an LLM call.
        if not self.schema:
            return self._fallback(normalized)

        # Anchor every relative/year-less date to one timezone-correct reference
        # shared by the prompt and the validator, so they never disagree by a day.
        reference = current_date()
        try:
            raw = self.llm.generate(
                build_intent_prompt(normalized, self.schema, today=reference)
            )
        except Exception:  # pragma: no cover - defensive, backend-dependent
            logger.warning("LLM intent analysis failed; using semantic fallback", exc_info=True)
            return self._fallback(normalized)

        parsed = self._parse(raw)
        if parsed is None:
            logger.warning("LLM intent analysis returned unparseable output")
            return self._fallback(normalized)

        semantic_query = self._semantic_query(parsed, normalized)
        filters = self.validator.validate(
            parsed.get("filters"), query=normalized, today=reference
        )
        filters = self._apply_temporal(
            parsed.get("temporal"), filters, reference, normalized
        )
        return SearchRequest(
            semantic_query=semantic_query,
            filters=filters,
            top_k=self.default_top_k,
        )

    def _fallback(self, query: str) -> SearchRequest:
        return SearchRequest(
            semantic_query=query,
            filters=[],
            top_k=self.default_top_k,
        )

    @staticmethod
    def _apply_temporal(descriptor, filters, reference, query):
        """Fold an LLM temporal descriptor into an authoring-time range filter.

        Resolution (dateparser + boundary snapping) lives in ``temporal_query``.
        When it yields bounds, the authoring-time axis is authoritative, so any
        competing content-date filter is dropped to avoid an empty AND. A missing
        or unresolvable descriptor leaves ``filters`` untouched.
        """
        bounds = resolve_temporal_descriptor(descriptor, reference, query)
        if bounds is None:
            return filters
        low, high = bounds
        field = settings.AUTHORSHIP_DATE_FIELD
        kept = [f for f in filters if f.field not in {field, "date"}]
        kept.append(Filter(field=field, operator=Operator.BETWEEN, value=[low, high]))
        return kept

    @staticmethod
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

    @staticmethod
    def _semantic_query(parsed: dict, original: str) -> str:
        value = parsed.get("semantic_query")
        if isinstance(value, str) and value.strip():
            return " ".join(value.split()).strip()
        # A missing/blank rewrite should never blank out retrieval.
        return original
