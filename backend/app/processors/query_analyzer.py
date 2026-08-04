from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.models.filter import Filter, Operator
from app.models.search_request import SearchRequest
from app.processors.metadata_registry import MetadataRegistry, default_metadata_registry


class QueryAnalyzer(ABC):
    """Convert a raw natural-language query into a structured SearchRequest.

    Implementations must not depend on any vector store or retrieval backend.
    Their only responsibility is query understanding: splitting a raw query into
    a semantic portion and structured metadata filters. This keeps query
    understanding cleanly separated from vector search, so a rule-based analyzer
    can later be swapped for an LLM-powered one without touching the retriever.
    """

    @abstractmethod
    def analyze(self, query: str) -> SearchRequest:
        """Transform a raw query string into a SearchRequest."""
        raise NotImplementedError


_SEPARATOR = r"\s*(?::|=|\bis\b)\s*"
_VALUE_STRIP_CHARS = ".,;:!?"

# A date-like token: three numeric groups separated by - or /. Either the first
# group (YYYY-MM-DD) or the last group (DD-MM-YYYY) may be the 4-digit year.
_DATE_TOKEN = re.compile(r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b")
_ISO_DATE = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$")
_DMY_DATE = re.compile(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$")


def _normalize_date(token: str) -> str | None:
    """Normalize a date token to ISO ``YYYY-MM-DD``; return None if unrecognized."""
    iso = _ISO_DATE.match(token)
    if iso:
        year, month, day = iso.groups()
    else:
        dmy = _DMY_DATE.match(token)
        if not dmy:
            return None
        day, month, year = dmy.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _alias_regex(surface: str) -> str:
    """Build a whitespace-flexible regex fragment for a (possibly multi-word) surface."""
    return r"\s+".join(re.escape(word) for word in surface.split())


def _dequote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


class RuleBasedQueryAnalyzer(QueryAnalyzer):
    """Rule-based query understanding driven by a configurable metadata registry.

    Detects ``field: value`` / ``field=value`` / ``field is value`` expressions
    for any field known to the registry (values may be multi-word or quoted).
    Text not consumed by a recognized filter becomes the semantic query. No LLM
    is used.
    """

    def __init__(
        self,
        registry: MetadataRegistry | None = None,
        default_top_k: int = 5,
    ) -> None:
        self.registry = registry or default_metadata_registry()
        self.default_top_k = default_top_k
        self._head_re = self._build_head_regex()

    def _build_head_regex(self) -> re.Pattern[str] | None:
        surfaces = self.registry.surfaces()
        if not surfaces:
            return None
        alias_pattern = "|".join(_alias_regex(surface) for surface in surfaces)
        return re.compile(
            rf"\b(?P<field>{alias_pattern})\b{_SEPARATOR}",
            re.IGNORECASE,
        )

    def analyze(self, query: str) -> SearchRequest:
        if self._head_re is None:
            return SearchRequest(
                semantic_query=query.strip(),
                filters=[],
                top_k=self.default_top_k,
            )

        heads = list(self._head_re.finditer(query))
        if not heads:
            return SearchRequest(
                semantic_query=" ".join(query.split()).strip(),
                filters=[],
                top_k=self.default_top_k,
            )

        filters: list[Filter] = []
        semantic_parts: list[str] = []
        prev_end = 0

        for index, head in enumerate(heads):
            # Text between the previous value and this field head is semantic.
            semantic_parts.append(query[prev_end : head.start()])

            value_start = head.end()
            value_end = (
                heads[index + 1].start()
                if index + 1 < len(heads)
                else len(query)
            )
            span = query[value_start:value_end]

            canonical = self.registry.resolve(head.group("field"))
            value = ""

            if canonical and self.registry.kind_of(canonical) == "date":
                # Only capture a date-shaped token; leave any trailing prose
                # (e.g. "on the leetcode topic") as semantic text instead of
                # swallowing it into the value.
                match = _DATE_TOKEN.search(span)
                if match:
                    value = _normalize_date(match.group()) or ""
                    remainder = (span[: match.start()] + " " + span[match.end() :]).strip()
                    if remainder:
                        semantic_parts.append(f" {remainder} ")
            else:
                raw_value = span.strip()
                value = _dequote(raw_value).strip(_VALUE_STRIP_CHARS).strip()

            if canonical and value:
                operator = (
                    Operator.CONTAINS
                    if self.registry.is_multi(canonical)
                    else Operator.EQUALS
                )
                filters.append(
                    Filter(field=canonical, operator=operator, value=value)
                )

            prev_end = value_end

        semantic_query = " ".join(" ".join(semantic_parts).split()).strip()

        return SearchRequest(
            semantic_query=semantic_query,
            filters=filters,
            top_k=self.default_top_k,
        )


class CompositeQueryAnalyzer(QueryAnalyzer):
    """Merge deterministic rule-based parsing with LLM intent understanding.

    The rule-based analyzer owns explicit, unambiguous signal (``field: value``
    syntax, explicit dates, obvious operators); the LLM analyzer owns semantic
    understanding (conversational phrasing, implicit metadata inference, query
    rewriting). Their outputs are merged into the single ``SearchRequest`` the
    retrieval engine consumes — the engine stays completely unaware that two
    analyzers, an LLM, or a schema were ever involved.

    Rule-based results take precedence on conflict: a filter the rule parser set
    for a field is never overridden by the LLM, and when the user used explicit
    filter syntax the deterministic semantic query is kept. The LLM analyzer is
    also treated as best-effort — if it raises, the rule-based result is used
    unchanged, so composition can never make retrieval less reliable.
    """

    def __init__(
        self,
        rule_based: QueryAnalyzer,
        llm_based: QueryAnalyzer,
    ) -> None:
        self.rule_based = rule_based
        self.llm_based = llm_based

    def analyze(self, query: str) -> SearchRequest:
        rule_request = self.rule_based.analyze(query)

        try:
            llm_request = self.llm_based.analyze(query)
        except Exception:  # pragma: no cover - defensive around the LLM stage
            return rule_request

        filters = self._merge_filters(rule_request.filters, llm_request.filters)
        semantic_query = self._merge_semantic_query(rule_request, llm_request)
        return SearchRequest(
            semantic_query=semantic_query,
            filters=filters,
            top_k=rule_request.top_k,
        )

    @staticmethod
    def _merge_filters(
        rule_filters: list[Filter], llm_filters: list[Filter]
    ) -> list[Filter]:
        """Union both filter sets, keeping rule-based filters on field conflicts."""
        merged: list[Filter] = list(rule_filters)
        claimed = {f.field for f in rule_filters}
        for candidate in llm_filters:
            if candidate.field in claimed:
                continue
            claimed.add(candidate.field)
            merged.append(candidate)
        return merged

    @staticmethod
    def _merge_semantic_query(
        rule_request: SearchRequest, llm_request: SearchRequest
    ) -> str:
        """Choose the semantic query, preferring determinism for explicit syntax.

        When the rule parser recognized explicit filter syntax, its cleaned
        semantic text is authoritative. Otherwise the LLM's rewritten/enriched
        query is preferred, falling back to the rule-based text if the LLM
        produced none.
        """
        if rule_request.filters:
            return rule_request.semantic_query or llm_request.semantic_query
        return llm_request.semantic_query or rule_request.semantic_query
