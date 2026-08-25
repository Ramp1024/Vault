"""Query-analysis quality metrics.

These metrics score the *query understanding* stage in isolation — before
retrieval runs — by comparing the ``SearchRequest`` an analyzer produces against
each case's ground-truth annotations (``expected_filters`` / ``expected_intent``).

They answer "did query analysis help or hurt?":

* **Filter Precision / Recall** — of the filters the analyzer generated, how many
  were correct, and of the filters it should have generated, how many did it find.
  Micro-averaged over every case so spurious filters on purely semantic queries
  are penalised as false positives.
* **Intent Classification Accuracy** — did the analyzer route the query to the
  right coarse intent (semantic / metadata / temporal)?
* **Filter Generation Accuracy** — for filter-expecting cases, did the analyzer
  produce *exactly* the right filter set?
* **Date Resolution Accuracy** — for temporal cases, did the analyzer resolve the
  correct date field, operator, and (day-normalised) value(s)?
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.evaluation.dataset import DATE_FIELDS, EvaluationCase, _derive_intent
from app.models.filter import Filter
from app.models.search_request import SearchRequest


@dataclass(frozen=True)
class AnalyzedCase:
    """A case paired with the ``SearchRequest`` an analyzer produced for it."""

    case: EvaluationCase
    request: SearchRequest

    @property
    def expected_filters(self) -> tuple[Filter, ...]:
        return self.case.expected_filters

    @property
    def produced_filters(self) -> tuple[Filter, ...]:
        return tuple(self.request.filters)


def _normalize_value(field: str, value: object) -> object:
    """Normalise a filter value for comparison.

    Date fields are compared by calendar day (first 10 chars of each ISO value),
    so an analyzer emitting a full ``YYYY-MM-DDTHH:MM:SS`` timestamp still matches
    a ground-truth ``YYYY-MM-DD``. Range values (lists/tuples) are normalised
    element-wise. Strings are compared case-insensitively and trimmed.
    """
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_value(field, item) for item in value)
    if field in DATE_FIELDS and isinstance(value, str):
        return value[:10]
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _filter_key(f: Filter) -> tuple:
    return (f.field, f.operator, _normalize_value(f.field, f.value))


def _filter_set(filters: Sequence[Filter]) -> set[tuple]:
    return {_filter_key(f) for f in filters}


def intent_of_request(request: SearchRequest) -> str:
    """The coarse routing intent implied by an analyzer's output filters."""
    return _derive_intent(tuple(request.filters))


@dataclass(frozen=True)
class FilterScore:
    """Micro-averaged filter precision/recall plus the raw confusion counts."""

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def filter_score(analyzed: Sequence[AnalyzedCase]) -> FilterScore:
    """Micro-averaged filter precision/recall across every analyzed case."""
    tp = fp = fn = 0
    for item in analyzed:
        expected = _filter_set(item.expected_filters)
        produced = _filter_set(item.produced_filters)
        tp += len(expected & produced)
        fp += len(produced - expected)
        fn += len(expected - produced)
    return FilterScore(true_positives=tp, false_positives=fp, false_negatives=fn)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def intent_classification_accuracy(analyzed: Sequence[AnalyzedCase]) -> float:
    """Fraction of cases whose analyzer intent matches the expected intent."""
    scores = [
        1.0 if intent_of_request(item.request) == item.case.expected_intent else 0.0
        for item in analyzed
        if item.case.expected_intent is not None
    ]
    return _mean(scores)


def filter_generation_accuracy(analyzed: Sequence[AnalyzedCase]) -> float:
    """Exact filter-set match rate over cases that expect at least one filter."""
    scores = [
        1.0
        if _filter_set(item.expected_filters) == _filter_set(item.produced_filters)
        else 0.0
        for item in analyzed
        if item.expected_filters
    ]
    return _mean(scores)


def date_resolution_accuracy(analyzed: Sequence[AnalyzedCase]) -> float:
    """Correct-date-filter rate over temporal cases.

    A temporal case passes when the set of date-field filters the analyzer
    produced exactly matches the expected date-field filters (field, operator,
    day-normalised values). Non-date filters are ignored here so this isolates
    date resolution from metadata filtering.
    """

    def date_filters(filters: Sequence[Filter]) -> set[tuple]:
        return {_filter_key(f) for f in filters if f.field in DATE_FIELDS}

    scores = [
        1.0
        if date_filters(item.expected_filters) == date_filters(item.produced_filters)
        else 0.0
        for item in analyzed
        if item.case.expected_intent == "temporal"
    ]
    return _mean(scores)


@dataclass(frozen=True)
class AnalysisReport:
    """The full query-analysis metric suite for one analyzer over a dataset."""

    count: int
    filter_precision: float
    filter_recall: float
    filter_f1: float
    intent_accuracy: float
    filter_generation_accuracy: float
    date_resolution_accuracy: float

    def as_dict(self) -> dict[str, float]:
        return {
            "Filter Precision": self.filter_precision,
            "Filter Recall": self.filter_recall,
            "Filter F1": self.filter_f1,
            "Intent Classification Accuracy": self.intent_accuracy,
            "Filter Generation Accuracy": self.filter_generation_accuracy,
            "Date Resolution Accuracy": self.date_resolution_accuracy,
        }


def build_analysis_report(analyzed: Sequence[AnalyzedCase]) -> AnalysisReport:
    """Compute every query-analysis metric for a batch of analyzed cases."""
    score = filter_score(analyzed)
    return AnalysisReport(
        count=len(analyzed),
        filter_precision=score.precision,
        filter_recall=score.recall,
        filter_f1=score.f1,
        intent_accuracy=intent_classification_accuracy(analyzed),
        filter_generation_accuracy=filter_generation_accuracy(analyzed),
        date_resolution_accuracy=date_resolution_accuracy(analyzed),
    )


def analyze_dataset(analyzer, cases: Sequence[EvaluationCase]) -> list[AnalyzedCase]:
    """Run an analyzer over each case's query, capturing its ``SearchRequest``.

    Analyzer failures are swallowed into an empty request (no filters), so a
    flaky LLM analyzer degrades to a measurable "produced nothing" outcome
    rather than aborting the whole benchmark.
    """
    analyzed: list[AnalyzedCase] = []
    for case in cases:
        try:
            request = analyzer.analyze(case.query)
        except Exception:  # pragma: no cover - defensive around the LLM stage
            request = SearchRequest(semantic_query=case.query, filters=[])
        analyzed.append(AnalyzedCase(case=case, request=request))
    return analyzed
