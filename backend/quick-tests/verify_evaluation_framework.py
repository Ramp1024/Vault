"""Offline verification for the retrieval evaluation framework.

Exercises the dataset model, metrics, runner, and report using a fake search
engine (no live Qdrant/Ollama). Proves Recall@k and MRR are computed correctly
and that the framework depends only on the public search(query) contract.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    MeanReciprocalRank,
    RecallAtK,
    RetrievalEvaluator,
    format_report,
)
from app.models.chunk import Chunk
from app.models.search_result import SearchResult


def _result(document_id: str, index: int, score: float) -> SearchResult:
    chunk = Chunk(
        id=f"{document_id}_{index}",
        document_id=document_id,
        document_title=document_id,
        content="c",
        chunk_index=index,
        metadata={},
    )
    return SearchResult(chunk=chunk, score=score)


class FakeSearchEngine:
    """Returns canned results per query; mimics the public SearchEngine API."""

    def __init__(self, responses: dict[str, list[SearchResult]]) -> None:
        self._responses = responses

    def search(self, query: str) -> list[SearchResult]:
        return self._responses.get(query, [])


def verify_dataset_roundtrip_and_forward_compat() -> None:
    raw = [
        {
            "id": "c1",
            "query": "q1",
            "category": "semantic",
            "expected_documents": ["doc-a"],
            "expected_chunks": ["doc-a_0"],
            # Future/annotation fields must be tolerated and ignored:
            "expected_answer": "unused",
            "required_citations": ["doc-a_0"],
            "evaluation_notes": "ignore me",
            "expected_document_titles": ["A"],
        }
    ]
    dataset = EvaluationDataset.from_list(raw)
    assert len(dataset) == 1
    case = dataset.cases[0]
    assert case.expected_documents == ("doc-a",)
    assert case.category == "semantic"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ds.json"
        dataset.to_file(path)
        reloaded = EvaluationDataset.from_file(path)
        assert reloaded.cases[0].id == "c1"
    print("  [1] dataset load/save round-trip; unknown fields ignored")


def verify_metrics_are_correct() -> None:
    cases = [
        EvaluationCase(id="q_top1", query="q1", expected_documents=("A",)),
        EvaluationCase(id="q_rank3", query="q2", expected_documents=("B",)),
        EvaluationCase(id="q_rank8", query="q3", expected_documents=("C",)),
        EvaluationCase(id="q_miss", query="q4", expected_documents=("D",)),
    ]
    dataset = EvaluationDataset(cases=tuple(cases))

    responses = {
        # relevant at rank 1
        "q1": [_result("A", 0, 0.9), _result("X", 0, 0.8)],
        # relevant at rank 3
        "q2": [_result("Y", 0, 0.9), _result("Z", 0, 0.8), _result("B", 0, 0.7)],
        # relevant at rank 8 (inside top-10, outside top-5)
        "q3": [_result(f"N{i}", 0, 1.0 - i * 0.05) for i in range(7)]
        + [_result("C", 0, 0.2)],
        # never retrieved
        "q4": [_result("Q", 0, 0.5)],
    }

    evaluator = RetrievalEvaluator(
        search_engine=FakeSearchEngine(responses),
        metrics=[RecallAtK(5), RecallAtK(10), MeanReciprocalRank()],
    )
    report = evaluator.evaluate(dataset)

    # Recall@5: q1 hit, q2 hit, q3 miss (rank 8), q4 miss -> 2/4 = 0.5
    assert abs(report.metrics["Recall@5"] - 0.5) < 1e-9, report.metrics
    # Recall@10: q1,q2,q3 hit, q4 miss -> 3/4 = 0.75
    assert abs(report.metrics["Recall@10"] - 0.75) < 1e-9, report.metrics
    # MRR: 1/1 + 1/3 + 1/8 + 0 = 1.458333 / 4 = 0.364583...
    expected_mrr = (1.0 + 1.0 / 3 + 1.0 / 8 + 0.0) / 4
    assert abs(report.metrics["MRR"] - expected_mrr) < 1e-9, report.metrics

    ranks = {e.case.id: e.first_relevant_rank for e in report.evaluations}
    assert ranks == {"q_top1": 1, "q_rank3": 3, "q_rank8": 8, "q_miss": None}
    print("  [2] Recall@5, Recall@10, MRR, and first-relevant-rank are correct")


def verify_report_highlights() -> None:
    cases = [
        EvaluationCase(id="pass1", query="p", expected_documents=("A",)),
        EvaluationCase(id="outside", query="o", expected_documents=("B",)),
        EvaluationCase(id="missing", query="m", expected_documents=("C",)),
    ]
    responses = {
        "p": [_result("A", 0, 0.9)],
        "o": [_result(f"N{i}", 0, 0.5) for i in range(6)] + [_result("B", 0, 0.1)],
        "m": [_result("Z", 0, 0.5)],
    }
    evaluator = RetrievalEvaluator(search_engine=FakeSearchEngine(responses))
    report = evaluator.evaluate(EvaluationDataset(cases=tuple(cases)))
    text = format_report(report, primary_k=5)

    assert "Overall metrics:" in text
    assert "Recall@5" in text and "MRR" in text
    assert "Failed queries" in text
    assert "outside" in text  # relevant but at rank 7
    assert "missing" in text  # no relevant retrieval
    assert "[PASS] pass1" in text
    print("  [3] report renders metrics, per-query rows, and highlights")


def verify_evaluation_framework() -> None:
    print("=" * 80)
    print("VAULT - RETRIEVAL EVALUATION FRAMEWORK VERIFICATION")
    print("=" * 80)
    verify_dataset_roundtrip_and_forward_compat()
    verify_metrics_are_correct()
    verify_report_highlights()
    print("\n\u2705 Retrieval evaluation framework verified")


if __name__ == "__main__":
    verify_evaluation_framework()
