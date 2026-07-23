"""Verification for metadata-aware retrieval.

Covers the query-understanding + retrieval pipeline:

    QueryAnalyzer -> SearchRequest -> Retriever -> QdrantFilterBuilder -> Qdrant

Required cases:
    1. Pure semantic search (no filters).
    2. Pure metadata search (filters only).
    3. Combined semantic + metadata search.
    4. Multiple metadata filters.
    5. Unknown metadata fields.
    6. Empty semantic query with filters only.
    7. Empty filters with semantic search only.

Runs fully offline using mocks (no live Qdrant/Ollama required).
"""

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from qdrant_client.http.models import (
    FieldCondition,
    Filter as QdrantFilter,
    MatchAny,
    MatchValue,
)

from app.connectors.notion.client import NotionDataSource
from app.connectors.notion.parser import NotionParser
from app.core.text import to_camel_case
from app.models.filter import Filter, Operator
from app.models.search_request import SearchRequest
from app.processors.metadata_registry import MetadataRegistry
from app.processors.query_analyzer import RuleBasedQueryAnalyzer
from app.services.qdrant_filter_builder import QdrantFilterBuilder
from app.services.qdrant_service import QdrantService
from app.services.retriever import Retriever


def _fake_point(chunk_id: str, score: float = 0.9, **extra):
    payload = {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "document_title": "Doc 1",
        "chunk_index": 0,
        "content": "some content",
        **extra,
    }
    return SimpleNamespace(payload=payload, score=score)


# ---------------------------------------------------------------------------
# camelCase util + registry
# ---------------------------------------------------------------------------


def verify_camel_case() -> None:
    assert to_camel_case("Leetcode Topic") == "leetcodeTopic"
    assert to_camel_case("Personal Win") == "personalWin"
    assert to_camel_case("Category ") == "category"
    assert to_camel_case("Date") == "date"
    print("  [C] to_camel_case normalizes field names")


def verify_registry() -> None:
    registry = MetadataRegistry()
    registry.register("Leetcode Topic")
    registry.register("Tags", multi=True, extra_aliases=("tag",))

    assert registry.resolve("leetcode topic") == "leetcodeTopic"
    assert registry.resolve("leetcodeTopic") == "leetcodeTopic"
    assert registry.resolve("tag") == "tags"
    assert registry.resolve("tags") == "tags"
    assert registry.is_multi("tags") is True
    assert registry.is_multi("leetcodeTopic") is False
    assert registry.resolve("unknown") is None
    print("  [R] registry resolves aliases and multi flags")


def verify_registry_from_indexed_fields() -> None:
    # Canonical camelCase payload keys, de-camelCased into surfaces internally.
    registry = MetadataRegistry.from_indexed_fields(
        ["aiProgress", "leetcodeTopic", "date", "notes"], multi_fields=[]
    )
    # Canonical names stay aligned with the indexed payload keys.
    assert registry.resolve("ai progress") == "aiProgress"
    assert registry.resolve("aiProgress") == "aiProgress"
    assert registry.resolve("leetcode topic") == "leetcodeTopic"
    assert registry.resolve("notes") == "notes"
    # A field name containing "date" is auto-typed as a date field.
    assert registry.kind_of("date") == "date"
    assert registry.kind_of("notes") == "text"

    analyzer = RuleBasedQueryAnalyzer(registry=registry)
    request = analyzer.analyze("what about ai progress: linear algebra")
    assert request.filters == [
        Filter(field="aiProgress", operator=Operator.EQUALS, value="linear algebra")
    ]
    print("  [D] registry derived from indexed field names recognizes new fields")


# ---------------------------------------------------------------------------
# QueryAnalyzer
# ---------------------------------------------------------------------------


def verify_analyzer_pure_semantic() -> None:
    request = RuleBasedQueryAnalyzer().analyze(
        "what did I learn about docker networking"
    )
    assert request.semantic_query == "what did I learn about docker networking"
    assert request.filters == []
    assert request.top_k == 5
    print("  [1] pure semantic -> no filters")


def verify_analyzer_pure_metadata() -> None:
    request = RuleBasedQueryAnalyzer().analyze("status: In Progress")
    assert request.semantic_query == ""
    assert request.filters == [
        Filter(field="status", operator=Operator.EQUALS, value="In Progress")
    ]
    print("  [2] pure metadata -> filters only, empty semantic (multi-word value)")


def verify_analyzer_combined() -> None:
    request = RuleBasedQueryAnalyzer().analyze(
        "deployment notes status: In Progress"
    )
    assert request.semantic_query == "deployment notes"
    assert request.filters == [
        Filter(field="status", operator=Operator.EQUALS, value="In Progress")
    ]
    print("  [3] combined -> semantic + filter split")


def verify_analyzer_multiple_filters() -> None:
    request = RuleBasedQueryAnalyzer().analyze(
        "retro leetcode topic: Graphs project: Vault team: AI tag: RAG"
    )
    assert request.semantic_query == "retro"
    assert request.filters == [
        Filter(field="leetcodeTopic", operator=Operator.EQUALS, value="Graphs"),
        Filter(field="project", operator=Operator.EQUALS, value="Vault"),
        Filter(field="team", operator=Operator.EQUALS, value="AI"),
        Filter(field="tags", operator=Operator.CONTAINS, value="RAG"),
    ]
    print("  [4] multiple filters (incl. CONTAINS for tags)")


def verify_analyzer_unknown_field() -> None:
    request = RuleBasedQueryAnalyzer().analyze("priority: high urgent tasks")
    # "priority" is not registered -> not extracted; whole text stays semantic.
    assert request.semantic_query == "priority: high urgent tasks"
    assert request.filters == []
    print("  [5] unknown field -> treated as semantic text")


def verify_analyzer_filters_only() -> None:
    request = RuleBasedQueryAnalyzer().analyze("date: 2026-07-20")
    assert request.semantic_query == ""
    assert request.filters == [
        Filter(field="date", operator=Operator.EQUALS, value="2026-07-20")
    ]
    print("  [6] empty semantic + filters only")


def verify_analyzer_semantic_only() -> None:
    request = RuleBasedQueryAnalyzer().analyze("docker compose networking")
    assert request.semantic_query == "docker compose networking"
    assert request.filters == []
    print("  [7] semantic only + empty filters")


def verify_analyzer_date_typed_extraction() -> None:
    analyzer = RuleBasedQueryAnalyzer()

    # DD-MM-YYYY is normalized to ISO.
    request = analyzer.analyze("what did I do on date: 21-07-2026")
    assert request.filters == [
        Filter(field="date", operator=Operator.EQUALS, value="2026-07-21")
    ]

    # Only the date token is captured; trailing prose stays semantic.
    request = analyzer.analyze(
        "what have i written about date: 2026-07-21 on the leetcode topic"
    )
    assert request.filters == [
        Filter(field="date", operator=Operator.EQUALS, value="2026-07-21")
    ]
    assert "2026-07-21" not in request.semantic_query
    assert "leetcode topic" in request.semantic_query
    print("  [8] date field extracts/normalizes token, leaves prose semantic")


# ---------------------------------------------------------------------------
# NotionParser property extraction (camelCase)
# ---------------------------------------------------------------------------


def verify_parser_emits_camelcase_properties() -> None:
    parser = NotionParser()
    page = {
        "id": "page-1",
        "url": "https://notion.so/page-1",
        "last_edited_time": "2026-07-20T10:00:00Z",
        "properties": {
            "Name": {"type": "title", "title": [{"plain_text": "Monday"}]},
            "Status": {"type": "select", "select": {"name": "Completed"}},
            "Leetcode Topic": {"type": "select", "select": {"name": "Graphs"}},
            "Tags": {
                "type": "multi_select",
                "multi_select": [{"name": "Graphs"}, {"name": "DFS"}],
            },
            "Leetcode": {"type": "number", "number": 133},
            "Favorite": {"type": "checkbox", "checkbox": True},
            "Date": {"type": "date", "date": {"start": "2026-07-20"}},
        },
    }

    document = parser.parse_page(
        NotionDataSource(id="ds-1", name="DS 1"), page, blocks=[]
    )

    assert document.metadata["properties"] == {
        "status": "Completed",
        "leetcodeTopic": "Graphs",
        "tags": ["Graphs", "DFS"],
        "leetcode": 133,
        "favorite": True,
        "date": "2026-07-20",
    }
    assert document.title == "Monday"
    print("  [P] parser emits typed camelCase property fields")


# ---------------------------------------------------------------------------
# QdrantFilterBuilder
# ---------------------------------------------------------------------------


def verify_filter_builder() -> None:
    builder = QdrantFilterBuilder()

    assert builder.build([]) is None

    built = builder.build(
        [
            Filter(field="category", operator=Operator.EQUALS, value="journal"),
            Filter(field="tags", operator=Operator.CONTAINS, value="RAG"),
            Filter(field="leetcodeTopic", operator=Operator.EQUALS, value=["a", "b"]),
        ]
    )
    expected = QdrantFilter(
        must=[
            FieldCondition(key="properties.category", match=MatchValue(value="journal")),
            FieldCondition(key="properties.tags", match=MatchValue(value="RAG")),
            FieldCondition(
                key="properties.leetcodeTopic", match=MatchAny(any=["a", "b"])
            ),
        ]
    )
    assert built == expected
    print("  [8] builder maps filters to properties.* payload conditions")


# ---------------------------------------------------------------------------
# QdrantService (prebuilt filter)
# ---------------------------------------------------------------------------


def verify_search_with_vector_and_filter() -> None:
    client = Mock()
    client.query_points.return_value = SimpleNamespace(
        points=[_fake_point("chunk-1", score=0.75)]
    )
    service = QdrantService(client=client)

    qfilter = QdrantFilter(
        must=[FieldCondition(key="properties.category", match=MatchValue(value="journal"))]
    )
    results = service.search(query_embedding=[0.1, 0.2], limit=5, query_filter=qfilter)

    assert len(results) == 1 and results[0].chunk.id == "chunk-1"
    _, kwargs = client.query_points.call_args
    assert kwargs["query"] == [0.1, 0.2]
    assert kwargs["query_filter"] == qfilter
    client.scroll.assert_not_called()
    print("  [9] vector search forwards prebuilt filter to query_points")


def verify_filter_only_search() -> None:
    client = Mock()
    client.scroll.return_value = ([_fake_point("chunk-2")], None)
    service = QdrantService(client=client)

    qfilter = QdrantFilter(
        must=[FieldCondition(key="properties.date", match=MatchValue(value="2026-07-20"))]
    )
    results = service.search(query_embedding=[], limit=5, query_filter=qfilter)

    assert len(results) == 1 and results[0].chunk.id == "chunk-2"
    assert results[0].score == 0.0
    client.query_points.assert_not_called()
    _, kwargs = client.scroll.call_args
    assert kwargs["scroll_filter"] == qfilter
    print("  [10] empty embedding + filter -> filter-only scroll")


def verify_search_no_embedding_no_filter() -> None:
    client = Mock()
    service = QdrantService(client=client)

    assert service.search(query_embedding=[], limit=5, query_filter=None) == []
    client.query_points.assert_not_called()
    client.scroll.assert_not_called()
    print("  [11] empty embedding + no filter -> empty result")


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


def verify_retriever_combined() -> None:
    embedding_service = Mock()
    embedding_service.embed.return_value = [0.4, 0.5, 0.6]
    qdrant_service = Mock()
    qdrant_service.search.return_value = ["result"]
    filter_builder = QdrantFilterBuilder()

    retriever = Retriever(
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        filter_builder=filter_builder,
    )

    request = SearchRequest(
        semantic_query="deployment notes",
        filters=[Filter(field="category", operator=Operator.EQUALS, value="journal")],
        top_k=3,
    )
    assert retriever.search(request) == ["result"]

    embedding_service.embed.assert_called_once_with("deployment notes")
    _, kwargs = qdrant_service.search.call_args
    assert kwargs["query_embedding"] == [0.4, 0.5, 0.6]
    assert kwargs["limit"] == 3
    assert kwargs["query_filter"] == QdrantFilter(
        must=[FieldCondition(key="properties.category", match=MatchValue(value="journal"))]
    )
    print("  [12] retriever embeds semantic + forwards built filter + top_k")


def verify_retriever_metadata_only() -> None:
    embedding_service = Mock()
    qdrant_service = Mock()
    qdrant_service.search.return_value = []

    retriever = Retriever(
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        filter_builder=QdrantFilterBuilder(),
    )

    request = SearchRequest(
        semantic_query="",
        filters=[Filter(field="date", operator=Operator.EQUALS, value="2026-07-20")],
        top_k=5,
    )
    retriever.search(request)

    embedding_service.embed.assert_not_called()
    _, kwargs = qdrant_service.search.call_args
    assert kwargs["query_embedding"] == []
    assert kwargs["query_filter"] == QdrantFilter(
        must=[FieldCondition(key="properties.date", match=MatchValue(value="2026-07-20"))]
    )
    print("  [13] metadata-only request skips embedding")


def verify_metadata_retrieval() -> None:
    print("=" * 80)
    print("VAULT - METADATA-AWARE RETRIEVAL VERIFICATION")
    print("=" * 80)

    print("\nUtil + Registry:")
    verify_camel_case()
    verify_registry()
    verify_registry_from_indexed_fields()

    print("\nQueryAnalyzer:")
    verify_analyzer_pure_semantic()
    verify_analyzer_pure_metadata()
    verify_analyzer_combined()
    verify_analyzer_multiple_filters()
    verify_analyzer_unknown_field()
    verify_analyzer_filters_only()
    verify_analyzer_semantic_only()
    verify_analyzer_date_typed_extraction()

    print("\nNotionParser:")
    verify_parser_emits_camelcase_properties()

    print("\nQdrantFilterBuilder:")
    verify_filter_builder()

    print("\nQdrantService:")
    verify_search_with_vector_and_filter()
    verify_filter_only_search()
    verify_search_no_embedding_no_filter()

    print("\nRetriever:")
    verify_retriever_combined()
    verify_retriever_metadata_only()

    print("\n\u2705 Metadata-aware retrieval verified")


if __name__ == "__main__":
    verify_metadata_retrieval()
