"""Build the production retrieval-evaluation dataset from the indexed corpus.

This is a *curation aid*, not part of the evaluation framework runtime. It loads
the live corpus from Qdrant and assembles the large, realistic benchmark that
simulates how a real user interacts with their personal knowledge base over
time. Every expected-document set is grounded in the actual index:

* deterministic categories (metadata / lexical / temporal) derive their ground
  truth directly from corpus properties, body terms, or authoring windows, so
  they are always correct by construction;
* the harder categories (factual / semantic / synthesis / assistant / ambiguous)
  are hand-authored with explicit document ids that this builder validates
  against the index — a typo'd or stale id aborts the build.

Each emitted case carries a ``category``, a ``difficulty`` label, and (where
applicable) the ``expected_filters`` / ``expected_intent`` ground truth the
query-analysis metrics score against.

Run:  python -m app.evaluation.build_production_dataset
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.core.clock import to_local_day
from app.core.config import settings
from app.evaluation.dataset import EvaluationCase, EvaluationDataset
from app.models.filter import Filter, Operator
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService

OUTPUT_PATH = Path(__file__).parent / "data" / "production_dataset.json"

# Authoring-time field the temporal-authorship layer routes "write/work/do"
# queries to (see repo architecture notes); temporal cases assert filters on it.
AUTHORSHIP_FIELD = "last_edited_time"


@dataclass
class DocInfo:
    document_id: str
    title: str
    chunk_ids: list[str]
    properties: dict
    content: str
    edited: str  # YYYY-MM-DD


class Corpus:
    """Indexed lookups over the loaded corpus used to ground expectations."""

    def __init__(self, docs: dict[str, DocInfo]) -> None:
        self.docs = docs

    def require(self, doc_ids: list[str]) -> tuple[str, ...]:
        missing = [d for d in doc_ids if d not in self.docs]
        if missing:
            raise ValueError(f"Unknown document ids (not indexed): {missing}")
        return tuple(doc_ids)

    def chunks_for(self, doc_ids: tuple[str, ...]) -> tuple[str, ...]:
        chunk_ids: list[str] = []
        for doc_id in doc_ids:
            chunk_ids.extend(self.docs[doc_id].chunk_ids)
        return tuple(chunk_ids)

    def with_property(self, field_name: str, value: str) -> tuple[str, ...]:
        out = []
        for info in self.docs.values():
            raw = info.properties.get(field_name)
            values = raw if isinstance(raw, list) else [raw]
            if any(isinstance(v, str) and v == value for v in values):
                out.append(info.document_id)
        return tuple(sorted(out))

    def containing(self, term: str) -> tuple[str, ...]:
        term_l = term.lower()
        return tuple(
            sorted(
                info.document_id
                for info in self.docs.values()
                if term_l in info.content.lower()
            )
        )

    def titled(self, title: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                info.document_id
                for info in self.docs.values()
                if info.title.strip() == title
            )
        )

    def in_window(self, start: str, end: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                info.document_id
                for info in self.docs.values()
                if start <= info.edited <= end
            )
        )


def load_corpus(qdrant: QdrantService) -> Corpus:
    client = qdrant.client
    docs: dict[str, DocInfo] = {}
    chunks_by_doc: dict[str, list[tuple[int, str, str]]] = defaultdict(list)

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            doc_id = payload.get("document_id")
            if not doc_id:
                continue
            index = int(payload.get("chunk_index", 0))
            chunks_by_doc[doc_id].append(
                (index, str(payload.get("chunk_id")), str(payload.get("content", "")))
            )
            if doc_id not in docs:
                _led = to_local_day(str(payload.get("last_edited_time") or ""))
                docs[doc_id] = DocInfo(
                    document_id=doc_id,
                    title=str(payload.get("document_title", "")),
                    chunk_ids=[],
                    properties=payload.get("properties") or {},
                    content="",
                    edited=_led.isoformat() if _led else "",
                )
        if offset is None:
            break

    for doc_id, info in docs.items():
        ordered = sorted(chunks_by_doc[doc_id])
        info.chunk_ids = [cid for _, cid, _ in ordered]
        info.content = "\n".join(c for _, _, c in ordered)
    return Corpus(docs)


# --------------------------------------------------------------------------- #
# Case specifications                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class Spec:
    """A dataset case before its expected documents are resolved and validated."""

    id: str
    query: str
    category: str
    difficulty: str
    docs: tuple[str, ...] = ()
    filters: tuple[Filter, ...] = ()
    intent: str | None = None


def _eq(field_name: str, value: str) -> Filter:
    return Filter(field=field_name, operator=Operator.EQUALS, value=value)


def _between(field_name: str, start: str, end: str) -> Filter:
    return Filter(field=field_name, operator=Operator.BETWEEN, value=[start, end])


def metadata_specs(corpus: Corpus) -> list[Spec]:
    """Metadata-search cases: natural language and explicit-syntax filters."""
    specs: list[Spec] = []

    # (property, value, natural query, difficulty) — ground truth is the filter.
    natural = [
        ("category", "AI Engineering", "What AI engineering notes do I have?", "medium"),
        ("category", "System Design", "Show my system design notes.", "medium"),
        (
            "category",
            "Front end System Design",
            "What front-end system design notes do I have?",
            "medium",
        ),
        ("category", "Leetcode Notes", "Show my leetcode concept notes.", "medium"),
        (
            "leetcodeTopic",
            "Backtracking",
            "What backtracking leetcode problems have I solved?",
            "medium",
        ),
        (
            "leetcodeTopic",
            "Heap/Priority Queue",
            "Which heap or priority queue problems did I practice?",
            "medium",
        ),
        (
            "leetcodeTopic",
            "Arrays & Hashing",
            "What arrays and hashing problems have I done?",
            "medium",
        ),
        ("status", "Not started", "What tasks have I not started yet?", "medium"),
        ("status", "In progress", "What tasks are currently in progress?", "medium"),
        ("status", "Done", "What tasks have I already completed?", "medium"),
    ]
    for field_name, value, query, difficulty in natural:
        docs = corpus.with_property(field_name, value)
        if not docs:
            continue
        specs.append(
            Spec(
                id=f"metadata-nl-{_slug(field_name)}-{_slug(value)}",
                query=query,
                category="metadata",
                difficulty=difficulty,
                docs=docs,
                filters=(_eq(field_name, value),),
            )
        )

    # Explicit filter-syntax cases — easy, the rule-based analyzer handles these.
    explicit = [
        ("category", "AI Engineering"),
        ("category", "System Design"),
        ("leetcodeTopic", "Backtracking"),
        ("status", "Done"),
    ]
    for field_name, value in explicit:
        docs = corpus.with_property(field_name, value)
        if not docs:
            continue
        specs.append(
            Spec(
                id=f"metadata-syntax-{_slug(field_name)}-{_slug(value)}",
                query=f"{_label(field_name)}: {value}",
                category="metadata",
                difficulty="easy",
                docs=docs,
                filters=(_eq(field_name, value),),
            )
        )

    # The dominant topic (many docs) — a hard metadata recall target.
    graphs = corpus.with_property("leetcodeTopic", "Graphs")
    if graphs:
        specs.append(
            Spec(
                id="metadata-nl-leetcodetopic-graphs",
                query="Which leetcode graph problems have I worked on?",
                category="metadata",
                difficulty="hard",
                docs=graphs,
                filters=(_eq("leetcodeTopic", "Graphs"),),
            )
        )
    return specs


def lexical_specs(corpus: Corpus) -> list[Spec]:
    """Rare-term lexical cases: expectation is every doc whose body has the term."""
    terms = [
        ("grep", "Where did I mention grep?", "easy"),
        ("conda", "Which note talks about conda?", "easy"),
        ("indegree", "What note contains the word indegree?", "easy"),
        ("orthonormal", "Where did I write about orthonormal vectors?", "easy"),
        ("cgroups", "Where did I mention cgroups?", "easy"),
        ("tree shaking", "Where did I discuss tree shaking?", "easy"),
        ("Ackermann", "Where did I mention the Ackermann function?", "easy"),
        ("hallucination", "Where did I write about hallucinations?", "medium"),
        (
            "materialized read model",
            "Where did I mention a materialized read model?",
            "medium",
        ),
        ("denormalization", "Which note discusses denormalization?", "medium"),
        ("eigenvalues", "Where did I write about eigenvalues?", "medium"),
        ("Eulerian", "Where did I mention Eulerian paths?", "medium"),
        ("Kahn", "Where did I write about Kahn's algorithm?", "medium"),
        ("Apache Arrow", "Where did I mention Apache Arrow?", "easy"),
        ("gradient descent", "Where did I write about gradient descent?", "easy"),
        ("chain rule", "Which note mentions the chain rule?", "easy"),
        ("Hessian", "Where did I mention the Hessian?", "easy"),
        ("cross encoder", "Where did I write about the cross encoder?", "easy"),
        ("BM25", "Where did I mention BM25?", "easy"),
        ("sideEffects", "Which note mentions the sideEffects flag?", "medium"),
        ("pyproject", "Where did I mention pyproject.toml?", "easy"),
        ("columnar", "Where did I write about columnar storage?", "medium"),
        ("amortization", "Where did I mention amortization?", "medium"),
        ("recursion path", "Where did I write about the recursion path?", "medium"),
        ("Kodaikkanal", "Where did I mention my trip to Kodaikkanal?", "easy"),
        ("air fryer", "Which note talks about the air fryer?", "easy"),
        ("sick leave", "Where did I mention taking a sick leave?", "easy"),
    ]
    specs: list[Spec] = []
    for term, query, difficulty in terms:
        docs = corpus.containing(term)
        if not docs:
            continue
        specs.append(
            Spec(
                id=f"lexical-{_slug(term)}",
                query=query,
                category="lexical",
                difficulty=difficulty,
                docs=docs,
                intent="semantic",
            )
        )
    return specs


def leetcode_problem_specs(corpus: Corpus) -> list[Spec]:
    """Lexical recall of specific leetcode problems mentioned in daily notes."""
    problems = [
        ("Permutations", "Which day did I solve the Permutations problem?", "easy"),
        ("Subsets II", "When did I work on Subsets II?", "easy"),
        ("Number of Islands", "Which note has my Number of Islands solution?", "easy"),
        ("Max Area of Islands", "When did I do Max Area of Islands?", "easy"),
        ("Clone Graph", "Which day did I solve Clone Graph?", "easy"),
        ("Walls and Gates", "When did I practice Walls and Gates?", "easy"),
        ("Rotten Oranges", "Where did I mention Rotten Oranges?", "easy"),
        (
            "Pacific Atlantic Water Flow",
            "When did I solve Pacific Atlantic Water Flow?",
            "easy",
        ),
        ("Surrounded Regions", "Which day did I do Surrounded Regions?", "easy"),
        ("Course Schedule II", "When did I solve Course Schedule II?", "easy"),
        ("Graph valid tree", "Which note mentions Graph valid tree?", "easy"),
        ("Recreate Itinerary", "When did I work on Recreate Itinerary?", "easy"),
        ("Swim in Rising Water", "Where did I mention Swim in Rising Water?", "easy"),
        (
            "Kth largest element in a stream",
            "When did I do Kth largest element in a stream?",
            "easy",
        ),
    ]
    specs: list[Spec] = []
    for problem, query, difficulty in problems:
        docs = corpus.containing(problem)
        if not docs:
            continue
        specs.append(
            Spec(
                id=f"leetcode-{_slug(problem)}",
                query=query,
                category="lexical",
                difficulty=difficulty,
                docs=docs,
                intent="semantic",
            )
        )
    return specs


def temporal_specs(corpus: Corpus) -> list[Spec]:
    """Temporal cases grounded in real authoring windows (and topic overlaps)."""
    specs: list[Spec] = []

    # Whole-window authoring queries. Broad windows are legitimately hard (recall
    # is capped by how many in-window docs fit in top-k); MRR stays meaningful.
    windows = [
        (
            "week-jul13",
            "What did I work on during the week of July 13?",
            "2026-07-13",
            "2026-07-19",
            "hard",
        ),
        (
            "week-aug17",
            "What was I working on in the week of August 17?",
            "2026-08-17",
            "2026-08-23",
            "hard",
        ),
        (
            "week-jun28",
            "What did I do in the last week of June?",
            "2026-06-29",
            "2026-07-05",
            "hard",
        ),
    ]
    for suffix, query, start, end, difficulty in windows:
        docs = corpus.in_window(start, end)
        if not docs:
            continue
        specs.append(
            Spec(
                id=f"temporal-{suffix}",
                query=query,
                category="temporal",
                difficulty=difficulty,
                docs=docs,
                filters=(_between(AUTHORSHIP_FIELD, start, end),),
            )
        )

    # Topic-scoped temporal queries: intersect a leetcode topic with a window so
    # the expected set stays small and the case tests hybrid temporal + metadata.
    graph_ids = set(corpus.with_property("leetcodeTopic", "Graphs"))
    aug = set(corpus.in_window("2026-08-01", "2026-08-31"))
    graph_aug = tuple(sorted(graph_ids & aug))
    if graph_aug:
        specs.append(
            Spec(
                id="temporal-graphs-august",
                query="Which graph problems did I practice in August?",
                category="temporal",
                difficulty="hard",
                docs=graph_aug,
                filters=(
                    _eq("leetcodeTopic", "Graphs"),
                    _between(AUTHORSHIP_FIELD, "2026-08-01", "2026-08-31"),
                ),
                intent="temporal",
            )
        )
    return specs


def _curated_specs(corpus: Corpus) -> list[Spec]:
    """Hand-authored cases for the categories that need human judgment.

    ``docs`` are explicit, real document ids (validated by the builder). These
    cover factual recall, semantic recall, cross-document synthesis, ambiguous
    natural language, and daily-assistant behaviour.
    """
    # Concept / knowledge documents referenced repeatedly below.
    UNION_FIND = "3c2f37cd-5024-8067-89fa-fde0afa58aab"
    DIJKSTRA = "3bff37cd-5024-80fb-abe3-dee5713f2738"
    PRIMS = "3bff37cd-5024-8072-b6b8-ec6619271d66"
    GRAM_SCHMIDT = "3b1f37cd-5024-8040-970d-e993d4796d03"
    MATRIX_TRANSFORMS = "3bbf37cd-5024-80a9-8f82-efac83f87ebe"
    WEBPACK = "397f37cd-5024-80b7-afcf-cebda96e11a2"
    DOCKER = "3bff37cd-5024-808b-ad3e-c7e40ce249d1"
    DOCKER_VS_VMS = "390f37cd-5024-8044-8c49-ced6f93dca0d"
    CONGA = "382f37cd-5024-800a-b22e-eca13d16a73b"
    HARNESS = "386f37cd-5024-807f-9b60-e666cc9d3a4d"
    APACHE_ARROW = "390f37cd-5024-80d8-a0e9-e4a22bfae2f5"
    TOPO_SORT = "3a3f37cd-5024-8031-976f-e1d40ba64fc6"
    DFS_CYCLE = "3a3f37cd-5024-80e1-bbbd-ee5d72b036a5"
    EULERIAN = "3bff37cd-5024-8042-8982-f5c3fbb35d8f"
    GRAPH_NOTES = "396f37cd-5024-802f-b610-f4e8d9c91a31"
    LINEAR_ALGEBRA = "3b1f37cd-5024-80d0-9008-d5d872a2b71c"
    LINEAR_ALGEBRA_INT = "3b1f37cd-5024-8075-8892-d0aa0e02b938"
    SEMANTIC_INDEXING = "38ef37cd-5024-800f-85aa-cb4938a4c032"
    CHUNKING = "395f37cd-5024-801d-aeb6-d10e396ab5ac"
    HLD = "38ef37cd-5024-8095-928d-d2f839d65184"
    LLD = "38ef37cd-5024-80e5-bfc0-d7d0e789d52b"
    PRD = "38ef37cd-5024-8007-a4c0-d45385700bd4"
    FUTURE_CONS = "394f37cd-5024-801d-973a-c7dfebad5d2e"
    RETRIEVAL_QA = "38ef37cd-5024-806d-aa7b-ecf937e635c7"
    NEXT_STEPS = "386f37cd-5024-8021-8a2c-f251671a4822"
    AI_TREE = "3a1f37cd-5024-808d-842c-cbf567adfffb"
    IDEAL_ROUTINE = "3b1f37cd-5024-8037-8556-cedb7e8bf542"
    AUGUST_GOALS = "3aef37cd-5024-8197-bc02-d9e679f02fd3"
    WEEK_AUG10 = "3b9f37cd-5024-8024-925e-f33542d7da4d"
    WEEK_AUG3 = "3b1f37cd-5024-805a-8741-e3302649b139"
    VISIBILITY = "3b1f37cd-5024-8018-b36b-f60b1cc6f73c"
    REG_AGENT = "3bdf37cd-5024-80ac-806b-dcfcf9ce85c4"
    UI_POLISH = "38ef37cd-5024-8090-af70-c42e6d8b720a"
    BIRTHDAY = "38af37cd-5024-8118-ad6b-f5374e49f5c7"
    STORY4_DONE = "39af37cd-5024-81d6-bbd4-d35da3fd460c"
    STORY5_DONE = "399f37cd-5024-81a8-b5e0-d58610001d37"
    TIRED_TUE = "3b8f37cd-5024-81aa-9ada-e8ceabdf8a5e"

    BST = "3a8f37cd-5024-80d3-bd20-ce4d4aee6d45"
    PRIORITY_QUEUE = "3bff37cd-5024-80d6-8b81-c1efd3fee91f"
    TREES = "3a8f37cd-5024-8036-ad90-f5250bc563bc"
    BACKTRACKING_NOTES = "383f37cd-5024-803a-8e0f-c7168bcac780"
    KNOWLEDGE_INGESTION = "38ef37cd-5024-809f-9676-ca30880e5192"
    TERMINAL_SHELL = "395f37cd-5024-805f-97e5-fc3bee2607ec"
    PYTHON_ENVS = "390f37cd-5024-80b1-9cb7-c9daf04d5537"
    PROJECT_FOUNDATION = "38ef37cd-5024-80fe-8886-cb58228599fc"
    KNOWLEDGE_GRAPH = "38df37cd-5024-809a-aa98-e19537f56b4c"
    IDEA_BANK = "3c2f37cd-5024-8074-8270-f310194c1280"

    # difficulty: f=factual, s=semantic, y=synthesis, a=assistant, m=ambiguous
    specs = [
        # --- Factual recall -------------------------------------------------
        Spec("factual-union-find", "What did I write about union find?",
             "factual", "easy", (UNION_FIND,)),
        Spec("factual-dijkstra", "What did I note about Dijkstra's algorithm?",
             "factual", "easy", (DIJKSTRA,)),
        Spec("factual-prims", "What did I write about Prim's algorithm?",
             "factual", "easy", (PRIMS,)),
        Spec("factual-gram-schmidt", "What did I learn about the Gram-Schmidt process?",
             "factual", "easy", (GRAM_SCHMIDT,)),
        Spec("factual-eigenvalues",
             "What did I write about eigenvalues and eigenvectors?",
             "factual", "medium", (MATRIX_TRANSFORMS,)),
        Spec("factual-webpack", "What did I write about Webpack?",
             "factual", "easy", (WEBPACK,)),
        Spec("factual-docker", "What did I note about how Docker works?",
             "factual", "medium", (DOCKER, DOCKER_VS_VMS)),
        Spec("factual-conga", "What did I write about the Conga platform architecture?",
             "factual", "easy", (CONGA,)),
        Spec("factual-harness", "What did I write about harness engineering?",
             "factual", "easy", (HARNESS,)),
        Spec("factual-apache-arrow", "What did I note about Apache Arrow?",
             "factual", "easy", (APACHE_ARROW,)),
        Spec("factual-topological-sort", "What did I write about topological sort?",
             "factual", "easy", (TOPO_SORT,)),
        Spec("factual-linear-algebra", "What did I write about linear algebra?",
             "factual", "medium", (LINEAR_ALGEBRA, LINEAR_ALGEBRA_INT)),
        Spec("factual-embeddings",
             "What did I write about embeddings and semantic indexing?",
             "factual", "medium", (SEMANTIC_INDEXING, CHUNKING)),
        Spec("factual-bst", "What did I write about binary search trees?",
             "factual", "medium", (BST,)),
        Spec("factual-priority-queue",
             "What did I note about priority queues and heaps?",
             "factual", "easy", (PRIORITY_QUEUE,)),
        Spec("factual-trees", "What are my notes on trees?",
             "factual", "medium", (TREES,)),
        Spec("factual-backtracking-notes",
             "What did I write about the backtracking pattern?",
             "factual", "medium", (BACKTRACKING_NOTES,)),
        Spec("factual-ingestion",
             "What did I write about the knowledge ingestion pipeline?",
             "factual", "easy", (KNOWLEDGE_INGESTION,)),
        Spec("factual-terminal",
             "What did I note about the terminal and shell pipes?",
             "factual", "medium", (TERMINAL_SHELL,)),
        Spec("factual-python-envs",
             "What did I write about Python environments and uv?",
             "factual", "easy", (PYTHON_ENVS,)),
        Spec("factual-dfs-cycle",
             "What did I write about detecting cycles with DFS?",
             "factual", "medium", (DFS_CYCLE,)),
        Spec("factual-knowledge-graph",
             "What tools did I list under knowledge graph?",
             "factual", "medium", (KNOWLEDGE_GRAPH,)),
        # --- Semantic recall (little/no keyword overlap) --------------------
        Spec("semantic-privacy",
             "How does the project keep all my data private without using the cloud?",
             "semantic", "medium", (HLD, PRD)),
        Spec("semantic-masters-research",
             "What were my thoughts on pursuing a master's and a research career?",
             "semantic", "medium", (NEXT_STEPS, AI_TREE)),
        Spec("semantic-why-ai",
             "What first got me curious about machine learning?",
             "semantic", "medium", (AI_TREE,)),
        Spec("semantic-burnout",
             "What did I write about feeling drained or avoiding burnout?",
             "semantic", "hard", (WEEK_AUG3, WEEK_AUG10, TIRED_TUE)),
        Spec("semantic-online-presence",
             "What are my plans for building an online presence and audience?",
             "semantic", "hard", (VISIBILITY,)),
        Spec("semantic-scaling",
             "How would I keep retrieval quality high as the knowledge base grows?",
             "semantic", "medium", (FUTURE_CONS,)),
        Spec("semantic-isolation",
             "How do I stop different projects' dependencies from clashing?",
             "semantic", "medium", (PYTHON_ENVS,)),
        Spec("semantic-startup-speed",
             "Why do containers start so much faster than full virtual machines?",
             "semantic", "medium", (DOCKER_VS_VMS,)),
        Spec("semantic-side-idea",
             "What product idea did I jot down for an automated job hunter?",
             "semantic", "medium", (IDEA_BANK,)),
        # --- Cross-document synthesis --------------------------------------
        Spec("synthesis-rag-learnings",
             "What have I learned about building retrieval and RAG systems?",
             "synthesis", "hard",
             (HLD, LLD, PRD, CHUNKING, FUTURE_CONS, SEMANTIC_INDEXING, RETRIEVAL_QA)),
        Spec("synthesis-graph-algorithms",
             "What graph algorithms have I studied and taken notes on?",
             "synthesis", "hard",
             (GRAPH_NOTES, TOPO_SORT, DFS_CYCLE, EULERIAN, PRIMS, DIJKSTRA, UNION_FIND)),
        Spec("synthesis-math-foundations",
             "What have I been learning across the math foundations chapters?",
             "synthesis", "hard",
             (LINEAR_ALGEBRA, LINEAR_ALGEBRA_INT, GRAM_SCHMIDT, MATRIX_TRANSFORMS)),
        Spec("synthesis-masters-evolution",
             "How has my thinking about doing a master's evolved over time?",
             "synthesis", "hard", (NEXT_STEPS, AI_TREE)),
        Spec("synthesis-project-stories",
             "What were the main user stories and epics for the LifeRAG project?",
             "synthesis", "hard",
             (PROJECT_FOUNDATION, KNOWLEDGE_INGESTION, SEMANTIC_INDEXING,
              RETRIEVAL_QA, UI_POLISH)),
        Spec("synthesis-system-design",
             "What system design topics have I taken notes on?",
             "synthesis", "hard", (DOCKER, DOCKER_VS_VMS, CONGA, WEBPACK)),
        # --- Ambiguous natural language ------------------------------------
        Spec("ambiguous-birthday",
             "What was I doing around my birthday?",
             "ambiguous", "hard", (BIRTHDAY,)),
        Spec("ambiguous-ai-infra",
             "When did I last think about AI infrastructure and GPU work?",
             "ambiguous", "medium", (NEXT_STEPS,)),
        Spec("ambiguous-rag-stories",
             "When did I finish the stories for the RAG project?",
             "ambiguous", "medium", (STORY4_DONE, STORY5_DONE)),
        # --- Daily assistant ------------------------------------------------
        Spec("assistant-goals",
             "What goals have I set for this month?",
             "assistant", "medium", (AUGUST_GOALS,)),
        Spec("assistant-focus",
             "What am I currently focused on?",
             "assistant", "hard", (AUGUST_GOALS, WEEK_AUG10)),
        Spec("assistant-active-projects",
             "What projects am I actively working on?",
             "assistant", "hard", (REG_AGENT, UI_POLISH)),
        Spec("assistant-routine",
             "What did I say I want to improve about my daily routine?",
             "assistant", "hard", (IDEAL_ROUTINE, WEEK_AUG10)),
        Spec("assistant-procrastinating",
             "What have I been putting off or not started yet?",
             "assistant", "hard", (VISIBILITY,)),
        Spec("assistant-portfolio",
             "What am I planning to do about my portfolio and personal brand?",
             "assistant", "hard", (VISIBILITY, AUGUST_GOALS)),
    ]
    # Validate every hand-authored id exists in the index.
    for spec in specs:
        corpus.require(list(spec.docs))
    return specs


def factual_title_specs(corpus: Corpus) -> list[Spec]:
    """Factual/keyword cases driven by repeated workout titles (tag-like)."""
    specs: list[Spec] = []
    title_queries = [
        ("Shoulder", "What shoulder workouts have I logged?", "medium"),
        ("Leg", "Show my leg day workouts.", "medium"),
    ]
    for title, query, difficulty in title_queries:
        docs = corpus.titled(title)
        if not docs:
            continue
        specs.append(
            Spec(
                id=f"factual-workout-{_slug(title)}",
                query=query,
                category="factual",
                difficulty=difficulty,
                docs=docs,
                intent="semantic",
            )
        )
    return specs


def _slug(text: str) -> str:
    return "-".join(
        "".join(c if c.isalnum() else " " for c in text).split()
    ).lower()


def _label(field_name: str) -> str:
    # Human "category"/"leetcode topic"/"status" surface for explicit syntax.
    out = []
    for i, ch in enumerate(field_name):
        if ch.isupper() and i:
            out.append(" ")
        out.append(ch.lower())
    return "".join(out)


def build(corpus: Corpus) -> EvaluationDataset:
    specs: list[Spec] = []
    specs += metadata_specs(corpus)
    specs += lexical_specs(corpus)
    specs += leetcode_problem_specs(corpus)
    specs += temporal_specs(corpus)
    specs += factual_title_specs(corpus)
    specs += _curated_specs(corpus)

    seen: set[str] = set()
    cases: list[EvaluationCase] = []
    for spec in specs:
        if spec.id in seen:
            raise ValueError(f"Duplicate case id: {spec.id}")
        seen.add(spec.id)
        docs = corpus.require(list(spec.docs))
        cases.append(
            EvaluationCase(
                id=spec.id,
                query=spec.query,
                expected_documents=docs,
                expected_chunks=corpus.chunks_for(docs),
                category=spec.category,
                difficulty=spec.difficulty,
                expected_filters=spec.filters,
                expected_intent=spec.intent,
            )
        )
    return EvaluationDataset(cases=tuple(cases))


def main(output_path: Path = OUTPUT_PATH) -> None:
    qdrant = QdrantService(get_qdrant_client())
    if not qdrant.collection_exists() or qdrant.count() == 0:
        raise SystemExit("Qdrant collection is empty; index the corpus first.")

    corpus = load_corpus(qdrant)
    dataset = build(corpus)
    dataset.to_file(output_path)

    by_category: dict[str, int] = defaultdict(int)
    by_difficulty: dict[str, int] = defaultdict(int)
    for case in dataset.cases:
        by_category[case.category or "uncategorized"] += 1
        by_difficulty[case.difficulty or "unlabeled"] += 1

    print(f"Wrote {len(dataset)} cases to {output_path}")
    print("By category:")
    for name, count in sorted(by_category.items()):
        print(f"  {name:<12} {count}")
    print("By difficulty:")
    for name, count in sorted(by_difficulty.items()):
        print(f"  {name:<10} {count}")


if __name__ == "__main__":
    main()
