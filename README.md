# Vault

> Your second brain, with a search bar that actually understands you.

Vault is a private, local-first RAG engine that sits on top of your Notion
workspace and answers questions about **your own notes** — in the blink of an
eye, with citations, without you ever having to remember which page you scribbled
that idea on.

---

## Why Vault exists

I take a lot of notes. Workout logs, LeetCode patterns, system-design sketches,
half-formed research ideas, journal entries. The problem was never *writing*
things down — it was *finding* them again.

Notion's search is keyword-literal. Ask it "what graph algorithms have I studied?"
and it shrugs unless you happen to use those exact words on a page. So the notes
piled up into a graveyard of good ideas I could never retrieve at the moment I
needed them. I was tracing through a dozen pages to reconstruct a concept I'd
already understood months ago.

**Vault fixes that.** Ask it a question the way you'd ask a friend who read
everything you ever wrote:

- *"Where did I mention grep?"*
- *"What have I learned about building RAG systems?"*
- *"What did I work on during the week of July 13?"*

…and it hands back a grounded answer with `[1] [2]` citations pointing straight
to the source notes. Concepts in one shot. No file-tracing. Everything runs on
your own machine — **no cloud, no API keys, your notes never leave your laptop.**

---

## What makes it interesting

Most RAG demos are "embed everything, stuff top-k into a prompt, hope for the
best." Vault is opinionated about the parts that actually break in the real
world:

- **Deterministic-first query understanding.** The LLM is *advisory*, never
  authoritative. A rule-based analyzer decides intent and filters; the LLM can
  only *add* constraints that survive a four-gate validation pipeline. It can
  never hallucinate a filter that tanks your recall. (More on why below — this
  is backed by benchmarks.)
- **Hybrid retrieval done honestly.** Dense embeddings *and* sparse BM25, fused
  with Reciprocal Rank Fusion so the two incompatible score scales never fight.
- **Metadata schema as a single source of truth.** The Notion connector
  discovers your properties (status, tags, topics, dates) at sync time and that
  one schema drives *everything*: Qdrant payload indexes, filter validation, and
  what the LLM is even allowed to propose.
- **Dates are math, not vibes.** LLMs are famously bad at date arithmetic, so
  "last week of June" is resolved deterministically with `dateparser` and an
  explicit Monday–Sunday / 1st–last boundary policy — timezone-anchored.
- **Streaming answers, backend-authoritative citations.** Tokens stream to the
  UI, but citations are validated *after* generation against the retrieved
  context, so the model can't invent a source.

---

## Technical architecture (the 2-minute version)

Vault is a small, boring-in-the-good-way stack: a FastAPI backend, a React +
Vite frontend, Qdrant for vectors, and Ollama running local models. No hosted
inference anywhere.

```
                 ┌──────────────┐
   Notion  ─────▶│  SyncService │  discover → chunk → embed → index
                 └──────┬───────┘
                        │  (nomic-embed-text, 768-dim)
                        ▼
                 ┌──────────────┐        ┌──────────────────┐
                 │    Qdrant    │◀──────▶│  BM25 index (.pkl)│
                 │ (dense + meta│        │   sparse/lexical  │
                 │   payload)   │        └──────────────────┘
                 └──────┬───────┘
                        ▲
   "what did I ...?"    │
        │               │
        ▼               │
 ┌──────────────┐   ┌───┴──────────────────────────────────────┐
 │  Chat API    │──▶│              SearchEngine                 │
 │ (streaming)  │   │  Vector  +  BM25  → RRF fusion → rerank   │
 └──────┬───────┘   └───┬──────────────────────────────────────┘
        │               │
        ▼               ▼
 ┌──────────────┐   ┌──────────────────────────────────────────┐
 │ AnswerService│◀──│  Query intent: Deterministic (+ optional  │
 │  prompt →    │   │  LLM augmenting, 4-gate validated)        │
 │  llama3.1:8b │   └──────────────────────────────────────────┘
 │  → citations │
 └──────────────┘
```

**Ingestion.** [`SyncService`](backend/app/services/sync_service.py) pulls pages
via the [`NotionConnector`](backend/app/connectors/notion/connector.py),
discovers a [`MetadataSchema`](backend/app/models/metadata_schema.py),
hierarchically chunks each page with the
[`Chunker`](backend/app/processors/chunker.py) (title + metadata prepended so
every chunk is self-contained), embeds with `nomic-embed-text`, and upserts to
Qdrant. The [`PayloadIndexManager`](backend/app/services/payload_index_manager.py)
provisions payload indexes straight from the schema, and the BM25 index is
rebuilt from the full store. Sync is incremental — unchanged pages are skipped.

**Retrieval.** [`SearchEngine`](backend/app/search/engine.py) runs
[`VectorSearchStrategy`](backend/app/search/strategy.py) and
[`BM25SearchStrategy`](backend/app/search/strategy.py), merges them with
[`ReciprocalRankFusion`](backend/app/search/fusion.py) (`k = 60`), and optionally
reranks with a cross-encoder
([`ms-marco-MiniLM-L-6-v2`](backend/app/search/reranker.py)). Retrieval is
**fail-open**: if metadata filters return nothing, it retries without them and
tells generation the filters were relaxed.

**Query understanding.** The
[`DeterministicIntentAnalyzer`](backend/app/processors/query_intent.py) is the
authority — it routes lexical queries to subject-only search and extracts
filters only from *known* schema values. When enabled, the
[`AugmentingIntentAnalyzer`](backend/app/processors/augmenting_analyzer.py) lets
an LLM *propose* extra constraints that must clear four gates — schema, grounding,
role, and confidence — in the
[`ConstraintValidator`](backend/app/processors/constraint_validation.py).
Deterministic filters always win.

**Answering.** [`AnswerService`](backend/app/services/answer_service.py) builds a
token-budgeted context, prompts `llama3.1:8b` at low temperature, streams tokens
to the client, and resolves `[n]` citations against the retrieved context with
the [`CitationMapper`](backend/app/processors/citation_mapper.py).

---

## Metrics — and why we stopped here

Vault ships with a real evaluation harness
([`app/evaluation/`](backend/app/evaluation/)), not vibes. The production
benchmark is **108 corpus-grounded cases** — real queries against real Qdrant
document IDs — spread across eight categories:

| lexical | factual | metadata | semantic | synthesis | assistant | temporal | ambiguous |
|--------:|--------:|---------:|---------:|----------:|----------:|---------:|----------:|
| 41 | 24 | 15 | 9 | 6 | 6 | 4 | 3 |

Every run reports **Recall@5 / Recall@10 / MRR**, a per-category and
per-difficulty breakdown, **query-analysis accuracy** (filter precision/recall,
intent classification, date-resolution accuracy), and **hybrid contribution**
(what fusion solves that neither vector nor BM25 solves alone).

```bash
cd backend
PYTHONPATH=. .venv/bin/python -m app.evaluation.run_production --rerank --markdown
```

### The result that shaped the whole design

We tested the obvious "just let the LLM figure out the filters" approach against
the deterministic analyzer. It lost — badly:

- **Rule-based analyzer: Recall@5 ≈ 0.76.**
- **LLM/composite analyzer: Recall@5 ≈ 0.27.** A ~65% drop.

The LLM wasn't adding intelligence, it was adding *hallucinated filters*:
matching "mention BM25" to a page property, snapping `status = done` onto "have I
done…", mangling canonical values (`heap or priority queue` vs `Heap/Priority
Queue`), and emitting timezone-shifted, off-by-one dates — all while running an
extra, slower inference step per query for the privilege.

That's *why* the architecture is deterministic-first. The lesson wasn't "LLMs are
useless" — it was "LLMs make excellent *proposers* and terrible *deciders*." So
the LLM was demoted to an augmenting role behind a validation wall it cannot get
past without grounded evidence, and date logic was handed to `dateparser` with an
explicit boundary policy. **We stopped here because the deterministic baseline is
fast, explainable, and every attempt to make the LLM "smarter" measurably made
retrieval worse.** The benchmark is now a permanent guardrail: any change has to
clear the rule-based bar before it ships.

---

## Example queries

Vault is built to handle very different *kinds* of asking, not just keyword
lookup:

| You ask… | Category | What Vault does |
|----------|----------|-----------------|
| *Where did I mention grep?* | lexical | Verb-anchored → subject-only BM25, no filters |
| *What did I write about union find?* | factual | Hybrid retrieval over your notes |
| *What AI engineering notes do I have?* | metadata | Schema-validated filter on your topic property |
| *What did I work on during the week of July 13?* | temporal | Deterministic Mon–Sun date range on `last_edited_time` |
| *How does the project keep all my data private?* | semantic | Dense embedding search over concepts |
| *What have I learned about building RAG systems?* | synthesis | Multi-note retrieval + grounded summary |
| *What am I currently focused on?* | assistant | Reads goals/status from your workspace |
| *What was I doing around my birthday?* | ambiguous | Best-effort temporal + semantic, fail-open |

Every answer comes back with `[1] [2]` citations you can click through to the
exact source note.

---

## Getting started

Vault runs entirely on your machine. You need Docker, and Ollama running
natively.

### 1. Install Ollama + models

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text   # embeddings
ollama pull llama3.1:8b        # generation
```

### 2. Start Ollama

```bash
ollama serve
```

> **Base URL:** inside Docker Compose the API reaches Ollama at
> `http://host.docker.internal:11434` (the compose default). Running the backend
> directly on the host? Use `http://localhost:11434`.

### 3. Bring up the stack

Docker Compose starts the app services (it does **not** manage Ollama):

```bash
docker-compose up --build
```

### 4. Sync your notes

```bash
cd backend
uv run python -m app.cli.sync
```

Then open the frontend and start asking your notes questions.

### Handy config

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `GENERATION_MODEL` | `llama3.1:8b` | Answer model |
| `RETRIEVAL_MODE` | `hybrid` | `vector` / `bm25` / `hybrid` |
| `RERANK_ENABLED` | `false` | Cross-encoder reranking |
| `INTENT_ANALYZER_ENABLED` | `false` | Turn on LLM augmentation (validated) |
| `APP_TIMEZONE` | `UTC` | Anchors relative-date queries — set to your zone |

Full reference lives in [`backend/app/core/config.py`](backend/app/core/config.py).

---

## Project layout

```
backend/    FastAPI · retrieval · query analysis · evaluation harness
frontend/   React + Vite + TypeScript chat UI
docs/       Key decision points & future considerations
docker-compose.yml
```

Deep dives: [`backend/README.md`](backend/README.md),
[`OLLAMA_SETUP.md`](OLLAMA_SETUP.md),
[`docs/Key-decision-points.md`](docs/Key-decision-points.md).

---

*Vault is a personal project — a search bar for a brain that writes too much and
remembers too little.*
