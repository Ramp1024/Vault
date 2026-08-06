Up next: Build a classifier that segregates filterable metadata from semantic content

Payload indexing / schema

 Free-text keyword indexes are dead weight — notes, quickJournal, aiProgress, personalWin, workWin get KEYWORD indexes that only match the entire value exactly; won't produce useful filter hits. Scope keyword indexing to discrete fields (category, leetcodeTopic, leetcode).
 CONTAINS on scalar strings = exact equality, not substring — silent false-negatives when the LLM emits "notes contains X". Either make it a real text match or drop CONTAINS for free-text.
 Index correctness rides entirely on infer_field_type — a non-ISO date would be typed STRING→KEYWORD and date range filters would silently degrade; boolean-ish leetcode is typed as string.
 Type-change reindex branch untested — on a type flip we call create_payload_index without delete-first; Qdrant behavior unverified. Consider safe delete-then-recreate.
 No stale-index GC — index lingers if a field drops out of the schema (harmless, untidy).
 NUMBER/FLOAT path unexercised — no numeric fields in current schema; correct-by-construction but untested on real data.
Intent analyzer / LLM

 Latency — intent adds a second full LLM inference (~74–93s cold), risking OLLAMA_TIMEOUT_SECONDS fallback; consider a smaller/faster dedicated intent model, a separate intent timeout, or model warming.
 Date/point-in-time resolution is prompt-only — fixes #1 (current-date injection) and #3 (single-day → =) are probabilistic guidance, not guaranteed; the model can still mis-resolve.

-> Conversation memory

-> CRON job to run automated fetch_documentsions

-> Implement a small CLI
    -> vault init - setups the app
    -> vault sync - indexes notion
    -> vault chat - starts API and UI

Next phases

➡️ Context Builder

➡️ Answer Generation

➡️ Source Citations

➡️ Conversational Benchmark

➡️ LLM Query Enhancement

➡️ Streaming Responses

➡️ Multi-Connector Support (GitHub, Markdown, PDFs)

➡️ Incremental Re-indexing & Background Sync

➡️ Production API & UI polish