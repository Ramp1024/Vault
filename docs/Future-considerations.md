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

➡️ Multi-Connector Support (GitHub, Markdown, PDFs)

➡️ Incremental Re-indexing & Background Sync

➡️ Production API & UI polish

->  Consider Cross Document Synthesis
->  Add parsing of Code block in ingestion
->  Rate limits

->  Fix dates
    ->  Use dateparser python library
    ->  Use LLM intent parser only for grammar
    ->  Make datetime system fields on notion pages are indexable

-> Debug Weekly Report pages