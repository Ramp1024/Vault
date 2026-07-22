-> Prompt engineering and Dynamic Prompt selection
    -> Understand role and intent clearly and act according to them
    -> Example 1:
        User query: What is Docker
        Internals:
            -> Fetch relevant 3-4 chunks -> Stream answer
    -> Example 2:
        User query: Summarize everything i know about Docker
        Internals:
            -> Fetch 15-20 chunks(if present) relevant to Docker, 
            -> Since user wants a summary of everything
            -> Better answer = More Relevant context
        
-> Retrieval Tuning
    -> Only top-k items
    -> Grounding thresholds, relevancy thresholds

-> Source citations

-> Hybrid Search
    -> Metadata based filtering followed by similarity search

-> Reranking
    -> Use re-ranking model to rank relevant chunks

-> Conversation memory

-> CRON job to run automated fetch_documentsions

-> Implement a small CLI
    -> vault init - setups the app
    -> vault sync - indexes notion
    -> vault chat - starts API and UI