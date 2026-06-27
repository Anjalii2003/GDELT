# GDELT RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built on GDELT global news event data.

## What it does
- Ingests GDELT news event data
- Chunks and embeds articles into a FAISS vector store
- Answers natural language questions about global news events
- Returns answers along with the top source articles used for retrieval

## Tech Stack
- Python
- FAISS (vector database)
- Ollama (LLM)
- PostgreSQL
- Docker

## Pipeline
1. `01_ingest.py` — Download and load GDELT data
2. `02_fetch_articles.py` — Fetch article content
3. `03_structure_data.py` — Clean and structure data
4. `04_chunk.py` — Split text into chunks
5. `05_embed.py` — Generate embeddings and store in FAISS
6. `query_pipeline.py` — Handle user queries end to end

## How to run
```bash
docker-compose up
```
