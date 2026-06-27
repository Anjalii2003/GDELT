# chatbot/app.py

"""
Phase 11: End-to-end chatbot that ties all phases together.
Run: python chatbot/app.py
"""

import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine

# ✅ FIXED IMPORTS (no numbers)
from pipeline.vector_store import load_index
from pipeline.query_pipeline import parse_query
from pipeline.retrieval import retrieve
from pipeline.rerank import load_reranker, rerank
from pipeline.rag import generate_answer

from config import DB_URL, EMBED_MODEL


# ----------------------------------
# 🔥 LOAD ALL COMPONENTS
# ----------------------------------
def load_all_components():
    print("Loading components...")

    print("  [1/3] Loading embedding model...")
    embed_model = SentenceTransformer(
        EMBED_MODEL,
        cache_folder="/root/.cache/huggingface"  # ✅ avoid HF rate limit
    )

    print("  [2/3] Loading reranker...")
    reranker = load_reranker()

    print("  [3/3] Loading FAISS index...")
    faiss_index, chunk_id_map = load_index()

    print("All components loaded!\n")

    return embed_model, reranker, faiss_index, chunk_id_map


# ----------------------------------
# 🔍 FULL PIPELINE
# ----------------------------------
def answer_question(
    query: str,
    embed_model,
    reranker,
    faiss_index,
    chunk_id_map
) -> dict:

    print(f"\nProcessing: {query}")

    # Step 1: Parse query
    intent = parse_query(query)

    print(f"  Date filter: {intent.date_from} → {intent.date_to}")
    print(f"  Event type:  {intent.event_type or 'any'}")

    # Step 2: Retrieve
    chunks = retrieve(intent, embed_model, faiss_index, chunk_id_map)

    print(f"  Retrieved:   {len(chunks)} chunks")

    if not chunks:
        return {
            "answer": "No relevant articles found.",
            "sources": []
        }

    # Step 3: Rerank
    top_chunks = rerank(intent.raw_query, chunks, reranker)

    print(f"  After rerank: {len(top_chunks)} chunks selected")

    # Step 4: Generate answer
    result = generate_answer(intent.raw_query, top_chunks)

    return result


# ----------------------------------
# 📄 FORMAT OUTPUT
# ----------------------------------
def format_output(result: dict) -> str:
    lines = [
        "=" * 60,
        "ANSWER:",
        result["answer"],
        "",
        "SOURCES:",
    ]

    for i, url in enumerate(result["sources"], 1):
        lines.append(f"  [{i}] {url}")

    lines.append("=" * 60)

    return "\n".join(lines)


# ----------------------------------
# 🚀 MAIN LOOP
# ----------------------------------
def main():
    embed_model, reranker, faiss_index, chunk_id_map = load_all_components()

    print("GDELT RAG Chatbot")
    print("Type your question (or 'quit' to exit)\n")

    while True:
        query = input("You: ").strip()

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            break

        result = answer_question(
            query,
            embed_model,
            reranker,
            faiss_index,
            chunk_id_map
        )

        print(format_output(result))


# ----------------------------------
if __name__ == "__main__":
    main()