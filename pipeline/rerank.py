# pipeline/rerank.py
"""
Phase 9: Rerank retrieved chunks using BGE reranker.
The reranker reads (query, chunk) pairs and gives precise relevance scores.
"""

from sentence_transformers import CrossEncoder
from config import RERANK_MODEL, TOP_K_RERANK

def load_reranker() -> CrossEncoder:
    print(f"Loading reranker: {RERANK_MODEL}")
    return CrossEncoder(RERANK_MODEL, max_length=512)

def rerank(
    query: str,
    chunks: list[dict],
    reranker: CrossEncoder,
    top_k: int = TOP_K_RERANK
) -> list[dict]:
    """
    Score each (query, chunk) pair and return top_k highest-scoring chunks.
    """
    if not chunks:
        return []

    pairs  = [(query, chunk["chunk_text"]) for chunk in chunks]
    scores = reranker.predict(pairs)

    # Attach scores and sort descending
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]