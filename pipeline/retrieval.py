# # pipeline/retrieval.py
# """
# Phase 8: Hybrid retrieval — metadata filtering + vector search.
# """

# import numpy as np
# from sqlalchemy import create_engine, text
# from sentence_transformers import SentenceTransformer
# from pipeline.vector_store import load_index
# from pipeline.query_pipeline import QueryIntent
# from config import DB_URL, EMBED_MODEL, TOP_K_VECTOR

# engine = create_engine(DB_URL)

# def embed_query(query_text: str, model: SentenceTransformer) -> np.ndarray:
#     """Embed the user query with the same model used for chunks."""
#     return model.encode([query_text], normalize_embeddings=True)[0].astype("float32")

# def metadata_filter_sql(intent: QueryIntent) -> tuple[str, dict]:
#     """Build SQL WHERE clause from query intent filters."""
#     conditions = []
#     params = {}

#     if intent.date_from:
#         conditions.append("a.date_str >= :date_from")
#         params["date_from"] = intent.date_from

#     if intent.date_to:
#         conditions.append("a.date_str <= :date_to")
#         params["date_to"] = intent.date_to

#     if intent.location:
#         conditions.append("LOWER(a.location) LIKE :location")
#         params["location"] = f"%{intent.location.lower()}%"

#     if intent.actor:
#         conditions.append("(LOWER(a.actor1) LIKE :actor OR LOWER(a.actor2) LIKE :actor)")
#         params["actor"] = f"%{intent.actor.lower()}%"

#     where = "WHERE " + " AND ".join(conditions) if conditions else ""
#     return where, params

# def get_candidate_chunk_ids(intent: QueryIntent) -> list[int] | None:
#     """
#     Use metadata filters to narrow down candidate chunks.
#     Returns list of chunk IDs, or None if no filters (search all).
#     """
#     where, params = metadata_filter_sql(intent)

#     if not where:
#         return None  # No filters = search everything

#     sql = text(f"""
#         SELECT c.id
#         FROM chunks c
#         JOIN articles a ON c.article_id = a.id
#         {where}
#         AND c.embedding IS NOT NULL
#     """)

#     with engine.connect() as conn:
#         rows = conn.execute(sql, params).fetchall()

#     return [r.id for r in rows]

# def vector_search(
#     query_vec: np.ndarray,
#     faiss_index,
#     chunk_id_map: list[int],
#     candidate_ids: list[int] | None = None,
#     top_k: int = TOP_K_VECTOR
# ) -> list[int]:
#     """
#     Search FAISS for most similar chunks.
#     If candidate_ids is given, filter results to only those IDs.
#     """
#     # Search more candidates if we need to filter
#     search_k = top_k * 10 if candidate_ids else top_k

#     scores, positions = faiss_index.search(
#         query_vec.reshape(1, -1),
#         min(search_k, faiss_index.ntotal)
#     )

#     results = []
#     candidate_set = set(candidate_ids) if candidate_ids else None

#     for pos, score in zip(positions[0], scores[0]):
#         if pos < 0:
#             continue
#         chunk_id = chunk_id_map[pos]
#         if candidate_set is not None and chunk_id not in candidate_set:
#             continue
#         results.append(chunk_id)
#         if len(results) >= top_k:
#             break

#     return results

# def fetch_chunks_by_ids(chunk_ids: list[int]) -> list[dict]:
#     """Fetch full chunk data + article metadata from PostgreSQL."""
#     if not chunk_ids:
#         return []

#     sql = text("""
#         SELECT
#             c.id, c.chunk_text, c.chunk_index,
#             a.id AS article_id, a.title, a.source_url,
#             a.date_str, a.actor1, a.actor2,
#             a.location, a.event_type, a.avg_tone
#         FROM chunks c
#         JOIN articles a ON c.article_id = a.id
#         WHERE c.id = ANY(:ids)
#     """)

#     with engine.connect() as conn:
#         rows = conn.execute(sql, {"ids": chunk_ids}).mappings().fetchall()

#     return [dict(r) for r in rows]

# def retrieve(
#     intent: QueryIntent,
#     model: SentenceTransformer,
#     faiss_index,
#     chunk_id_map: list[int]
# ) -> list[dict]:
#     """Full hybrid retrieval pipeline."""
#     query_vec      = embed_query(intent.search_text, model)
#     candidate_ids  = get_candidate_chunk_ids(intent)
#     top_chunk_ids  = vector_search(query_vec, faiss_index, chunk_id_map, candidate_ids)
#     chunks         = fetch_chunks_by_ids(top_chunk_ids)
#     return chunks


"""
Phase 8: Hybrid retrieval — metadata filtering + vector search + scoring
"""

import numpy as np
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from pipeline.query_pipeline import QueryIntent
from config import DB_URL, TOP_K_VECTOR

engine = create_engine(DB_URL)


# -------------------------------
# 🔍 EMBEDDING
# -------------------------------
def embed_query(query_text: str, model: SentenceTransformer) -> np.ndarray:
    return model.encode([query_text], normalize_embeddings=True)[0].astype("float32")


# -------------------------------
# 🧠 METADATA FILTER (SOFT)
# -------------------------------
def metadata_filter_sql(intent: QueryIntent) -> tuple[str, dict]:
    conditions = []
    params = {}

    if intent.date_from:
        conditions.append("a.date_str >= :date_from")
        params["date_from"] = intent.date_from

    if intent.date_to:
        conditions.append("a.date_str <= :date_to")
        params["date_to"] = intent.date_to

    if intent.location:
        conditions.append("LOWER(a.location) LIKE :location")
        params["location"] = f"%{intent.location.lower()}%"

    if intent.actor:
        conditions.append("(LOWER(a.actor1) LIKE :actor OR LOWER(a.actor2) LIKE :actor)")
        params["actor"] = f"%{intent.actor.lower()}%"

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    return where, params


# -------------------------------
# 📊 METADATA SCORING (BOOST)
# -------------------------------
def compute_metadata_score(intent: QueryIntent, row: dict) -> float:
    score = 0.0

    # Location boost
    if intent.location and row.get("location"):
        if intent.location.lower() in row["location"].lower():
            score += 0.15

    # Actor boost
    if intent.actor and (row.get("actor1") or row.get("actor2")):
        actors = f"{row.get('actor1','')} {row.get('actor2','')}".lower()
        if intent.actor.lower() in actors:
            score += 0.15

    # Event type boost
    if intent.event_type and row.get("event_type"):
        if intent.event_type.lower() in row["event_type"].lower():
            score += 0.15

    # Tone boost (optional)
    if row.get("avg_tone") is not None:
        if row["avg_tone"] > 0:
            score += 0.05

    return score


# -------------------------------
# 🔎 VECTOR SEARCH
# -------------------------------
def vector_search(
    query_vec: np.ndarray,
    faiss_index,
    chunk_id_map: list[int],
    candidate_ids: list[int] | None = None,
    top_k: int = TOP_K_VECTOR
):
    search_k = top_k * 10 if candidate_ids else top_k

    scores, positions = faiss_index.search(
        query_vec.reshape(1, -1),
        min(search_k, faiss_index.ntotal)
    )

    results = []
    candidate_set = set(candidate_ids) if candidate_ids else None

    for pos, score in zip(positions[0], scores[0]):
        if pos < 0:
            continue

        chunk_id = chunk_id_map[pos]

        if candidate_set is not None and chunk_id not in candidate_set:
            continue

        results.append((chunk_id, float(score)))

        if len(results) >= top_k * 3:  # keep extra for reranking
            break

    return results


# -------------------------------
# 📄 FETCH CHUNKS + METADATA
# -------------------------------
def fetch_chunks(chunk_score_pairs: list[tuple[int, float]]) -> list[dict]:
    if not chunk_score_pairs:
        return []

    chunk_ids = [cid for cid, _ in chunk_score_pairs]
    score_map = {cid: score for cid, score in chunk_score_pairs}

    sql = text("""
        SELECT
            c.id, c.chunk_text, c.chunk_index,
            a.id AS article_id, a.title, a.source_url,
            a.date_str, a.actor1, a.actor2,
            a.location, a.event_type, a.avg_tone
        FROM chunks c
        JOIN articles a ON c.article_id = a.id
        WHERE c.id = ANY(:ids)
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, {"ids": chunk_ids}).mappings().fetchall()

    results = []
    for r in rows:
        d = dict(r)
        d["semantic_score"] = score_map.get(d["id"], 0.0)
        results.append(d)

    return results


# -------------------------------
# 🚀 FINAL RETRIEVE (HYBRID)
# -------------------------------
def retrieve(
    intent: QueryIntent,
    model: SentenceTransformer,
    faiss_index,
    chunk_id_map: list[int]
) -> list[dict]:

    # Step 1: Embed query
    query_vec = embed_query(intent.search_text, model)

    # Step 2: Metadata filter (optional)
    where, params = metadata_filter_sql(intent)

    candidate_ids = None
    if where:
        sql = text(f"""
            SELECT c.id
            FROM chunks c
            JOIN articles a ON c.article_id = a.id
            {where}
            AND c.embedding IS NOT NULL
        """)

        with engine.connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        candidate_ids = [r[0] for r in rows]

    # Step 3: Vector search
    chunk_score_pairs = vector_search(
        query_vec,
        faiss_index,
        chunk_id_map,
        candidate_ids
    )

    # Step 4: Fetch chunks
    chunks = fetch_chunks(chunk_score_pairs)

    # Step 5: Hybrid scoring
    for c in chunks:
        meta_score = compute_metadata_score(intent, c)

        c["final_score"] = (
            0.7 * c["semantic_score"] +
            0.3 * meta_score
        )

    # Step 6: Sort by final score
    chunks = sorted(chunks, key=lambda x: x["final_score"], reverse=True)

    # Step 7: Return top_k
    return chunks[:TOP_K_VECTOR]