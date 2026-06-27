# # pipeline/vector_store.py
# """
# Phase 6: Build a FAISS index from all chunk embeddings.
# The index maps FAISS position -> chunk_id in PostgreSQL.
# """

# import os
# import json
# import numpy as np
# import faiss
# from sqlalchemy import create_engine, text
# from tqdm import tqdm
# from config import DB_URL, FAISS_INDEX

# def build_faiss_index(engine) -> tuple[faiss.Index, list[int]]:
#     """
#     Load all embeddings from PostgreSQL and build a FAISS index.
#     Returns (index, list_of_chunk_ids) — position in list = FAISS index position.
#     """
#     print("Loading embeddings from database...")

#     with engine.connect() as conn:
#         rows = conn.execute(text("""
#             SELECT id, embedding FROM chunks
#             WHERE embedding IS NOT NULL
#             ORDER BY id
#         """)).fetchall()

#     if not rows:
#         raise ValueError("No embeddings found. Run Phase 5 first.")

#     chunk_ids  = [r.id for r in rows]
#     embeddings = np.array([r.embedding for r in rows], dtype="float32")

#     print(f"  Loaded {len(chunk_ids):,} embeddings, dimension={embeddings.shape[1]}")

#     # Inner Product index (best for normalized vectors = cosine similarity)
#     dim   = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     index.add(embeddings)

#     print(f"  FAISS index built with {index.ntotal:,} vectors")
#     return index, chunk_ids

# def save_index(index: faiss.Index, chunk_ids: list[int]):
#     os.makedirs(os.path.dirname(FAISS_INDEX), exist_ok=True)
#     faiss.write_index(index, FAISS_INDEX)

#     id_map_path = FAISS_INDEX.replace(".index", "_id_map.json")
#     with open(id_map_path, "w") as f:
#         json.dump(chunk_ids, f)

#     print(f"Saved FAISS index to {FAISS_INDEX}")
#     print(f"Saved ID map to {id_map_path}")

# def load_index() -> tuple[faiss.Index, list[int]]:
#     index = faiss.read_index(FAISS_INDEX)
#     id_map_path = FAISS_INDEX.replace(".index", "_id_map.json")
#     with open(id_map_path) as f:
#         chunk_ids = json.load(f)
#     return index, chunk_ids

# def main():
#     engine = create_engine(DB_URL)
#     index, chunk_ids = build_faiss_index(engine)
#     save_index(index, chunk_ids)

# if __name__ == "__main__":
#     main()

# pipeline/vector_store.py

# pipeline/vector_store.py

import sys
import os

# 🔥 Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import faiss
from sqlalchemy import create_engine, text
from tqdm import tqdm
from config import DB_URL, FAISS_INDEX


def build_faiss_index(engine):
    print("Loading embeddings from database...")

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, embedding FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY id
        """)).fetchall()

    if not rows:
        raise ValueError("❌ No embeddings found. Run Phase 5 first.")

    chunk_ids = []
    embeddings = []

    valid = 0
    skipped = 0

    for r in tqdm(rows, desc="Processing embeddings"):
        cid = r[0]
        emb = r[1]

        if emb is None:
            skipped += 1
            continue

        try:
            # 🔥 FIX 1: Handle pgvector string format "[...]"
            if isinstance(emb, str):
                emb = emb.strip("[]")
                emb = [float(x) for x in emb.split(",") if x.strip()]

            # 🔥 FIX 2: Handle pgvector object
            elif hasattr(emb, "tolist"):
                emb = emb.tolist()

            # 🔥 Convert to numpy
            emb_array = np.array(emb, dtype="float32")

            # Validate shape
            if emb_array.ndim != 1:
                skipped += 1
                continue

            embeddings.append(emb_array)
            chunk_ids.append(cid)
            valid += 1

        except Exception:
            skipped += 1
            continue

    print(f"\nValid embeddings: {valid}")
    print(f"Skipped embeddings: {skipped}")

    # 🔥 Safety check
    if len(embeddings) == 0:
        raise ValueError("❌ No valid embeddings parsed — check conversion.")

    embeddings = np.vstack(embeddings)

    print(f"\nLoaded {len(chunk_ids):,} embeddings")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # 🔥 FAISS (cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal:,} vectors")

    return index, chunk_ids


def save_index(index, chunk_ids):
    os.makedirs(os.path.dirname(FAISS_INDEX), exist_ok=True)

    faiss.write_index(index, FAISS_INDEX)

    id_map_path = FAISS_INDEX.replace(".index", "_id_map.json")

    with open(id_map_path, "w") as f:
        json.dump(chunk_ids, f)

    print(f"\nSaved FAISS index to {FAISS_INDEX}")
    print(f"Saved ID map to {id_map_path}")


def load_index():
    index = faiss.read_index(FAISS_INDEX)

    id_map_path = FAISS_INDEX.replace(".index", "_id_map.json")

    with open(id_map_path) as f:
        chunk_ids = json.load(f)

    return index, chunk_ids


def main():
    engine = create_engine(DB_URL)

    index, chunk_ids = build_faiss_index(engine)

    save_index(index, chunk_ids)


if __name__ == "__main__":
    main()