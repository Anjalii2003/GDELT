# pipeline/05_embed.py

import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from tqdm import tqdm
from config import DB_URL, EMBED_MODEL

BATCH_SIZE = 64
FETCH_SIZE = 500


def load_model():
    print(f"Loading embedding model: {EMBED_MODEL}")

    model = SentenceTransformer(
        EMBED_MODEL,
        cache_folder="/root/.cache/huggingface"
    )

    print(f"Model dimension: {model.get_sentence_embedding_dimension()}")
    return model


def get_unembedded_chunks(engine, limit=FETCH_SIZE):
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, chunk_text
            FROM chunks
            WHERE embedding IS NULL
            LIMIT :limit
        """), {"limit": limit}).fetchall()
    return rows


def update_embeddings_bulk(engine, chunk_ids, embeddings):
    """🔥 FINAL FIX — direct pgvector insert"""

    with engine.begin() as conn:
        for cid, emb in zip(chunk_ids, embeddings):

            # Convert embedding to pgvector string format
            vector_str = "[" + ",".join(map(str, emb.tolist())) + "]"

            # Direct SQL (no parameter binding issues)
            conn.execute(
                text(f"""
                    UPDATE chunks
                    SET embedding = '{vector_str}'::vector
                    WHERE id = {cid}
                """)
            )


def embed_all_chunks(engine, model):
    print("Generating embeddings...")

    total_processed = 0

    while True:
        chunks = get_unembedded_chunks(engine)
        if not chunks:
            break

        chunk_ids = [c[0] for c in chunks]
        chunk_texts = [c[1] for c in chunks]

        all_embeddings = []

        for i in range(0, len(chunk_texts), BATCH_SIZE):
            batch = chunk_texts[i:i+BATCH_SIZE]

            embeddings = model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=BATCH_SIZE
            )

            all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings)

        # 🔥 CALL FIXED FUNCTION
        update_embeddings_bulk(engine, chunk_ids, all_embeddings)

        total_processed += len(chunk_ids)
        print(f"Processed {total_processed:,} chunks...")

    print(f"\n✅ Embedding complete. Total: {total_processed:,}")


def main():
    engine = create_engine(DB_URL)
    model = load_model()
    embed_all_chunks(engine, model)


if __name__ == "__main__":
    main()