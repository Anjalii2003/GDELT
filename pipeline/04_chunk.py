# # pipeline/04_chunk.py
# """
# Phase 4: Split article text into overlapping chunks.
# Uses LangChain's RecursiveCharacterTextSplitter (token-aware).
# """

# import tiktoken
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sqlalchemy import create_engine, text
# from tqdm import tqdm
# from config import DB_URL, CHUNK_SIZE, CHUNK_OVERLAP

# def get_text_splitter():
#     """
#     RecursiveCharacterTextSplitter splits on paragraphs first,
#     then sentences, then words. Much better than fixed-size splits.
#     """
#     encoding = tiktoken.get_encoding("cl100k_base")

#     def token_len(text: str) -> int:
#         return len(encoding.encode(text))

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         length_function=token_len,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     return splitter

# def chunk_articles(engine, splitter):
#     """Read articles from DB and insert chunks."""

#     # Get articles that haven't been chunked yet
#     with engine.connect() as conn:
#         articles = conn.execute(text("""
#             SELECT a.id, a.article_text, a.title, a.source_url
#             FROM articles a
#             LEFT JOIN chunks c ON a.id = c.article_id
#             WHERE c.id IS NULL
#               AND a.article_text IS NOT NULL
#               AND LENGTH(a.article_text) > 100
#         """)).fetchall()

#     print(f"Chunking {len(articles):,} articles...")

#     for article in tqdm(articles):
#         # Prepend title to each article for context
#         full_text = f"Title: {article.title}\n\n{article.article_text}"
#         chunks = splitter.split_text(full_text)

#         chunk_records = [
#             {
#                 "article_id":  article.id,
#                 "chunk_index": i,
#                 "chunk_text":  chunk,
#             }
#             for i, chunk in enumerate(chunks)
#         ]

#         if not chunk_records:
#             continue

#         with engine.begin() as conn:
#             conn.execute(
#                 text("""
#                     INSERT INTO chunks (article_id, chunk_index, chunk_text)
#                     VALUES (:article_id, :chunk_index, :chunk_text)
#                 """),
#                 chunk_records
#             )

#     print("Chunking complete.")

# def main():
#     engine = create_engine(DB_URL)
#     splitter = get_text_splitter()
#     chunk_articles(engine, splitter)

# if __name__ == "__main__":
#     main()
# pipeline/04_chunk.py

import sys
import os

# 🔥 Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, text
from tqdm import tqdm
from config import DB_URL, CHUNK_SIZE, CHUNK_OVERLAP


def get_text_splitter():
    encoding = tiktoken.get_encoding("cl100k_base")

    def token_len(text: str) -> int:
        return len(encoding.encode(text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter


def chunk_articles(engine, splitter):
    with engine.connect() as conn:
        articles = conn.execute(text("""
            SELECT a.id, a.article_text, a.title, a.source_url
            FROM articles a
            LEFT JOIN chunks c ON a.id = c.article_id
            WHERE c.id IS NULL
              AND a.article_text IS NOT NULL
              AND LENGTH(a.article_text) > 100
        """)).fetchall()

    print(f"Chunking {len(articles):,} articles...")

    batch = []
    batch_size = 500
    total_chunks = 0

    for article in tqdm(articles):
        # 🔥 FIX: correct row access
        article_id = article[0]
        article_text = article[1]
        title = article[2]

        full_text = f"Title: {title}\n\n{article_text}"
        chunks = splitter.split_text(full_text)

        for i, chunk in enumerate(chunks):
            batch.append({
                "article_id": article_id,
                "chunk_index": i,
                "chunk_text": chunk
            })

        total_chunks += len(chunks)

        # 🔥 Batch insert
        if len(batch) >= batch_size:
            insert_chunks(engine, batch)
            batch = []

    # Insert remaining
    if batch:
        insert_chunks(engine, batch)

    print(f"\n✅ Total chunks created: {total_chunks:,}")
    print("Chunking complete.")


def insert_chunks(engine, records):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO chunks (article_id, chunk_index, chunk_text)
                VALUES (:article_id, :chunk_index, :chunk_text)
                ON CONFLICT DO NOTHING
            """),
            records
        )


def main():
    engine = create_engine(DB_URL)
    splitter = get_text_splitter()
    chunk_articles(engine, splitter)


if __name__ == "__main__":
    main()