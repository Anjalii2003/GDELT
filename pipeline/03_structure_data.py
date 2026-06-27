# # pipeline/03_structure_data.py
# """
# Phase 3: Combine GDELT metadata with fetched article text, store in PostgreSQL.
# """

# import sys
# import os

# # 🔥 Fix import path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import json
# import pandas as pd
# from sqlalchemy import create_engine, text
# from tqdm import tqdm
# from config import DB_URL, ARTICLES_DIR

# def url_to_filename(url: str) -> str:
#     import hashlib
#     return hashlib.md5(url.encode()).hexdigest() + ".json"

# def load_article_json(url: str) -> dict | None:
#     path = os.path.join(ARTICLES_DIR, url_to_filename(url))
#     if not os.path.exists(path):
#         return None
#     with open(path) as f:
#         return json.load(f)

# def build_record(row: pd.Series, article: dict) -> dict:
#     """Combine one GDELT row + article into a database record."""
#     return {
#         "global_event_id": int(row.GlobalEventID) if pd.notna(row.GlobalEventID) else None,
#         "source_url":      row.SOURCEURL,
#         "date_str":        row.date_str,
#         "actor1":          row.Actor1Name,
#         "actor2":          row.Actor2Name,
#         "event_code":      str(row.EventCode) if pd.notna(row.EventCode) else None,
#         "event_type":      row.event_type_label,
#         "goldstein_scale": float(row.GoldsteinScale) if pd.notna(row.GoldsteinScale) else None,
#         "avg_tone":        float(row.AvgTone) if pd.notna(row.AvgTone) else None,
#         "location":        row.ActionGeo_FullName if pd.notna(row.ActionGeo_FullName) else None,
#         "latitude":        float(row.ActionGeo_Lat) if pd.notna(row.ActionGeo_Lat) else None,
#         "longitude":       float(row.ActionGeo_Long) if pd.notna(row.ActionGeo_Long) else None,
#         "title":           article.get("title", ""),
#         "article_text":    article.get("text", ""),
#         "publish_date":    article.get("publish_date"),
#     }

# def insert_records(records: list[dict], engine):
#     insert_sql = text("""
#         INSERT INTO articles
#             (global_event_id, source_url, date_str, actor1, actor2,
#              event_code, event_type, goldstein_scale, avg_tone,
#              location, latitude, longitude, title, article_text, publish_date)
#         VALUES
#             (:global_event_id, :source_url, :date_str, :actor1, :actor2,
#              :event_code, :event_type, :goldstein_scale, :avg_tone,
#              :location, :latitude, :longitude, :title, :article_text, :publish_date)
#         ON CONFLICT (source_url) DO NOTHING
#     """)
#     with engine.begin() as conn:
#         conn.execute(insert_sql, records)

# def main():
#     df = pd.read_parquet("data/raw/gdelt_filtered.parquet")
#     engine = create_engine(DB_URL)

#     batch, batch_size = [], 100
#     missing = 0

#     for _, row in tqdm(df.iterrows(), total=len(df)):
#         article = load_article_json(row.SOURCEURL)
#         if not article or not article.get("text"):
#             missing += 1
#             continue

#         record = build_record(row, article)
#         batch.append(record)

#         if len(batch) >= batch_size:
#             insert_records(batch, engine)
#             batch = []

#     if batch:
#         insert_records(batch, engine)

#     print(f"Done. Inserted records. Missing articles: {missing:,}")

# if __name__ == "__main__":
#     main()

# pipeline/03_structure_data.py

import sys
import os

# 🔥 Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm
from config import DB_URL, ARTICLES_DIR


def url_to_filename(url: str) -> str:
    import hashlib
    return hashlib.md5(url.encode()).hexdigest() + ".json"


def load_article_json(url: str) -> dict | None:
    path = os.path.join(ARTICLES_DIR, url_to_filename(url))
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_record(row: pd.Series, article: dict) -> dict:
    return {
        "global_event_id": int(row.GlobalEventID) if pd.notna(row.GlobalEventID) else None,
        "source_url": row.SOURCEURL,
        "date_str": row.date_str,
        "actor1": row.Actor1Name,
        "actor2": row.Actor2Name,
        "event_code": str(row.EventCode) if pd.notna(row.EventCode) else None,
        "event_type": row.event_type_label,
        "goldstein_scale": float(row.GoldsteinScale) if pd.notna(row.GoldsteinScale) else None,
        "avg_tone": float(row.AvgTone) if pd.notna(row.AvgTone) else None,
        "location": row.ActionGeo_FullName if pd.notna(row.ActionGeo_FullName) else None,
        "latitude": float(row.ActionGeo_Lat) if pd.notna(row.ActionGeo_Lat) else None,
        "longitude": float(row.ActionGeo_Long) if pd.notna(row.ActionGeo_Long) else None,
        "title": article.get("title", ""),
        "article_text": article.get("text", ""),
        "publish_date": article.get("publish_date"),
    }


def insert_records(records: list[dict], engine):
    insert_sql = text("""
        INSERT INTO articles
            (global_event_id, source_url, date_str, actor1, actor2,
             event_code, event_type, goldstein_scale, avg_tone,
             location, latitude, longitude, title, article_text, publish_date)
        VALUES
            (:global_event_id, :source_url, :date_str, :actor1, :actor2,
             :event_code, :event_type, :goldstein_scale, :avg_tone,
             :location, :latitude, :longitude, :title, :article_text, :publish_date)
        ON CONFLICT (source_url) DO NOTHING
    """)
    with engine.begin() as conn:
        conn.execute(insert_sql, records)


def get_fetched_urls():
    """🔥 Get URLs from downloaded JSON files (FAST)"""
    files = os.listdir(ARTICLES_DIR)
    return set(f.replace(".json", "") for f in files)


def main():
    df = pd.read_parquet("data/raw/gdelt_filtered.parquet")
    engine = create_engine(DB_URL)

    print(f"Loaded {len(df):,} rows")

    # 🔥 Get only fetched articles
    fetched_files = os.listdir(ARTICLES_DIR)
    print(f"Found {len(fetched_files):,} downloaded articles")

    # 🔥 Filter dataframe to only fetched URLs
    df["file_hash"] = df["SOURCEURL"].apply(
        lambda x: url_to_filename(x).replace(".json", "")
    )

    df = df[df["file_hash"].isin(set(f.replace(".json", "") for f in fetched_files))]

    print(f"Filtered to {len(df):,} rows with articles")

    batch, batch_size = [], 100
    inserted = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        article = load_article_json(row.SOURCEURL)
        if not article or not article.get("text"):
            continue

        record = build_record(row, article)
        batch.append(record)

        if len(batch) >= batch_size:
            insert_records(batch, engine)
            inserted += len(batch)
            batch = []

    if batch:
        insert_records(batch, engine)
        inserted += len(batch)

    print(f"\n✅ Inserted records: {inserted:,}")


if __name__ == "__main__":
    main()