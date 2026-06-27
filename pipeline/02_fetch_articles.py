# pipeline/02_fetch_articles.py

import sys
import os

# 🔥 Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import hashlib
import pandas as pd
import requests
from newspaper import Article
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import ARTICLES_DIR, FETCH_TIMEOUT, MAX_RETRIES

os.makedirs(ARTICLES_DIR, exist_ok=True)

MAX_WORKERS = 10   # 🔥 parallel threads (adjust if needed)

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((requests.RequestException, Exception)),
    reraise=False
)
def fetch_article(url: str) -> dict | None:
    try:
        article = Article(url)
        article.download()
        article.parse()

        if not article.text or len(article.text.strip()) < 100:
            return None

        return {
            "title": article.title or "",
            "text": article.text.strip(),
            "publish_date": str(article.publish_date) if article.publish_date else None,
            "authors": article.authors,
            "top_image": article.top_image,
        }

    except Exception:
        return None


def url_to_filename(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest() + ".json"


def article_exists(url: str) -> bool:
    filename = url_to_filename(url)
    return os.path.exists(os.path.join(ARTICLES_DIR, filename))


def process_url(url):
    if article_exists(url):
        return "skipped", url

    result = fetch_article(url)

    if result is None:
        return "failed", url

    filename = url_to_filename(url)
    result["source_url"] = url

    with open(os.path.join(ARTICLES_DIR, filename), "w") as f:
        json.dump(result, f, ensure_ascii=False)

    return "success", url


def fetch_all_articles(df: pd.DataFrame, limit: int = 1000):
    urls = df["SOURCEURL"].dropna().unique()

    # 🔥 LIMIT applied
    urls = urls[:limit]

    print(f"Fetching {len(urls):,} articles using {MAX_WORKERS} threads...")

    success, failed, skipped = 0, 0, 0
    failed_urls = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_url, url): url for url in urls}

        for future in tqdm(as_completed(futures), total=len(futures)):
            status, url = future.result()

            if status == "success":
                success += 1
            elif status == "failed":
                failed += 1
                failed_urls.append(url)
            else:
                skipped += 1

    print(f"\nDone.")
    print(f"Success: {success:,}")
    print(f"Failed: {failed:,}")
    print(f"Skipped: {skipped:,}")

    if failed_urls:
        with open("data/raw/failed_urls.txt", "w") as f:
            f.write("\n".join(failed_urls))


def main():
    df = pd.read_parquet("data/raw/gdelt_filtered.parquet")

    # 🔥 SMART FILTER (recommended)
    df = df.sort_values("date_str", ascending=False)

    # 🔥 Run with limit
    fetch_all_articles(df, limit=1000)


if __name__ == "__main__":
    main()