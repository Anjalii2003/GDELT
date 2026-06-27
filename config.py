# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
GDELT_FOLDER  = os.getenv("GDELT_FOLDER", "data/raw/2025")
ARTICLES_DIR  = "data/articles/"
CHUNKS_DIR    = "data/chunks/"
FAISS_INDEX   = "db/vector_store/faiss.index"

# ── Database ──────────────────────────────────────────────────────────────────
DB_URL = os.getenv("DATABASE_URL", "postgresql://gdelt:gdelt_pass@localhost:5432/gdelt_rag")

# ── Ollama ────────────────────────────────────────────────────────────────────
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8002/v1")
LLM_MODEL    = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# ── Models ────────────────────────────────────────────────────────────────────
EMBED_MODEL  = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # tokens
CHUNK_OVERLAP = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_VECTOR = 20    # candidates from FAISS
TOP_K_RERANK = 5     # final chunks after reranking

# ── Article fetching ──────────────────────────────────────────────────────────
FETCH_TIMEOUT = 10   # seconds
MAX_RETRIES   = 3