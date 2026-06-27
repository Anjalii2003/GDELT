-- db/schema.sql
-- Auto-executed by PostgreSQL on first container start
-- Sets up all tables, indexes, and extensions

-- Enable pgvector for similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Articles table ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS articles (
    id              SERIAL PRIMARY KEY,
    global_event_id BIGINT,
    source_url      TEXT UNIQUE NOT NULL,
    date_str        DATE,
    actor1          TEXT,
    actor2          TEXT,
    event_code      TEXT,
    event_type      TEXT,
    goldstein_scale FLOAT,
    avg_tone        FLOAT,
    location        TEXT,
    latitude        FLOAT,
    longitude       FLOAT,
    title           TEXT,
    article_text    TEXT,
    publish_date    TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Indexes for fast metadata filtering
CREATE INDEX IF NOT EXISTS idx_articles_date      ON articles(date_str);
CREATE INDEX IF NOT EXISTS idx_articles_location  ON articles(location);
CREATE INDEX IF NOT EXISTS idx_articles_actor1    ON articles(actor1);
CREATE INDEX IF NOT EXISTS idx_articles_actor2    ON articles(actor2);
CREATE INDEX IF NOT EXISTS idx_articles_event     ON articles(event_type);
CREATE INDEX IF NOT EXISTS idx_articles_tone      ON articles(avg_tone);
CREATE INDEX IF NOT EXISTS idx_articles_goldstein ON articles(goldstein_scale);

-- Full-text search index (useful for keyword fallback)
ALTER TABLE articles ADD COLUMN IF NOT EXISTS
    search_vector tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english',
            COALESCE(title, '') || ' ' ||
            COALESCE(actor1, '') || ' ' ||
            COALESCE(actor2, '') || ' ' ||
            COALESCE(location, '') || ' ' ||
            COALESCE(article_text, '')
        )
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_articles_fts ON articles USING GIN(search_vector);

-- ── Chunks table ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    article_id  INT REFERENCES articles(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text  TEXT NOT NULL,
    embedding   vector(1024),     -- BGE-M3 produces 1024-dimensional vectors
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_article  ON chunks(article_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);           -- pgvector ANN index (faster than exact search)

-- ── Fetch log table (track article fetch status) ───────────────────────────
CREATE TABLE IF NOT EXISTS fetch_log (
    id          SERIAL PRIMARY KEY,
    source_url  TEXT UNIQUE NOT NULL,
    status      TEXT NOT NULL,    -- 'success', 'failed', 'skipped'
    error_msg   TEXT,
    fetched_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fetch_log_status ON fetch_log(status);

-- ── Done ───────────────────────────────────────────────────────────────────
SELECT 'Schema initialized successfully' AS status;