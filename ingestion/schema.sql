CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id TEXT NOT NULL,
    page_title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    connector_type TEXT NOT NULL DEFAULT 'notion',
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT,
    embedding VECTOR(1536),
    last_updated TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- For pre-existing tables created before content_hash was added
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT;

CREATE INDEX IF NOT EXISTS documents_page_hash_idx ON documents (page_id, content_hash);

-- Cosine similarity index — tune lists= to ~sqrt(row count) once you have data
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS documents_page_id_idx ON documents (page_id);
CREATE INDEX IF NOT EXISTS documents_connector_idx ON documents (connector_type);
