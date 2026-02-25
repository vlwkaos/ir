-- Content-addressable storage (deduplication across files in this collection)
CREATE TABLE IF NOT EXISTS content (
    hash     TEXT PRIMARY KEY,
    doc      TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Document metadata; path is unique within the collection
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    hash        TEXT NOT NULL REFERENCES content(hash),
    created_at  TEXT NOT NULL,
    modified_at TEXT NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1
);

-- Full-text search index
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    path, title, body,
    tokenize='porter unicode61'
);

-- Sync FTS on insert
CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, path, title, body)
    SELECT new.id, new.path, new.title, c.doc
    FROM content c WHERE c.hash = new.hash;
END;

-- Sync FTS on delete
CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    DELETE FROM documents_fts WHERE rowid = old.id;
END;

-- Sync FTS on update
CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    DELETE FROM documents_fts WHERE rowid = old.id;
    INSERT INTO documents_fts(rowid, path, title, body)
    SELECT new.id, new.path, new.title, c.doc
    FROM content c WHERE c.hash = new.hash;
END;

-- Vector embeddings (sqlite-vec)
CREATE VIRTUAL TABLE IF NOT EXISTS vectors_vec USING vec0(
    hash_seq TEXT PRIMARY KEY,
    embedding float[768] distance_metric=cosine
);

-- Chunk-to-vector mapping
CREATE TABLE IF NOT EXISTS content_vectors (
    hash       TEXT NOT NULL,
    seq        INTEGER NOT NULL DEFAULT 0,
    pos        INTEGER NOT NULL DEFAULT 0,
    model      TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (hash, seq)
);

-- LLM response cache (reranking scores keyed by hash of query+doc)
CREATE TABLE IF NOT EXISTS llm_cache (
    hash       TEXT PRIMARY KEY,
    result     TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Collection metadata
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_documents_hash   ON documents (hash);
CREATE INDEX IF NOT EXISTS idx_documents_active ON documents (active);
CREATE INDEX IF NOT EXISTS idx_content_vectors_hash ON content_vectors (hash);
