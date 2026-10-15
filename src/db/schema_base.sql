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

-- Full-text search index (porter unicode61: English stemming + Unicode normalization)
-- FTS tokenizer is always porter unicode61 regardless of preprocessor.
-- Preprocessors handle CJK morphology before this stage; porter handles English stemming.
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    path, title, body,
    tokenize='porter unicode61'
);

-- Content whose search text is raw because the preprocessor failed while it
-- was indexed. The next update stores those documents again.
CREATE TABLE IF NOT EXISTS fts_raw (
    hash TEXT PRIMARY KEY
);

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

-- Frontmatter key-value metadata extracted at index time.
-- One row per (document, key, value); multi-valued fields (e.g. tags) have one row per element.
-- ON DELETE CASCADE: rows are removed automatically when the parent document is deleted.
CREATE TABLE IF NOT EXISTS document_metadata (
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    PRIMARY KEY (document_id, key, value)
);

-- kNN document similarity graph (research: built via IR_GRAPH_BUILD=1 during `ir embed`).
-- Doc-level edges; weight = max cosine similarity across the two docs' chunk embeddings.
CREATE TABLE IF NOT EXISTS doc_graph (
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    neighbor_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    weight      REAL NOT NULL,
    PRIMARY KEY (doc_id, neighbor_id)
);

-- ANN sidecar key mapping (research: IR_ANN=hnsw). usearch keys are u64;
-- this table allocates stable integer keys per embedded chunk (hash_seq).
CREATE TABLE IF NOT EXISTS ann_keys (
    key      INTEGER PRIMARY KEY AUTOINCREMENT,
    hash_seq TEXT NOT NULL UNIQUE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_documents_hash   ON documents (hash);
CREATE INDEX IF NOT EXISTS idx_documents_active ON documents (active);
CREATE INDEX IF NOT EXISTS idx_content_vectors_hash ON content_vectors (hash);
CREATE INDEX IF NOT EXISTS idx_document_metadata_key_value ON document_metadata (key, value);
