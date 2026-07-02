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

-- Indexes
CREATE INDEX IF NOT EXISTS idx_documents_hash   ON documents (hash);
CREATE INDEX IF NOT EXISTS idx_documents_active ON documents (active);
CREATE INDEX IF NOT EXISTS idx_content_vectors_hash ON content_vectors (hash);
CREATE INDEX IF NOT EXISTS idx_document_metadata_key_value ON document_metadata (key, value);

-- Searchable units extracted from a document. Markdown documents use the existing chunker;
-- code documents use lightweight symbol/range extraction. Line ranges are presentation hints:
-- durable retrieval uses hash + seq + stored text.
CREATE TABLE IF NOT EXISTS content_units (
    hash        TEXT NOT NULL REFERENCES content(hash) ON DELETE CASCADE,
    seq         INTEGER NOT NULL DEFAULT 0,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    unit_kind   TEXT NOT NULL,
    language    TEXT,
    symbol      TEXT,
    start_byte  INTEGER NOT NULL DEFAULT 0,
    end_byte    INTEGER NOT NULL DEFAULT 0,
    start_line  INTEGER NOT NULL DEFAULT 1,
    end_line    INTEGER NOT NULL DEFAULT 1,
    title       TEXT NOT NULL,
    text        TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    indexed_at  TEXT NOT NULL,
    PRIMARY KEY (hash, seq)
);

-- Explicit parseable links emitted from Markdown/prose/comments/frontmatter.
-- Targets may remain unresolved; search expansion uses exact target matches first.
CREATE TABLE IF NOT EXISTS unit_links (
    source_hash          TEXT NOT NULL,
    source_seq           INTEGER NOT NULL DEFAULT 0,
    document_id          INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    kind                 TEXT NOT NULL,
    target               TEXT NOT NULL,
    raw                  TEXT NOT NULL,
    resolved_document_id INTEGER,
    resolved_hash        TEXT,
    resolved_seq         INTEGER,
    PRIMARY KEY (source_hash, source_seq, kind, target, raw)
);

CREATE INDEX IF NOT EXISTS idx_content_units_document ON content_units (document_id);
CREATE INDEX IF NOT EXISTS idx_content_units_symbol ON content_units (symbol);
CREATE INDEX IF NOT EXISTS idx_unit_links_target ON unit_links (target);
CREATE INDEX IF NOT EXISTS idx_unit_links_source ON unit_links (source_hash, source_seq);
