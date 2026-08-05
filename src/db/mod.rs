// Per-collection SQLite connection management.
// sqlite-vec is registered as a static extension before any connection opens.
// docs: https://alexgarcia.xyz/sqlite-vec/rust.html

pub mod ann;
pub mod expander_cache;
pub mod fts;
pub mod graph;
pub mod schema;
pub mod vectors;

use crate::error::Result;
use crate::preprocess::PreprocessChain;
use rusqlite::ffi::{sqlite3, sqlite3_api_routines, sqlite3_auto_extension};
use rusqlite::{Connection, OpenFlags};
use std::cell::RefCell;
use std::collections::HashMap;
use std::os::raw::c_char;
use std::path::Path;
use std::sync::Once;

static SQLITE_VEC_INIT: Once = Once::new();

/// Register the sqlite-vec extension exactly once per process.
/// Safe to call from multiple threads or call sites.
pub fn ensure_sqlite_vec() {
    SQLITE_VEC_INIT.call_once(|| {
        // SAFETY: sqlite3_auto_extension is idempotent and the function pointer
        // is valid for the lifetime of the process. Once is used to prevent
        // double-registration.
        //
        // Transmute through the actual SQLite extension init ABI rather than
        // *const () — if the signature changes, mismatched sizes fail at compile time.
        type ExtInit = unsafe extern "C" fn(
            *mut sqlite3,
            *mut *mut c_char,
            *const sqlite3_api_routines,
        ) -> i32;
        unsafe {
            let fn_ptr = sqlite_vec::sqlite3_vec_init as unsafe extern "C" fn();
            let init: ExtInit = std::mem::transmute(fn_ptr);
            sqlite3_auto_extension(Some(init));
        }
    });
}

pub struct CollectionDb {
    pub name: String,
    /// Resolved preprocessor command strings for this collection (e.g. ["kiwi-tokenize", "mecab -Owakati"]).
    /// Empty if no preprocessing is configured. Used at query time to spawn a PreprocessChain.
    pub preprocessor_commands: Vec<String>,
    pub routing: Option<crate::types::RoutingConfig>,
    /// Per-collection retrieval-pipeline overrides (resolved into a RetrievalProfile
    /// at query time). None → env/default. See search::profile.
    pub retrieval: Option<crate::types::RetrievalConfig>,
    conn: Connection,
    /// Lazily spawned preprocessor chain, reused across BM25 calls within one search request.
    preprocess_chain: RefCell<Option<PreprocessChain>>,
}

impl CollectionDb {
    /// Open (or create) a collection DB at the given path with read-write access.
    /// `has_preprocessor` controls whether FTS triggers are installed in the schema.
    pub fn open(name: impl Into<String>, db_path: &Path, has_preprocessor: bool) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        ensure_sqlite_vec();

        let conn = Connection::open_with_flags(
            db_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )?;

        configure(&conn)?;

        let name = name.into();
        schema::init(&conn, &name, has_preprocessor)?;

        Ok(Self {
            name,
            preprocessor_commands: vec![],
            routing: None,
            retrieval: None,
            conn,
            preprocess_chain: RefCell::new(None),
        })
    }

    /// Open an existing collection DB with read-write access (no schema init).
    /// Use for search paths that need cache writes.
    /// `preprocessor_commands` are the resolved command strings for query-time preprocessing.
    pub fn open_rw(
        name: impl Into<String>,
        db_path: &Path,
        preprocessor_commands: Vec<String>,
        routing: Option<crate::types::RoutingConfig>,
        retrieval: Option<crate::types::RetrievalConfig>,
    ) -> Result<Self> {
        ensure_sqlite_vec();

        let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;

        configure(&conn)?;

        Ok(Self {
            name: name.into(),
            preprocessor_commands,
            routing,
            retrieval,
            conn,
            preprocess_chain: RefCell::new(None),
        })
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    pub fn active_doc_count(&self) -> usize {
        self.conn
            .query_row("SELECT COUNT(*) FROM documents WHERE active = 1", [], |r| {
                r.get::<_, usize>(0)
            })
            .unwrap_or(0)
    }

    /// Preprocess a query using the collection's configured preprocessor chain.
    /// Lazily spawns the chain on first call and reuses it for subsequent calls.
    /// Falls back to the raw query on spawn failure or I/O error.
    pub fn preprocess_query(&self, query: &str) -> String {
        if self.preprocessor_commands.is_empty() {
            return query.to_string();
        }
        let mut chain_ref = self.preprocess_chain.borrow_mut();
        let chain =
            chain_ref.get_or_insert_with(|| PreprocessChain::spawn(&self.preprocessor_commands));
        if !chain.is_active() {
            eprintln!(
                "warning: preprocessor ({}) failed to start; using raw query",
                self.preprocessor_commands.join(", ")
            );
            return query.to_string();
        }
        chain
            .process_text(query)
            .unwrap_or_else(|_| query.to_string())
    }
}

// ── content helpers ───────────────────────────────────────────────────────────

/// Batch-lookup document text by content hashes.
pub fn fetch_content_batch(conn: &Connection, hashes: &[&str]) -> HashMap<String, String> {
    if hashes.is_empty() {
        return HashMap::new();
    }
    let placeholders: Vec<&str> = hashes.iter().map(|_| "?").collect();
    let sql = format!(
        "SELECT hash, doc FROM content WHERE hash IN ({})",
        placeholders.join(",")
    );
    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(_) => return HashMap::new(),
    };
    let params: Vec<&dyn rusqlite::types::ToSql> = hashes
        .iter()
        .map(|h| h as &dyn rusqlite::types::ToSql)
        .collect();
    let rows = match stmt.query_map(params.as_slice(), |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }) {
        Ok(r) => r,
        Err(_) => return HashMap::new(),
    };
    let mut map = HashMap::new();
    for row in rows.flatten() {
        map.insert(row.0, row.1);
    }
    map
}

// ── llm_cache helpers ─────────────────────────────────────────────────────────

/// Batch-lookup reranker scores from llm_cache.
pub fn get_rerank_scores(conn: &Connection, keys: &[String]) -> HashMap<String, f64> {
    if keys.is_empty() {
        return HashMap::new();
    }
    let placeholders: Vec<&str> = keys.iter().map(|_| "?").collect();
    let sql = format!(
        "SELECT hash, result FROM llm_cache WHERE hash IN ({})",
        placeholders.join(",")
    );
    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(_) => return HashMap::new(),
    };
    let params: Vec<&dyn rusqlite::types::ToSql> = keys
        .iter()
        .map(|k| k as &dyn rusqlite::types::ToSql)
        .collect();
    let rows = match stmt.query_map(params.as_slice(), |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }) {
        Ok(r) => r,
        Err(_) => return HashMap::new(),
    };
    let mut map = HashMap::new();
    for row in rows.flatten() {
        if let Ok(score) = row.1.parse::<f64>() {
            map.insert(row.0, score);
        }
    }
    map
}

/// Batch-insert reranker scores into llm_cache.
pub fn put_rerank_scores(conn: &Connection, entries: &[(String, f64)]) {
    if entries.is_empty() {
        return;
    }
    let now = chrono::Utc::now().to_rfc3339();
    let tx = match conn.unchecked_transaction() {
        Ok(t) => t,
        Err(_) => return,
    };
    {
        let mut stmt = match tx.prepare(
            "INSERT OR REPLACE INTO llm_cache (hash, result, created_at) VALUES (?1, ?2, ?3)",
        ) {
            Ok(s) => s,
            Err(_) => return,
        };
        for (hash, score) in entries {
            let _ = stmt.execute(rusqlite::params![hash, score.to_string(), now]);
        }
    }
    let _ = tx.commit();
}

// ── vector dimension helpers ──────────────────────────────────────────────────

/// Read the current embedding dimension from the vectors_vec virtual table DDL.
fn current_vector_dim(conn: &Connection) -> Option<usize> {
    let ddl: String = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vectors_vec'",
            [],
            |row| row.get(0),
        )
        .ok()?;
    let marker = "float[";
    let start = ddl.find(marker)? + marker.len();
    let rest = &ddl[start..];
    let end = rest.find(']')?;
    rest[..end].trim().parse::<usize>().ok()
}

/// Ensure vectors_vec matches the given dimension. Rebuilds the table on mismatch.
pub fn ensure_vector_dimension(conn: &Connection, dim: usize) -> Result<()> {
    if dim == 0 {
        return Err(crate::error::Error::Other(
            "embedding dimension resolved to 0".into(),
        ));
    }
    match current_vector_dim(conn) {
        Some(existing) if existing == dim => return Ok(()),
        Some(existing) => {
            eprintln!("  vector dimension mismatch ({existing} -> {dim}), rebuilding vector table");
            conn.execute_batch(
                "DROP TABLE IF EXISTS vectors_vec;
                 DELETE FROM content_vectors;",
            )?;
        }
        None => {
            conn.execute_batch("DROP TABLE IF EXISTS vectors_vec;")?;
        }
    }
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vectors_vec USING vec0(
            hash_seq TEXT PRIMARY KEY,
            embedding float[{dim}] distance_metric=cosine
        );"
    ))?;
    Ok(())
}

fn configure(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA busy_timeout = 5000;
         PRAGMA synchronous  = NORMAL;
         PRAGMA cache_size   = -64000;
         PRAGMA foreign_keys = ON;",
    )?;
    Ok(())
}
