// Per-collection SQLite connection management.
// sqlite-vec is registered as a static extension before any connection opens.
// docs: https://alexgarcia.xyz/sqlite-vec/rust.html

pub mod expander_cache;
pub mod fts;
pub mod schema;
pub mod vectors;

use crate::error::Result;
use rusqlite::ffi::{sqlite3, sqlite3_api_routines, sqlite3_auto_extension};
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
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
            *mut *mut i8,
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
    conn: Connection,
}

impl CollectionDb {
    /// Open (or create) a collection DB at the given path with read-write access.
    pub fn open(name: impl Into<String>, db_path: &Path) -> Result<Self> {
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
        schema::init(&conn, &name)?;

        Ok(Self { name, conn })
    }

    /// Open an existing collection DB with read-write access (no schema init).
    /// Use for search paths that need cache writes.
    pub fn open_rw(name: impl Into<String>, db_path: &Path) -> Result<Self> {
        ensure_sqlite_vec();

        let conn = Connection::open_with_flags(
            db_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE,
        )?;

        configure(&conn)?;

        Ok(Self {
            name: name.into(),
            conn,
        })
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }
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
    let params: Vec<&dyn rusqlite::types::ToSql> =
        keys.iter().map(|k| k as &dyn rusqlite::types::ToSql).collect();
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
