// Global expander output cache — stores Vec<SubQuery> keyed by sha256(model_id + "\0" + query).
// Separate SQLite file so it persists across collection changes and daemon restarts.
// Silent-failure design: any cache error degrades to no-cache, search still works.

use crate::config;
use crate::index::hasher;
use crate::llm::expander::SubQuery;
use rusqlite::{Connection, OpenFlags};

pub struct ExpanderCache {
    conn: Connection,
}

impl ExpanderCache {
    pub fn open() -> crate::error::Result<Self> {
        let path = config::expander_cache_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open_with_flags(
            &path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous  = NORMAL;
             PRAGMA cache_size   = -8000;
             CREATE TABLE IF NOT EXISTS expander_cache (
                 hash       TEXT PRIMARY KEY,
                 result     TEXT NOT NULL,
                 created_at TEXT NOT NULL
             );",
        )?;
        Ok(Self { conn })
    }

    pub fn get(&self, model_id: &str, query: &str) -> Option<Vec<SubQuery>> {
        let key = cache_key(model_id, query);
        let result: Option<String> = self
            .conn
            .query_row(
                "SELECT result FROM expander_cache WHERE hash = ?1",
                [&key],
                |row| row.get(0),
            )
            .ok();
        result.and_then(|json| serde_json::from_str(&json).ok())
    }

    pub fn put(&self, model_id: &str, query: &str, subs: &[SubQuery]) {
        let key = cache_key(model_id, query);
        let Ok(json) = serde_json::to_string(subs) else {
            return;
        };
        let now = chrono::Utc::now().to_rfc3339();
        let _ = self.conn.execute(
            "INSERT OR REPLACE INTO expander_cache (hash, result, created_at) VALUES (?1, ?2, ?3)",
            rusqlite::params![key, json, now],
        );
    }
}

fn cache_key(model_id: &str, query: &str) -> String {
    let q = query.trim().to_lowercase();
    hasher::hash_bytes(format!("{model_id}\0{q}").as_bytes())
}
