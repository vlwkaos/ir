// Per-collection SQLite schema.
// Each collection gets its own .sqlite file — no collection column needed.

use crate::error::Result;
use rusqlite::Connection;

const SCHEMA_VERSION: i64 = 1;

pub fn init(conn: &Connection, collection_name: &str) -> Result<()> {
    conn.execute_batch(include_str!("schema.sql"))?;
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?1)",
        [SCHEMA_VERSION.to_string()],
    )?;
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('collection', ?1)",
        [collection_name],
    )?;
    Ok(())
}

pub fn get_version(conn: &Connection) -> Result<i64> {
    let version: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        )
        .ok();
    Ok(version.and_then(|v| v.parse().ok()).unwrap_or(0))
}
