// Per-collection SQLite schema.
// Each collection gets its own .sqlite file — no collection column needed.

use crate::error::Result;
use rusqlite::Connection;

const SCHEMA_VERSION: i64 = 2;

pub fn init(conn: &Connection, collection_name: &str, has_preprocessor: bool) -> Result<()> {
    // ^ One transaction: processes opening a new collection at the same time
    //   otherwise interleave DDL and fail with "no such table".
    super::write_txn(conn, || {
        init_locked(conn, collection_name, has_preprocessor)
    })
}

fn init_locked(conn: &Connection, collection_name: &str, has_preprocessor: bool) -> Result<()> {
    conn.execute_batch(include_str!("schema_base.sql"))?;

    if has_preprocessor {
        // Drop any pre-existing triggers; FTS is managed explicitly by the index pipeline.
        conn.execute_batch(
            "DROP TRIGGER IF EXISTS documents_ai;
             DROP TRIGGER IF EXISTS documents_ad;
             DROP TRIGGER IF EXISTS documents_au;",
        )?;
    } else {
        conn.execute_batch(include_str!("schema_triggers.sql"))?;
    }

    // Bootstrap version to 0 for brand-new DBs (so migration check is uniform)
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '0')",
        [],
    )?;

    let current_version: i64 = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |row| row.get::<_, String>(0),
        )
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    if current_version < 2 {
        migrate_v1_to_v2(conn)?;
    }

    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?1)",
        [SCHEMA_VERSION.to_string()],
    )?;
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('collection', ?1)",
        [collection_name],
    )?;
    Ok(())
}

/// Backfill `document_metadata` from YAML frontmatter stored in `content.doc`.
/// Idempotent via `INSERT OR IGNORE`; safe to re-run.
fn migrate_v1_to_v2(conn: &Connection) -> Result<()> {
    // Collect active document IDs + content before inserting
    // (rusqlite can't run a SELECT iterator and INSERT statements on the same connection simultaneously).
    let docs: Vec<(i64, String)> = {
        let mut stmt = conn.prepare(
            "SELECT d.id, c.doc \
             FROM documents d \
             JOIN content c ON d.hash = c.hash \
             WHERE d.active = 1",
        )?;
        stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?
    };

    // document_metadata table already created by schema_base.sql (CREATE TABLE IF NOT EXISTS).
    // Runs inside init's transaction.
    for (doc_id, content) in &docs {
        let Some(mapping) = crate::frontmatter::extract(content) else {
            continue;
        };
        for (key, value) in crate::frontmatter::flatten(&mapping) {
            conn.execute(
                "INSERT OR IGNORE INTO document_metadata (document_id, key, value) \
                 VALUES (?1, ?2, ?3)",
                rusqlite::params![doc_id, key, value],
            )?;
        }
    }
    Ok(())
}
