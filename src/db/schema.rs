// Per-collection SQLite schema.
// Each collection gets its own .sqlite file — no collection column needed.

use crate::error::Result;
use rusqlite::Connection;

const SCHEMA_VERSION: i64 = 3;

pub fn init(conn: &Connection, collection_name: &str, has_preprocessor: bool) -> Result<()> {
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
    if current_version < 3 {
        migrate_to_v3(conn)?;
    }

    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?1)",
        [SCHEMA_VERSION.to_string()],
    )?;
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('collection', ?1)",
        [collection_name],
    )?;
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('has_preprocessor', ?1)",
        [if has_preprocessor { "1" } else { "0" }],
    )?;
    Ok(())
}

/// Backfill `document_metadata` from YAML frontmatter stored in `content.doc`.
/// Idempotent via `INSERT OR IGNORE`; safe to re-run.
fn migrate_v1_to_v2(conn: &Connection) -> Result<()> {
    // Collect active document IDs + content before opening the write transaction
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

    // document_metadata table already created by schema_base.sql (CREATE TABLE IF NOT EXISTS)
    let tx = conn.unchecked_transaction()?;
    for (doc_id, content) in &docs {
        let Some(mapping) = crate::frontmatter::extract(content) else {
            continue;
        };
        for (key, value) in crate::frontmatter::flatten(&mapping) {
            tx.execute(
                "INSERT OR IGNORE INTO document_metadata (document_id, key, value) \
                 VALUES (?1, ?2, ?3)",
                rusqlite::params![doc_id, key, value],
            )?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Backfill searchable units and links for existing documents. Safe to re-run.
fn migrate_to_v3(conn: &Connection) -> Result<()> {
    let docs: Vec<(i64, String, String, String)> = {
        let mut stmt = conn.prepare(
            "SELECT d.id, d.path, d.hash, c.doc \
             FROM documents d \
             JOIN content c ON d.hash = c.hash \
             WHERE d.active = 1",
        )?;
        stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?
    };

    let tx = conn.unchecked_transaction()?;
    for (doc_id, path, hash, content) in &docs {
        crate::index::units::store_units(&tx, *doc_id, path, hash, content)?;
    }
    tx.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn init_migrates_v2_db_to_v3_and_backfills_units_and_links() {
        crate::db::ensure_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(include_str!("schema_base.sql")).unwrap();
        conn.execute_batch(
            "DROP TABLE content_units;
             DROP TABLE unit_links;",
        )
        .unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '2')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO content (hash, doc, created_at) VALUES ('h1', ?1, '2026-01-01')",
            [r#"---
related:
  - cache-key
aliases:
  - cache policy
---
# Cache Policy

Use [cache-key] and [[Retry Budget]] as retrieval anchors.
"#],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO documents (id, path, title, hash, created_at, modified_at, active)
             VALUES (1, 'docs/cache.md', 'Cache Policy', 'h1', '2026-01-01', '2026-01-01', 1)",
            [],
        )
        .unwrap();

        init(&conn, "col", false).unwrap();

        let version: String = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(version, "3");

        let unit_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM content_units", [], |r| r.get(0))
            .unwrap();
        assert!(unit_count > 0);

        let targets: Vec<String> = {
            let mut stmt = conn
                .prepare("SELECT target FROM unit_links ORDER BY target")
                .unwrap();
            stmt.query_map([], |row| row.get::<_, String>(0))
                .unwrap()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap()
        };
        assert!(targets.contains(&"cache-key".to_string()));
        assert!(targets.contains(&"Retry Budget".to_string()));
        assert!(targets.contains(&"cache policy".to_string()));
    }
}
