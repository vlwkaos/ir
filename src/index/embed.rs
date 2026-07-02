// Embedding pipeline: load un-embedded documents → chunk → embed → store.
// Runs after `ir update`. Safe to re-run: skips already-embedded hashes.

use crate::db::{self, CollectionDb, vectors};
use crate::error::Result;
use crate::index::new_progress_bar;
use crate::llm::embedding::Embedder;
use rusqlite::{Connection, params};

const EMBED_BATCH_SIZE: usize = 256;

type PendingDoc = (String, String, String, String);

pub struct EmbedOptions {
    /// Re-embed even chunks that already have embeddings.
    pub force: bool,
}

/// Embed all un-embedded document chunks in the collection.
/// Returns (embedded_docs, total_chunks) counts.
pub fn embed(
    db: &CollectionDb,
    embedder: &Embedder,
    opts: &EmbedOptions,
    model_name: &str,
) -> Result<(usize, usize)> {
    let conn = db.conn();

    // Adapt vector table dimension to match the loaded model.
    db::ensure_vector_dimension(conn, embedder.embedding_dim())?;

    // Clean up embeddings for hashes no longer referenced by any active document.
    cleanup_orphaned(conn)?;

    // Find documents whose content is not yet embedded (or force all).
    let pending_count = count_pending(conn, opts.force)?;

    if pending_count == 0 {
        return Ok((0, 0));
    }

    let pb = new_progress_bar(pending_count as u64);
    pb.println(format!("embedding {pending_count} pending documents"));
    pb.tick();

    let mut total_chunks = 0usize;
    let mut total_docs = 0usize;
    let mut cursor: Option<String> = None;

    loop {
        let pending = load_pending_batch(conn, opts.force, cursor.as_deref(), EMBED_BATCH_SIZE)?;
        if pending.is_empty() {
            break;
        }

        for (path, _title, hash, _doc_text) in &pending {
            pb.set_message(path.clone());

            // Compute embeddings before touching the DB.
            let units = load_units_for_hash(conn, hash)?;
            let inputs: Vec<(String, String)> = units
                .iter()
                .map(|u| (u.title.clone(), u.text.clone()))
                .collect();
            let embeddings = embedder.embed_doc_batch(&inputs)?;

            // Write atomically: a crash mid-insert would leave no partial state.
            conn.execute_batch("BEGIN IMMEDIATE")?;
            let write = || -> Result<()> {
                if opts.force {
                    // Collect seqs before deleting content_vectors (sqlite-vec can't LIKE on PK).
                    let seqs: Vec<i64> = {
                        let mut stmt =
                            conn.prepare("SELECT seq FROM content_vectors WHERE hash = ?1")?;
                        stmt.query_map([hash], |r| r.get(0))?
                            .filter_map(|r| r.ok())
                            .collect()
                    };
                    if !seqs.is_empty() {
                        let hash_seqs: Vec<String> =
                            seqs.iter().map(|s| format!("{hash}_{s}")).collect();
                        let ph = hash_seqs.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                        conn.execute(
                            &format!("DELETE FROM vectors_vec WHERE hash_seq IN ({ph})"),
                            rusqlite::params_from_iter(hash_seqs.iter().map(|s| s.as_str())),
                        )?;
                    }
                    conn.execute("DELETE FROM content_vectors WHERE hash = ?1", [hash])?;
                }
                for (unit, emb) in units.iter().zip(embeddings.iter()) {
                    let hash_seq = format!("{hash}_{}", unit.seq);
                    vectors::insert(conn, &hash_seq, emb)?;
                    vectors::mark_embedded(
                        conn,
                        hash,
                        unit.seq as i64,
                        unit.start_byte as i64,
                        model_name,
                    )?;
                }
                Ok(())
            };
            match write() {
                Ok(()) => conn.execute_batch("COMMIT")?,
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    return Err(e);
                }
            }

            total_chunks += units.len();
            total_docs += 1;
            pb.inc(1);
        }

        cursor = pending.last().map(|(_, _, hash, _)| hash.clone());
    }

    pb.finish_with_message("done");
    Ok((total_docs, total_chunks))
}

#[derive(Debug)]
struct PendingUnit {
    seq: usize,
    start_byte: usize,
    title: String,
    text: String,
}

fn load_units_for_hash(conn: &Connection, hash: &str) -> Result<Vec<PendingUnit>> {
    let mut stmt = conn.prepare(
        "SELECT seq, start_byte, title, text
         FROM content_units
         WHERE hash = ?1
         ORDER BY seq",
    )?;
    let rows = stmt.query_map([hash], |row| {
        Ok(PendingUnit {
            seq: row.get::<_, i64>(0)? as usize,
            start_byte: row.get::<_, i64>(1)? as usize,
            title: row.get(2)?,
            text: row.get(3)?,
        })
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn count_pending(conn: &Connection, force: bool) -> Result<usize> {
    let sql = if force {
        "
        SELECT COUNT(DISTINCT d.hash)
        FROM documents d
        WHERE d.active = 1
        "
    } else {
        "
        SELECT COUNT(DISTINCT d.hash)
        FROM documents d
        WHERE d.active = 1
          AND NOT EXISTS (
              SELECT 1 FROM content_vectors cv WHERE cv.hash = d.hash
          )
        "
    };
    conn.query_row(sql, [], |row| row.get(0))
        .map_err(Into::into)
}

/// Load one batch of active content-addressed docs for embedding.
fn load_pending_batch(
    conn: &Connection,
    force: bool,
    after_hash: Option<&str>,
    limit: usize,
) -> Result<Vec<PendingDoc>> {
    let sql = if force {
        "
        SELECT MIN(d.path), MIN(d.title), d.hash, c.doc
        FROM documents d
        JOIN content c ON c.hash = d.hash
        WHERE d.active = 1
          AND (?1 IS NULL OR d.hash > ?1)
        GROUP BY d.hash, c.doc
        ORDER BY d.hash
        LIMIT ?2
        "
    } else {
        "
        SELECT MIN(d.path), MIN(d.title), d.hash, c.doc
        FROM documents d
        JOIN content c ON c.hash = d.hash
        WHERE d.active = 1
          AND NOT EXISTS (
              SELECT 1 FROM content_vectors cv WHERE cv.hash = d.hash
          )
          AND (?1 IS NULL OR d.hash > ?1)
        GROUP BY d.hash, c.doc
        ORDER BY d.hash
        LIMIT ?2
        "
    };
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map(params![after_hash, limit as i64], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

/// Remove content_vectors (and their vectors) for hashes no longer in active documents.
fn cleanup_orphaned(conn: &Connection) -> Result<()> {
    let orphaned: Vec<String> = {
        let sql = "
            SELECT DISTINCT cv.hash
            FROM content_vectors cv
            WHERE NOT EXISTS (
                SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1
            )
        ";
        let mut stmt = conn.prepare(sql)?;
        stmt.query_map([], |r| r.get(0))?
            .filter_map(|r| r.ok())
            .collect()
    };

    if orphaned.is_empty() {
        return Ok(());
    }

    // Collect all hash_seqs to delete in one query.
    let hash_seqs: Vec<String> = {
        let ph = orphaned.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!("SELECT hash, seq FROM content_vectors WHERE hash IN ({ph})");
        let mut stmt = conn.prepare(&sql)?;
        stmt.query_map(
            rusqlite::params_from_iter(orphaned.iter().map(|s| s.as_str())),
            |r| {
                Ok(format!(
                    "{}_{}",
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?
                ))
            },
        )?
        .filter_map(|r| r.ok())
        .collect()
    };

    if !hash_seqs.is_empty() {
        let ph = hash_seqs.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        conn.execute(
            &format!("DELETE FROM vectors_vec WHERE hash_seq IN ({ph})"),
            rusqlite::params_from_iter(hash_seqs.iter().map(|s| s.as_str())),
        )?;
    }

    let ph = orphaned.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    conn.execute(
        &format!("DELETE FROM content_vectors WHERE hash IN ({ph})"),
        rusqlite::params_from_iter(orphaned.iter().map(|s| s.as_str())),
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn open_test_db() -> Connection {
        crate::db::ensure_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(include_str!("../db/schema_base.sql"))
            .unwrap();
        conn.execute_batch(include_str!("../db/schema_triggers.sql"))
            .unwrap();
        conn
    }

    #[test]
    fn load_pending_batch_finds_pending() {
        let conn = open_test_db();
        let hash = "abc";
        conn.execute(
            "INSERT INTO content (hash, doc, created_at) VALUES (?1,'hello','2024-01-01')",
            [hash],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO documents (path,title,hash,created_at,modified_at,active)
             VALUES ('a.md','A',?1,'2024-01-01','2024-01-01',1)",
            [hash],
        )
        .unwrap();

        let pending = load_pending_batch(&conn, false, None, 10).unwrap();
        assert_eq!(pending.len(), 1, "should find 1 unembedded doc");
        assert_eq!(pending[0].2, hash);
        assert_eq!(count_pending(&conn, false).unwrap(), 1);
    }

    #[test]
    fn load_pending_batch_skips_already_embedded() {
        let conn = open_test_db();
        let hash = "def";
        conn.execute(
            "INSERT INTO content (hash, doc, created_at) VALUES (?1,'world','2024-01-01')",
            [hash],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO documents (path,title,hash,created_at,modified_at,active)
             VALUES ('b.md','B',?1,'2024-01-01','2024-01-01',1)",
            [hash],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO content_vectors (hash,seq,pos,model,embedded_at)
             VALUES (?1,0,0,'test','2024-01-01')",
            [hash],
        )
        .unwrap();

        let pending = load_pending_batch(&conn, false, None, 10).unwrap();
        assert_eq!(pending.len(), 0, "should skip already-embedded doc");
        assert_eq!(count_pending(&conn, false).unwrap(), 0);
    }

    #[test]
    fn load_pending_batch_dedups_shared_hashes() {
        let conn = open_test_db();
        let hash = "shared";
        conn.execute(
            "INSERT INTO content (hash, doc, created_at) VALUES (?1,'hello','2024-01-01')",
            [hash],
        )
        .unwrap();
        for path in ["a.md", "b.md"] {
            conn.execute(
                "INSERT INTO documents (path,title,hash,created_at,modified_at,active)
                 VALUES (?1,'T',?2,'2024-01-01','2024-01-01',1)",
                params![path, hash],
            )
            .unwrap();
        }

        let pending = load_pending_batch(&conn, false, None, 10).unwrap();
        assert_eq!(pending.len(), 1, "shared content hash should embed once");
        assert_eq!(count_pending(&conn, false).unwrap(), 1);
    }

    /// Requires embedding model — skip in CI.
    #[test]
    #[ignore]
    fn embed_creates_vector_entries() {
        use crate::db::CollectionDb;
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.sqlite");
        let db = CollectionDb::open("test", &db_path, false).unwrap();
        let conn = db.conn();

        let hash = "testhash";
        conn.execute(
            "INSERT INTO content(hash,doc,created_at) VALUES(?1,'# Hello\n\nWorld content.','2024-01-01')",
            [hash],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO documents(path,title,hash,created_at,modified_at,active)
             VALUES('test.md','Hello',?1,'2024-01-01','2024-01-01',1)",
            [hash],
        )
        .unwrap();

        let embedder = Embedder::load_default().unwrap();
        let (docs, chunks) = embed(
            &db,
            &embedder,
            &EmbedOptions { force: false },
            crate::llm::models::EMBEDDING,
        )
        .unwrap();

        assert_eq!(docs, 1);
        assert!(chunks >= 1);

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM content_vectors WHERE hash=?1",
                [hash],
                |r| r.get(0),
            )
            .unwrap();
        assert!(count >= 1);
    }
}
