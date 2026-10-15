// Embedding pipeline: load un-embedded documents → chunk → embed → store.
// Runs after `ir update`. Safe to re-run: skips already-embedded hashes.

use crate::db::{self, CollectionDb, vectors};
use crate::error::Result;
use crate::index::{chunker, new_progress_bar};
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

    let pending_count = pending_count(conn, opts.force)?;

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

        for (path, title, hash, doc_text) in &pending {
            pb.set_message(path.clone());

            // Compute embeddings before touching the DB.
            let chunks = chunker::chunk_document(doc_text);
            let inputs: Vec<(String, String)> = chunks
                .iter()
                .map(|c| (title.clone(), c.text.clone()))
                .collect();
            let embeddings = embedder.embed_doc_batch(&inputs)?;

            // Write atomically: a crash mid-insert would leave no partial state.
            db::write_txn(conn, || {
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
                for (chunk, emb) in chunks.iter().zip(embeddings.iter()) {
                    let hash_seq = format!("{hash}_{}", chunk.seq);
                    vectors::insert(conn, &hash_seq, emb)?;
                    vectors::mark_embedded(
                        conn,
                        hash,
                        chunk.seq as i64,
                        chunk.pos as i64,
                        model_name,
                    )?;
                }
                Ok(())
            })?;

            total_chunks += chunks.len();
            total_docs += 1;
            pb.inc(1);
        }

        cursor = pending.last().map(|(_, _, hash, _)| hash.clone());
    }

    pb.finish_with_message("done");
    Ok((total_docs, total_chunks))
}

/// Return the number of documents requiring inference without loading the model.
/// Unreferenced hashes remain reusable caches until an explicit force rebuild.
pub fn pending_count(conn: &Connection, force: bool) -> Result<usize> {
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
        assert_eq!(pending_count(&conn, false).unwrap(), 1);
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
        assert_eq!(pending_count(&conn, false).unwrap(), 0);
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
        assert_eq!(pending_count(&conn, false).unwrap(), 1);
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
