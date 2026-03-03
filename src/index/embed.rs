// Embedding pipeline: load un-embedded documents → chunk → embed → store.
// Runs after `ir update`. Safe to re-run: skips already-embedded hashes.

use crate::db::{CollectionDb, vectors};
use crate::error::Result;
use crate::index::{chunker, new_progress_bar};
use crate::llm::embedding::Embedder;
use rusqlite::Connection;

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

    // Clean up embeddings for hashes no longer referenced by any active document.
    cleanup_orphaned(conn)?;

    // Find documents whose content is not yet embedded (or force all).
    let pending = if opts.force {
        // Re-embed all active documents.
        load_all_active(conn)?
    } else {
        load_unembedded(conn)?
    };

    if pending.is_empty() {
        return Ok((0, 0));
    }

    // Content-addressed: multiple paths can share the same hash. Dedup so we don't
    // attempt to insert the same hash_seq twice (which would violate vectors_vec PK).
    let mut seen = std::collections::HashSet::new();
    let pending: Vec<_> = pending
        .into_iter()
        .filter(|(_, _, hash, _)| seen.insert(hash.clone()))
        .collect();

    let pb = new_progress_bar(pending.len() as u64);

    let mut total_chunks = 0usize;

    for (path, title, hash, doc_text) in &pending {
        pb.set_message(path.clone());

        // Force: remove existing embeddings for this hash first.
        if opts.force {
            // Collect seqs before deleting content_vectors (sqlite-vec can't LIKE on PK).
            let seqs: Vec<i64> = {
                let mut stmt = conn.prepare("SELECT seq FROM content_vectors WHERE hash = ?1")?;
                stmt.query_map([hash], |r| r.get(0))?
                    .filter_map(|r| r.ok())
                    .collect()
            };
            if !seqs.is_empty() {
                let hash_seqs: Vec<String> = seqs.iter().map(|s| format!("{hash}_{s}")).collect();
                let ph = hash_seqs.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                conn.execute(
                    &format!("DELETE FROM vectors_vec WHERE hash_seq IN ({ph})"),
                    rusqlite::params_from_iter(hash_seqs.iter().map(|s| s.as_str())),
                )?;
            }
            conn.execute("DELETE FROM content_vectors WHERE hash = ?1", [hash])?;
        }

        let chunks = chunker::chunk_document(doc_text);
        let inputs: Vec<(String, String)> = chunks
            .iter()
            .map(|c| (title.clone(), c.text.clone()))
            .collect();

        let embeddings = embedder.embed_doc_batch(&inputs)?;

        for (chunk, emb) in chunks.iter().zip(embeddings.iter()) {
            let hash_seq = format!("{hash}_{}", chunk.seq);
            vectors::insert(conn, &hash_seq, emb)?;
            vectors::mark_embedded(conn, hash, chunk.seq as i64, chunk.pos as i64, model_name)?;
        }

        total_chunks += chunks.len();
        pb.inc(1);
    }

    pb.finish_with_message("done");
    Ok((pending.len(), total_chunks))
}

/// Load active documents that have no entry in content_vectors.
fn load_unembedded(conn: &Connection) -> Result<Vec<(String, String, String, String)>> {
    let sql = "
        SELECT d.path, d.title, d.hash, c.doc
        FROM documents d
        JOIN content c ON c.hash = d.hash
        WHERE d.active = 1
          AND NOT EXISTS (
              SELECT 1 FROM content_vectors cv WHERE cv.hash = d.hash
          )
        ORDER BY d.path
    ";
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map([], |row| {
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

fn load_all_active(conn: &Connection) -> Result<Vec<(String, String, String, String)>> {
    let sql = "
        SELECT d.path, d.title, d.hash, c.doc
        FROM documents d
        JOIN content c ON c.hash = d.hash
        WHERE d.active = 1
        ORDER BY d.path
    ";
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map([], |row| {
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
            |r| Ok(format!("{}_{}", r.get::<_, String>(0)?, r.get::<_, i64>(1)?)),
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
        conn.execute_batch(include_str!("../db/schema.sql"))
            .unwrap();
        conn
    }

    #[test]
    fn load_unembedded_finds_pending() {
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

        let pending = load_unembedded(&conn).unwrap();
        assert_eq!(pending.len(), 1, "should find 1 unembedded doc");
        assert_eq!(pending[0].2, hash);
    }

    #[test]
    fn load_unembedded_skips_already_embedded() {
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

        let pending = load_unembedded(&conn).unwrap();
        assert_eq!(pending.len(), 0, "should skip already-embedded doc");
    }

    /// Requires embedding model — skip in CI.
    #[test]
    #[ignore]
    fn embed_creates_vector_entries() {
        use crate::db::CollectionDb;
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.sqlite");
        let db = CollectionDb::open("test", &db_path).unwrap();
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
