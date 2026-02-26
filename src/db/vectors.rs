// Vector storage and kNN search via sqlite-vec.
// Vectors are stored as little-endian f32 blobs.
// docs: https://alexgarcia.xyz/sqlite-vec/api-reference.html

use crate::error::Result;
use crate::llm::to_bytes;
use crate::types::SearchResult;
use rusqlite::Connection;

pub struct VecSearchResult {
    pub hash_seq: String, // "{hash}_{seq}"
    pub distance: f64,
}

/// Insert a normalized embedding for a document chunk.
/// hash_seq = "{content_hash}_{chunk_seq}"
pub fn insert(conn: &Connection, hash_seq: &str, embedding: &[f32]) -> Result<()> {
    let blob = to_bytes(embedding);
    // ! sqlite-vec virtual tables don't support INSERT OR REPLACE conflict resolution;
    //   use explicit DELETE + INSERT to handle duplicate hash_seqs safely.
    conn.execute("DELETE FROM vectors_vec WHERE hash_seq = ?1", [hash_seq])?;
    conn.execute(
        "INSERT INTO vectors_vec(hash_seq, embedding) VALUES (?1, ?2)",
        rusqlite::params![hash_seq, blob],
    )?;
    Ok(())
}

/// Record chunk metadata so embed knows what's already done.
pub fn mark_embedded(conn: &Connection, hash: &str, seq: i64, pos: i64, model: &str) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embedded_at)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params![hash, seq, pos, model, now],
    )?;
    Ok(())
}

/// kNN search: find the `limit` closest vectors to `query_embedding`.
/// Returns (hash_seq, distance) pairs sorted by distance asc.
pub fn knn(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> Result<Vec<VecSearchResult>> {
    let blob = to_bytes(query_embedding);
    let sql = "
        SELECT hash_seq, distance
        FROM vectors_vec
        WHERE embedding MATCH ?1
          AND k = ?2
        ORDER BY distance
    ";
    let mut stmt = conn.prepare_cached(sql)?;
    let rows = stmt.query_map(rusqlite::params![blob, limit as i64], |row| {
        Ok(VecSearchResult {
            hash_seq: row.get(0)?,
            distance: row.get(1)?,
        })
    })?;

    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

/// Full vector search: kNN → join documents → deduplicate by path.
pub fn search(
    conn: &Connection,
    query_embedding: &[f32],
    collection: &str,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    // Over-fetch to deduplicate (multiple chunks per doc)
    let raw = knn(conn, query_embedding, limit * 4)?;
    if raw.is_empty() {
        return Ok(vec![]);
    }

    let mut results: Vec<SearchResult> = Vec::new();

    for r in &raw {
        // hash_seq = "{hash}_{seq}"
        let hash = match r.hash_seq.rsplit_once('_') {
            Some((h, _)) => h,
            None => &r.hash_seq,
        };

        // Look up the active document for this hash.
        let row: Option<(String, String)> = conn
            .query_row(
                "SELECT d.path, d.title FROM documents d
                  WHERE d.hash = ?1 AND d.active = 1
                  LIMIT 1",
                [hash],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .ok();

        if let Some((path, title)) = row {
            // Deduplicate: keep best (lowest distance) per path.
            let score = 1.0 - r.distance; // cosine distance → similarity
            if let Some(existing) = results.iter_mut().find(|x| x.path == path) {
                if score > existing.score {
                    existing.score = score;
                }
            } else {
                results.push(SearchResult {
                    collection: collection.to_string(),
                    path,
                    title,
                    score,
                    snippet: None,
                    hash: hash.to_string(),
                    doc_id: format!("#{}", &hash[..6.min(hash.len())]),
                });
            }
        }

        if results.len() >= limit {
            break;
        }
    }

    SearchResult::sort_desc(&mut results);
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn open_test_db() -> Connection {
        crate::db::ensure_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE vectors_vec USING vec0(
                hash_seq TEXT PRIMARY KEY,
                embedding float[4] distance_metric=cosine
             );
             CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE,
                title TEXT,
                hash TEXT,
                active INTEGER DEFAULT 1
             );
             CREATE TABLE content_vectors (
                hash TEXT, seq INTEGER, pos INTEGER, model TEXT, embedded_at TEXT,
                PRIMARY KEY (hash, seq)
             );",
        )
        .unwrap();
        conn
    }

    #[test]
    fn insert_and_knn_roundtrip() {
        let conn = open_test_db();
        let v1 = vec![1.0f32, 0.0, 0.0, 0.0];
        let v2 = vec![0.0f32, 1.0, 0.0, 0.0];
        insert(&conn, "hash1_0", &v1).unwrap();
        insert(&conn, "hash2_0", &v2).unwrap();

        // Query with v1 — should return hash1_0 first
        let results = knn(&conn, &v1, 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].hash_seq, "hash1_0");
        assert!(results[0].distance < results[1].distance);
    }

    #[test]
    fn mark_embedded_persists() {
        let conn = open_test_db();
        mark_embedded(&conn, "abc123", 0, 0, "test-model").unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM content_vectors WHERE hash='abc123'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn search_joins_document() {
        let conn = open_test_db();
        let hash = "deadbeef";
        conn.execute(
            "INSERT INTO documents (path, title, hash, active) VALUES ('doc.md','Doc Title',?1,1)",
            [hash],
        )
        .unwrap();

        let v = vec![1.0f32, 0.0, 0.0, 0.0];
        insert(&conn, &format!("{hash}_0"), &v).unwrap();

        let results = search(&conn, &v, "test_col", 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "doc.md");
        assert_eq!(results[0].title, "Doc Title");
        assert!(results[0].score > 0.9);
    }
}
