// Vector storage and kNN search via sqlite-vec.
// Vectors are stored as little-endian f32 blobs.
// docs: https://alexgarcia.xyz/sqlite-vec/api-reference.html

use crate::error::Result;
use crate::llm::to_bytes;
use crate::types::SearchResult;
use rusqlite::Connection;
use std::collections::HashMap;

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

// sqlite-vec enforces a hard kNN limit of 4096; exceeding it returns 0 rows silently.
pub const KNN_MAX: usize = 4096;

/// kNN search: find the `limit` closest vectors to `query_embedding`.
/// Returns (hash_seq, distance) pairs sorted by distance asc.
pub fn knn(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> Result<Vec<VecSearchResult>> {
    let k = limit.min(KNN_MAX);
    let blob = to_bytes(query_embedding);
    let sql = "
        SELECT hash_seq, distance
        FROM vectors_vec
        WHERE embedding MATCH ?1
          AND k = ?2
        ORDER BY distance
    ";
    let mut stmt = conn.prepare_cached(sql)?;
    let rows = stmt.query_map(rusqlite::params![blob, k as i64], |row| {
        Ok(VecSearchResult {
            hash_seq: row.get(0)?,
            distance: row.get(1)?,
        })
    })?;

    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

/// Full vector search: kNN → batch document lookup → deduplicate by path.
pub fn search(
    conn: &Connection,
    query_embedding: &[f32],
    collection: &str,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    // Over-fetch to deduplicate (multiple chunks per doc).
    let raw = knn(conn, query_embedding, limit * 4)?;
    if raw.is_empty() {
        return Ok(vec![]);
    }

    // kNN results are sorted by distance asc; first occurrence of each hash is its best chunk.
    // Carry the chunk seq so callers can retrieve that specific chunk if needed.
    let mut hash_order: Vec<(&str, f64, usize)> = Vec::new();
    let mut seen_hashes: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for r in &raw {
        let (hash, seq) = match r.hash_seq.rsplit_once('_') {
            Some((h, s)) => (h, s.parse::<usize>().unwrap_or(0)),
            None => (r.hash_seq.as_str(), 0),
        };
        if seen_hashes.insert(hash) {
            hash_order.push((hash, r.distance, seq));
        }
    }

    // Single query to fetch all document metadata for matched hashes.
    let placeholders = hash_order.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT d.hash, d.path, d.title, cu.seq, cu.unit_kind, cu.language, cu.symbol,
                cu.start_byte, cu.end_byte, cu.start_line, cu.end_line, cu.text_hash, cu.indexed_at
         FROM documents d
         LEFT JOIN content_units cu ON cu.hash = d.hash
         WHERE d.hash IN ({placeholders}) AND d.active = 1"
    );
    let hashes: Vec<&str> = hash_order.iter().map(|(h, _, _)| *h).collect();
    let mut stmt = conn.prepare(&sql)?;
    type UnitMeta = (
        String,
        String,
        Option<i64>,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<i64>,
        Option<i64>,
        Option<i64>,
        Option<i64>,
        Option<String>,
        Option<String>,
    );
    let mut hash_meta: HashMap<String, Vec<UnitMeta>> = HashMap::new();
    for row in stmt.query_map(rusqlite::params_from_iter(hashes.iter().copied()), |row| {
        Ok((
            row.get::<_, String>(0)?,
            (
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<i64>>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, Option<String>>(6)?,
                row.get::<_, Option<i64>>(7)?,
                row.get::<_, Option<i64>>(8)?,
                row.get::<_, Option<i64>>(9)?,
                row.get::<_, Option<i64>>(10)?,
                row.get::<_, Option<String>>(11)?,
                row.get::<_, Option<String>>(12)?,
            ),
        ))
    })? {
        let (hash, meta) = row?;
        hash_meta.entry(hash).or_default().push(meta);
    }

    // Build result list, deduplicating by path with O(1) lookup.
    let mut results: Vec<SearchResult> = Vec::new();
    let mut path_idx: HashMap<String, usize> = HashMap::new();

    for (hash, distance, seq) in &hash_order {
        if let Some(metas) = hash_meta.get(*hash) {
            let meta = metas
                .iter()
                .find(|m| m.2 == Some(*seq as i64))
                .or_else(|| metas.first())
                .expect("non-empty metadata list");
            let (
                path,
                title,
                unit_seq,
                unit_kind,
                language,
                symbol,
                start_byte,
                end_byte,
                start_line,
                end_line,
                text_hash,
                indexed_at,
            ) = meta;
            let score = 1.0 - distance;
            if let Some(&idx) = path_idx.get(path) {
                if score > results[idx].score {
                    results[idx].score = score;
                    results[idx].chunk_seq = Some(*seq);
                    results[idx].unit_seq = unit_seq.map(|v| v as usize).or(Some(*seq));
                    results[idx].unit_kind = unit_kind.clone();
                    results[idx].language = language.clone();
                    results[idx].symbol = symbol.clone();
                    results[idx].start_line = start_line.map(|v| v as usize);
                    results[idx].end_line = end_line.map(|v| v as usize);
                    results[idx].start_byte = start_byte.map(|v| v as usize);
                    results[idx].end_byte = end_byte.map(|v| v as usize);
                    results[idx].indexed_hash = text_hash.clone();
                    results[idx].indexed_at = indexed_at.clone();
                }
            } else {
                path_idx.insert(path.clone(), results.len());
                results.push(SearchResult {
                    collection: collection.to_string(),
                    path: path.clone(),
                    title: title.clone(),
                    score,
                    snippet: None,
                    hash: hash.to_string(),
                    doc_id: format!("#{}", &hash[..6.min(hash.len())]),
                    content: None,
                    chunk_seq: Some(*seq),
                    unit_seq: unit_seq.map(|v| v as usize).or(Some(*seq)),
                    unit_kind: unit_kind.clone(),
                    language: language.clone(),
                    symbol: symbol.clone(),
                    start_line: start_line.map(|v| v as usize),
                    end_line: end_line.map(|v| v as usize),
                    start_byte: start_byte.map(|v| v as usize),
                    end_byte: end_byte.map(|v| v as usize),
                    indexed_hash: text_hash.clone(),
                    indexed_at: indexed_at.clone(),
                    markers: Vec::new(),
                    related: Vec::new(),
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
             );
             CREATE TABLE content_units (
                hash TEXT NOT NULL,
                seq INTEGER NOT NULL DEFAULT 0,
                document_id INTEGER NOT NULL DEFAULT 1,
                unit_kind TEXT NOT NULL,
                language TEXT,
                symbol TEXT,
                start_byte INTEGER NOT NULL DEFAULT 0,
                end_byte INTEGER NOT NULL DEFAULT 0,
                start_line INTEGER NOT NULL DEFAULT 1,
                end_line INTEGER NOT NULL DEFAULT 1,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
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
        assert_eq!(results[0].chunk_seq, Some(0));
    }

    #[test]
    fn search_chunk_seq_best_chunk_wins() {
        let conn = open_test_db();
        let hash = "aabbccdd";
        conn.execute(
            "INSERT INTO documents (path, title, hash, active) VALUES ('multi.md','Multi',?1,1)",
            [hash],
        )
        .unwrap();

        // chunk 0 is further away, chunk 2 is closest
        let far = vec![0.5f32, 0.5, 0.5, 0.5];
        let near = vec![1.0f32, 0.0, 0.0, 0.0];
        insert(&conn, &format!("{hash}_0"), &far).unwrap();
        insert(&conn, &format!("{hash}_2"), &near).unwrap();

        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let results = search(&conn, &query, "col", 5).unwrap();
        assert_eq!(results.len(), 1);
        // chunk 2 is closest — its seq should win
        assert_eq!(results[0].chunk_seq, Some(2));
    }

    #[test]
    fn search_returns_metadata_for_best_matching_unit() {
        let conn = open_test_db();
        let hash = "unitmeta";
        conn.execute(
            "INSERT INTO documents (id, path, title, hash, active)
             VALUES (1, 'src/lib.rs', 'Lib', ?1, 1)",
            [hash],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO content_units
             (hash, seq, document_id, unit_kind, language, symbol, start_byte, end_byte,
              start_line, end_line, title, text, text_hash, indexed_at)
             VALUES
             (?1, 0, 1, 'function', 'rust', 'far_fn', 0, 10, 1, 2,
              'far_fn', 'fn far_fn() {}', 'far', '2026-01-01'),
             (?1, 2, 1, 'function', 'rust', 'near_fn', 20, 40, 5, 8,
              'near_fn', 'fn near_fn() {}', 'near', '2026-01-02')",
            [hash],
        )
        .unwrap();

        insert(&conn, &format!("{hash}_0"), &[0.5, 0.5, 0.5, 0.5]).unwrap();
        insert(&conn, &format!("{hash}_2"), &[1.0, 0.0, 0.0, 0.0]).unwrap();

        let results = search(&conn, &[1.0, 0.0, 0.0, 0.0], "col", 5).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_seq, Some(2));
        assert_eq!(results[0].unit_seq, Some(2));
        assert_eq!(results[0].unit_kind.as_deref(), Some("function"));
        assert_eq!(results[0].language.as_deref(), Some("rust"));
        assert_eq!(results[0].symbol.as_deref(), Some("near_fn"));
        assert_eq!(results[0].start_line, Some(5));
        assert_eq!(results[0].end_line, Some(8));
        assert_eq!(results[0].indexed_at.as_deref(), Some("2026-01-02"));
    }
}
