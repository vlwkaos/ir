// HNSW ANN sidecar index (research: IR_ANN=hnsw).
//
// A usearch index file lives next to each collection DB
// (`{collection}.sqlite` → `{collection}.usearch`) and mirrors `vectors_vec`.
// Keys are allocated in the `ann_keys` table (u64 rowid ↔ hash_seq).
//
// Lifecycle:
// - `sync()` runs at embed time: incrementally adds missing chunks; a model or
//   dimension change, or any removed vector, triggers a from-scratch rebuild.
// - `search()` runs at query time: returns None (→ caller falls back to exact
//   brute-force) when the flag is off, the sidecar is absent, or the index is
//   stale (count mismatch with vectors_vec). Approximation never silently
//   covers less than the full vector set.
//
// Distance semantics match sqlite-vec: cosine distance = 1 − cosine similarity.

use crate::db::vectors::VecSearchResult;
use crate::error::{Error, Result};
use crate::llm::from_bytes;
use rusqlite::Connection;
use std::path::PathBuf;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// Sidecar path: the collection sqlite path with a `.usearch` extension.
/// None for in-memory connections (tests). rusqlite's `path()` yields `&str`,
/// so the value is already valid UTF-8 — to_string_lossy downstream is lossless.
pub fn index_path(conn: &Connection) -> Option<PathBuf> {
    let db_path = conn.path()?;
    if db_path.is_empty() {
        return None; // in-memory
    }
    Some(PathBuf::from(db_path).with_extension("usearch"))
}

/// usearch save() writes in place; a crash mid-write leaves a truncated file
/// that later load() rejects (and a concurrent reader could mmap torn bytes).
/// Write to a sibling temp path and atomically rename over the target.
fn save_atomic(index: &Index, path: &std::path::Path) -> Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    index
        .save(tmp.to_string_lossy().as_ref())
        .map_err(ann_err)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn index_options(dim: usize) -> IndexOptions {
    IndexOptions {
        dimensions: dim,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        connectivity: env_usize("IR_ANN_M", 16),
        expansion_add: env_usize("IR_ANN_EF_CONSTRUCTION", 200),
        // Default 200: measured nDCG@10-identical to exact on the 50k-doc
        // validation set (0.917), 99.2% top-10 overlap, no latency penalty vs 96.
        expansion_search: env_usize("IR_ANN_EF", 200),
        multi: false,
    }
}

fn meta_get(conn: &Connection, key: &str) -> Option<String> {
    conn.query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0))
        .ok()
}

fn meta_set(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
        [key, value],
    )?;
    Ok(())
}

fn vectors_count(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("SELECT COUNT(*) FROM vectors_vec", [], |r| r.get(0))?)
}

fn ann_err(e: impl std::fmt::Display) -> Error {
    Error::Other(format!("usearch: {e}"))
}

/// Sync the sidecar with vectors_vec. Called after embedding when enabled.
/// Returns (indexed_total, added_now).
pub fn sync(conn: &Connection) -> Result<(usize, usize)> {
    let Some(path) = index_path(conn) else {
        return Ok((0, 0));
    };
    let n_vec = vectors_count(conn)?;
    if n_vec == 0 {
        let _ = std::fs::remove_file(&path);
        conn.execute("DELETE FROM ann_keys", [])?;
        meta_set(conn, "ann_count", "0")?;
        return Ok((0, 0));
    }

    // Discover dim + model from stored vectors.
    let sample: Vec<u8> = conn.query_row("SELECT embedding FROM vectors_vec LIMIT 1", [], |r| {
        r.get(0)
    })?;
    let dim = from_bytes(&sample).len();
    let model: String = conn
        .query_row("SELECT model FROM content_vectors LIMIT 1", [], |r| {
            r.get(0)
        })
        .unwrap_or_default();

    // Rebuild triggers: model/dim change, missing file, or shrunk vector set
    // (force re-embed wipes vectors_vec; usearch removal churn isn't worth it).
    let keyed: i64 = conn.query_row("SELECT COUNT(*) FROM ann_keys", [], |r| r.get(0))?;
    let stored_model = meta_get(conn, "ann_model");
    let stored_dim: usize = meta_get(conn, "ann_dim")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let stale_keys: i64 = conn.query_row(
        "SELECT COUNT(*) FROM ann_keys a
         WHERE NOT EXISTS (SELECT 1 FROM vectors_vec v WHERE v.hash_seq = a.hash_seq)",
        [],
        |r| r.get(0),
    )?;
    let rebuild = !path.exists()
        || stored_model.as_deref() != Some(model.as_str())
        || stored_dim != dim
        || stale_keys > 0;

    if rebuild {
        conn.execute("DELETE FROM ann_keys", [])?;
        return rebuild_from_scratch(conn, &path, dim, &model, n_vec);
    }

    let index = Index::new(&index_options(dim)).map_err(ann_err)?;
    // A truncated/corrupt sidecar (e.g. crash mid-save) must not wedge every
    // future embed: on load failure OR size drift, rebuild from stored vectors.
    let load_ok = index.load(path.to_string_lossy().as_ref()).is_ok();
    if !load_ok || index.size() != keyed as usize {
        conn.execute("DELETE FROM ann_keys", [])?;
        return rebuild_from_scratch(conn, &path, dim, &model, n_vec);
    }

    // Incremental: add vectors not yet keyed.
    let added = add_missing(conn, &index, n_vec as usize)?;
    if added > 0 {
        save_atomic(&index, &path)?;
    }
    meta_set(conn, "ann_model", &model)?;
    meta_set(conn, "ann_dim", &dim.to_string())?;
    meta_set(conn, "ann_count", &index.size().to_string())?;
    Ok((index.size(), added))
}

fn rebuild_from_scratch(
    conn: &Connection,
    path: &std::path::Path,
    dim: usize,
    model: &str,
    n_vec: i64,
) -> Result<(usize, usize)> {
    let index = Index::new(&index_options(dim)).map_err(ann_err)?;
    let added = add_missing(conn, &index, n_vec as usize)?;
    save_atomic(&index, path)?;
    meta_set(conn, "ann_model", model)?;
    meta_set(conn, "ann_dim", &dim.to_string())?;
    meta_set(conn, "ann_count", &index.size().to_string())?;
    Ok((index.size(), added))
}

/// Add every vectors_vec row without an ann_keys entry. Returns rows added.
fn add_missing(conn: &Connection, index: &Index, reserve_total: usize) -> Result<usize> {
    index.reserve(reserve_total).map_err(ann_err)?;
    let mut stmt = conn.prepare(
        "SELECT v.hash_seq, v.embedding FROM vectors_vec v
         WHERE NOT EXISTS (SELECT 1 FROM ann_keys a WHERE a.hash_seq = v.hash_seq)",
    )?;
    let rows: Vec<(String, Vec<u8>)> = stmt
        .query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
        })?
        .collect::<std::result::Result<_, _>>()?;
    let mut insert = conn.prepare("INSERT INTO ann_keys (hash_seq) VALUES (?1)")?;
    let mut added = 0usize;
    for (hash_seq, blob) in rows {
        insert.execute([&hash_seq])?;
        let key = conn.last_insert_rowid() as u64;
        index.add(key, &from_bytes(&blob)).map_err(ann_err)?;
        added += 1;
    }
    Ok(added)
}

/// ANN kNN. Returns None when the caller must fall back to exact search:
/// `use_ann` off (resolved from the retrieval profile), in-memory DB, sidecar
/// missing, or index stale vs vectors_vec.
pub fn search(
    conn: &Connection,
    query: &[f32],
    k: usize,
    use_ann: bool,
) -> Option<Vec<VecSearchResult>> {
    if !use_ann {
        return None;
    }
    let path = index_path(conn)?;
    if !path.exists() {
        return None;
    }
    let ann_count: i64 = meta_get(conn, "ann_count")?.parse().ok()?;
    let n_vec = vectors_count(conn).ok()?;
    if ann_count != n_vec || n_vec == 0 {
        return None; // stale — exact fallback until the next embed sync
    }

    let index = Index::new(&index_options(query.len())).ok()?;
    index.view(path.to_string_lossy().as_ref()).ok()?;
    if index.size() != n_vec as usize {
        return None;
    }
    let matches = index.search(query, k).ok()?;
    if matches.keys.is_empty() {
        return Some(vec![]);
    }

    // Map keys back to hash_seq in one query, preserving match order.
    let placeholders = matches
        .keys
        .iter()
        .map(|_| "?")
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!("SELECT key, hash_seq FROM ann_keys WHERE key IN ({placeholders})");
    let mut stmt = conn.prepare(&sql).ok()?;
    let key_params: Vec<i64> = matches.keys.iter().map(|&k| k as i64).collect();
    let map: std::collections::HashMap<u64, String> = stmt
        .query_map(rusqlite::params_from_iter(key_params.iter()), |r| {
            Ok((r.get::<_, i64>(0)? as u64, r.get::<_, String>(1)?))
        })
        .ok()?
        .flatten()
        .collect();

    Some(
        matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .filter_map(|(key, dist)| {
                map.get(key).map(|hs| VecSearchResult {
                    hash_seq: hs.clone(),
                    distance: f64::from(*dist),
                })
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::to_bytes;
    use rusqlite::Connection;

    // ANN activation is now an explicit `use_ann` arg to search() (resolved from
    // the retrieval profile), so these tests no longer touch the process env.

    fn open_file_db(dir: &std::path::Path) -> Connection {
        crate::db::ensure_sqlite_vec();
        let conn = Connection::open(dir.join("test.sqlite")).unwrap();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE vectors_vec USING vec0(
                hash_seq TEXT PRIMARY KEY,
                embedding float[4] distance_metric=cosine
             );",
        )
        .unwrap();
        conn.execute_batch(include_str!("schema_base.sql")).unwrap();
        conn
    }

    fn add_vec(conn: &Connection, hash_seq: &str, emb: &[f32]) {
        conn.execute(
            "INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
             VALUES (?1, 0, 0, 'test-model', '2024-01-01')",
            [hash_seq.split('_').next().unwrap()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?1, ?2)",
            rusqlite::params![hash_seq, to_bytes(emb)],
        )
        .unwrap();
    }

    #[test]
    fn sync_and_search_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let conn = open_file_db(dir.path());
        add_vec(&conn, "ha_0", &[1.0, 0.0, 0.0, 0.0]);
        add_vec(&conn, "hb_0", &[0.9, 0.44, 0.0, 0.0]);
        add_vec(&conn, "hc_0", &[0.0, 0.0, 1.0, 0.0]);

        let (total, added) = sync(&conn).unwrap();
        assert_eq!((total, added), (3, 3));
        assert!(index_path(&conn).unwrap().exists());

        let hits = search(&conn, &[1.0, 0.0, 0.0, 0.0], 2, true).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].hash_seq, "ha_0");
        assert!(hits[0].distance < 1e-4, "self distance ~0");
        assert_eq!(hits[1].hash_seq, "hb_0");
        // cosine distance semantics: 1 - cos(a, b)
        assert!((hits[1].distance - 0.1).abs() < 0.05);
    }

    #[test]
    fn search_falls_back_when_stale_or_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let conn = open_file_db(dir.path());
        add_vec(&conn, "ha_0", &[1.0, 0.0, 0.0, 0.0]);
        sync(&conn).unwrap();

        // Disabled (use_ann=false) → None even with a fresh index.
        assert!(search(&conn, &[1.0, 0.0, 0.0, 0.0], 1, false).is_none());

        // New vector not yet synced → stale → None.
        add_vec(&conn, "hb_0", &[0.0, 1.0, 0.0, 0.0]);
        assert!(search(&conn, &[1.0, 0.0, 0.0, 0.0], 1, true).is_none());

        // After sync it works again (incremental add).
        let (total, added) = sync(&conn).unwrap();
        assert_eq!((total, added), (2, 1));
        assert!(search(&conn, &[0.0, 1.0, 0.0, 0.0], 1, true).is_some());
    }

    #[test]
    fn model_change_triggers_rebuild() {
        let dir = tempfile::tempdir().unwrap();
        let conn = open_file_db(dir.path());
        add_vec(&conn, "ha_0", &[1.0, 0.0, 0.0, 0.0]);
        sync(&conn).unwrap();

        conn.execute("UPDATE content_vectors SET model = 'other-model'", [])
            .unwrap();
        let (total, added) = sync(&conn).unwrap();
        assert_eq!((total, added), (1, 1), "full rebuild re-adds everything");
    }

    #[test]
    fn corrupt_sidecar_triggers_rebuild_not_error() {
        let dir = tempfile::tempdir().unwrap();
        let conn = open_file_db(dir.path());
        add_vec(&conn, "ha_0", &[1.0, 0.0, 0.0, 0.0]);
        add_vec(&conn, "hb_0", &[0.0, 1.0, 0.0, 0.0]);
        sync(&conn).unwrap();

        // Simulate a crash mid-save: truncate the sidecar to garbage. The keys
        // table still says 2, model/dim unchanged → the old code hit load()-Err
        // and propagated, wedging every future embed. Now it must rebuild.
        let path = index_path(&conn).unwrap();
        std::fs::write(&path, b"not a usearch file").unwrap();
        let (total, added) = sync(&conn).unwrap();
        assert_eq!((total, added), (2, 2), "corrupt file rebuilds from vectors");

        assert!(search(&conn, &[1.0, 0.0, 0.0, 0.0], 1, true).is_some());
    }

    #[test]
    fn in_memory_db_is_noop() {
        crate::db::ensure_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        assert!(index_path(&conn).is_none());
    }
}
