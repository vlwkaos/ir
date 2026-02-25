// Vector search via sqlite-vec.
// Placeholder — implemented in Phase 4.

use crate::error::Result;
use crate::types::SearchResult;
use rusqlite::Connection;

pub struct VecQuery<'a> {
    pub embedding: Vec<f32>,
    pub collection: &'a str,
    pub limit: usize,
}

pub fn search(_conn: &Connection, _q: &VecQuery) -> Result<Vec<SearchResult>> {
    // Phase 4: embed query → knn via vec0 → join documents
    Ok(vec![])
}

pub fn insert_chunk(
    _conn: &Connection,
    _hash: &str,
    _seq: i64,
    _pos: i64,
    _embedding: &[f32],
    _model: &str,
) -> Result<()> {
    // Phase 4
    Ok(())
}
