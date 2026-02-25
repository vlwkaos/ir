// Cross-collection search dispatch.
// Single-collection (common case): direct call, no overhead.
// Multi-collection: sequential queries merged by score.
// Note: rayon parallelism requires Send; rusqlite::Connection is !Send.
// Future: use connection pool or spawn threads with Arc<Mutex<Connection>>.

use crate::db::{fts, CollectionDb};
use crate::error::Result;
use crate::types::SearchResult;

pub struct SearchRequest<'a> {
    pub query: &'a str,
    pub limit: usize,
    pub min_score: Option<f64>,
}

/// Run BM25 search across one or more collection DBs, merge and sort by score.
/// Note: rusqlite::Connection is !Send, so we use sequential iteration here.
/// In practice the common case is a single collection (no overhead).
pub fn bm25(dbs: &[CollectionDb], req: &SearchRequest) -> Result<Vec<SearchResult>> {
    let fts_query = fts::build_query(req.query);
    if fts_query.is_empty() {
        return Ok(vec![]);
    }

    let results: Vec<Vec<SearchResult>> = dbs
        .iter()
        .map(|db| {
            let q = fts::BM25Query {
                fts_query: fts_query.clone(),
                collection: &db.name,
                limit: req.limit * 2, // over-fetch to allow for merging
            };
            fts::search(db.conn(), &q).unwrap_or_default()
        })
        .collect();

    merge_and_filter(results, req)
}

fn merge_and_filter(
    result_sets: Vec<Vec<SearchResult>>,
    req: &SearchRequest,
) -> Result<Vec<SearchResult>> {
    let mut merged: Vec<SearchResult> = result_sets.into_iter().flatten().collect();

    // Sort by score descending.
    merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Apply min_score filter.
    if let Some(min) = req.min_score {
        merged.retain(|r| r.score >= min);
    }

    // Deduplicate by (collection, path) — take best score.
    merged.dedup_by(|a, b| {
        if a.collection == b.collection && a.path == b.path {
            // b has higher score (we sorted desc), keep b
            true
        } else {
            false
        }
    });

    merged.truncate(req.limit);
    Ok(merged)
}
