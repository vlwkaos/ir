// Cross-collection search dispatch.
// Single-collection (common case): direct call, no overhead.
// Multi-collection: sequential queries merged by score.
// Note: rayon parallelism requires Send; rusqlite::Connection is !Send.
// Future: use connection pool or spawn threads with Arc<Mutex<Connection>>.

use crate::db::{CollectionDb, fts};
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
            fts::search(db.conn(), &q).unwrap_or_else(|e| {
                eprintln!("warn: bm25 search on '{}' failed: {e}", db.name);
                vec![]
            })
        })
        .collect();

    merge_and_filter(results, req)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SearchResult;

    fn make(collection: &str, path: &str, score: f64) -> SearchResult {
        SearchResult {
            collection: collection.into(),
            path: path.into(),
            title: path.into(),
            score,
            snippet: None,
            hash: "h".into(),
            doc_id: "#h".into(),
        }
    }

    #[test]
    fn sorts_by_score_desc() {
        let req = SearchRequest {
            query: "q",
            limit: 10,
            min_score: None,
        };
        let out = merge_and_filter(
            vec![vec![make("c", "a.md", 0.5), make("c", "b.md", 0.9)]],
            &req,
        )
        .unwrap();
        assert_eq!(out[0].path, "b.md");
        assert_eq!(out[1].path, "a.md");
    }

    #[test]
    fn applies_min_score() {
        let req = SearchRequest {
            query: "q",
            limit: 10,
            min_score: Some(0.6),
        };
        let out = merge_and_filter(
            vec![vec![make("c", "a.md", 0.9), make("c", "b.md", 0.3)]],
            &req,
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].path, "a.md");
    }

    #[test]
    fn respects_limit() {
        let req = SearchRequest {
            query: "q",
            limit: 3,
            min_score: None,
        };
        let docs: Vec<_> = (0..10)
            .map(|i| make("c", &format!("{i}.md"), 0.5))
            .collect();
        let out = merge_and_filter(vec![docs], &req).unwrap();
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn merges_multiple_collections() {
        let req = SearchRequest {
            query: "q",
            limit: 10,
            min_score: None,
        };
        let out = merge_and_filter(
            vec![
                vec![make("col_a", "x.md", 0.7)],
                vec![make("col_b", "y.md", 0.9)],
            ],
            &req,
        )
        .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].collection, "col_b");
    }
}

fn merge_and_filter(
    result_sets: Vec<Vec<SearchResult>>,
    req: &SearchRequest,
) -> Result<Vec<SearchResult>> {
    let mut merged: Vec<SearchResult> = result_sets.into_iter().flatten().collect();

    // Sort by score descending.
    SearchResult::sort_desc(&mut merged);

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
