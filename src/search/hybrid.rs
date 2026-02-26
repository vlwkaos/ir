// Hybrid search pipeline:
//   1. BM25 probe → strong-signal shortcut (skip expansion if top score ≥ 0.85, gap ≥ 0.15)
//   2. Query expansion → lex/vec/hyde sub-queries
//   3. Parallel BM25 + vector retrieval per sub-query
//   4. RRF fusion (k=60, weights [1.0, 0.8])
//   5. LLM reranking (top 20) → final score = RRF×0.4 + rerank×0.6

use crate::db::{fts, vectors, CollectionDb};
use crate::error::Result;
use crate::llm::{
    embedding::Embedder,
    expander::{fallback, Expander, SubQueryKind},
    reranker::Reranker,
};
use crate::search::rrf::{self, RankedList};
use crate::types::SearchResult;
use rusqlite::Connection;

pub struct HybridRequest<'a> {
    pub query: &'a str,
    pub limit: usize,
    pub min_score: Option<f64>,
}

pub struct HybridSearch {
    pub embedder: Embedder,
    pub expander: Option<Expander>,
    pub reranker: Option<Reranker>,
}

impl HybridSearch {
    pub fn search(
        &self,
        dbs: &[CollectionDb],
        req: &HybridRequest,
    ) -> Result<Vec<SearchResult>> {
        // 1. BM25 probe for strong-signal shortcut.
        let probe_results = bm25_across(dbs, req.query, req.limit);
        if is_strong_signal(&probe_results) {
            return Ok(apply_min_score(probe_results, req.min_score, req.limit));
        }

        // 2. Query expansion.
        let sub_queries = match &self.expander {
            Some(exp) => exp.expand(req.query).unwrap_or_else(|_| fallback(req.query)),
            None => fallback(req.query),
        };

        // 3. Retrieve results for each sub-query → build ranked lists for RRF.
        let mut ranked_lists: Vec<RankedList> = Vec::new();

        for (i, sub) in sub_queries.iter().enumerate() {
            // First sub-query gets weight 1.0, rest get 0.8.
            let weight = if i == 0 { 1.0 } else { 0.8 };

            match sub.kind {
                SubQueryKind::Lex => {
                    let results = bm25_across(dbs, &sub.text, req.limit * 2);
                    if !results.is_empty() {
                        ranked_lists.push(RankedList { results, weight });
                    }
                }
                SubQueryKind::Vec | SubQueryKind::Hyde => {
                    if let Ok(emb) = self.embedder.embed_query(&sub.text) {
                        let results = vec_across(dbs, &emb, req.limit * 2);
                        if !results.is_empty() {
                            ranked_lists.push(RankedList { results, weight });
                        }
                    }
                }
            }
        }

        // Also include the BM25 probe results in the fusion.
        if !probe_results.is_empty() {
            ranked_lists.push(RankedList { results: probe_results, weight: 1.0 });
        }

        if ranked_lists.is_empty() {
            return Ok(vec![]);
        }

        // 4. RRF fusion.
        let fused = rrf::fuse(&ranked_lists, req.limit * 2);

        // 5. Rerank top-20 if reranker available.
        let final_results = match &self.reranker {
            Some(reranker) => rerank(reranker, req.query, fused, dbs, req.limit)?,
            None => fused,
        };

        Ok(apply_min_score(final_results, req.min_score, req.limit))
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn bm25_across(dbs: &[CollectionDb], query: &str, limit: usize) -> Vec<SearchResult> {
    let fts_query = fts::build_query(query);
    if fts_query.is_empty() {
        return vec![];
    }
    dbs.iter()
        .flat_map(|db| {
            let q = fts::BM25Query { fts_query: fts_query.clone(), collection: &db.name, limit };
            fts::search(db.conn(), &q).unwrap_or_default()
        })
        .collect()
}

fn vec_across(dbs: &[CollectionDb], embedding: &[f32], limit: usize) -> Vec<SearchResult> {
    dbs.iter()
        .flat_map(|db| {
            vectors::search(db.conn(), embedding, &db.name, limit).unwrap_or_default()
        })
        .collect()
}

/// Strong-signal shortcut: top BM25 score ≥ 0.85 AND gap to second ≥ 0.15.
fn is_strong_signal(results: &[SearchResult]) -> bool {
    if results.len() < 2 {
        return results.first().map(|r| r.score >= 0.85).unwrap_or(false);
    }
    results[0].score >= 0.85 && (results[0].score - results[1].score) >= 0.15
}

fn apply_min_score(
    mut results: Vec<SearchResult>,
    min_score: Option<f64>,
    limit: usize,
) -> Vec<SearchResult> {
    if let Some(min) = min_score {
        results.retain(|r| r.score >= min);
    }
    results.truncate(limit);
    results
}

/// Rerank top-20 using LLM cross-encoder, blend with RRF scores.
fn rerank(
    reranker: &Reranker,
    query: &str,
    mut candidates: Vec<SearchResult>,
    dbs: &[CollectionDb],
    limit: usize,
) -> Result<Vec<SearchResult>> {
    let top_n = candidates.len().min(20);
    let (to_rerank, rest) = candidates.split_at_mut(top_n);

    for result in to_rerank.iter_mut() {
        // Fetch doc content for reranking.
        let doc_text = fetch_doc_text(dbs, &result.hash, &result.collection);
        if let Some(text) = doc_text {
            let rrf_score = result.score;
            match reranker.score(query, &text) {
                Ok(rerank_score) => {
                    result.score = rrf_score * 0.4 + rerank_score * 0.6;
                }
                Err(_) => {} // keep RRF score on error
            }
        }
    }

    let mut all: Vec<SearchResult> = to_rerank
        .iter()
        .cloned()
        .chain(rest.iter().cloned())
        .collect();
    all.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    all.truncate(limit);
    Ok(all)
}

fn fetch_doc_text(dbs: &[CollectionDb], hash: &str, collection: &str) -> Option<String> {
    let db = dbs.iter().find(|d| d.name == collection)?;
    fetch_content(db.conn(), hash)
}

fn fetch_content(conn: &Connection, hash: &str) -> Option<String> {
    conn.query_row(
        "SELECT doc FROM content WHERE hash = ?1",
        [hash],
        |row| row.get(0),
    )
    .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strong_signal_requires_gap() {
        let make = |score: f64| SearchResult {
            collection: "c".into(),
            path: "p".into(),
            title: "t".into(),
            score,
            snippet: None,
            hash: "h".into(),
            doc_id: "#h".into(),
        };

        // Top ≥ 0.85 but gap < 0.15 → not strong
        let r1 = vec![make(0.90), make(0.82)];
        assert!(!is_strong_signal(&r1), "gap of 0.08 should not be strong signal");

        // Top ≥ 0.85 and gap ≥ 0.15 → strong
        let r2 = vec![make(0.92), make(0.70)];
        assert!(is_strong_signal(&r2), "gap of 0.22 should be strong signal");

        // Top < 0.85 → not strong
        let r3 = vec![make(0.80), make(0.50)];
        assert!(!is_strong_signal(&r3), "score below 0.85 should not be strong signal");
    }

    #[test]
    fn apply_min_score_filters() {
        let make = |s: f64| SearchResult {
            collection: "c".into(),
            path: "p".into(),
            title: "t".into(),
            score: s,
            snippet: None,
            hash: "h".into(),
            doc_id: "#h".into(),
        };
        let results = vec![make(0.9), make(0.5), make(0.3)];
        let filtered = apply_min_score(results, Some(0.6), 10);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].score, 0.9);
    }
}
