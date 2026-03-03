// Hybrid search pipeline:
//   1. BM25 probe → strong-signal shortcut (skip expansion if top score ≥ 0.85, gap ≥ 0.15)
//   2a. With query expander: lex/vec/hyde sub-queries → RRF fusion (k=60)
//   2b. Without expander: score-based fusion (0.80·vec + 0.20·bm25) — empirically optimal
//       on NFCorpus (nDCG@10: score-fusion 0.393 > vector-only 0.387 > RRF 0.372)
//   3. LLM reranking (top 20) → final score = fused×0.4 + rerank×0.6
//
// Score-fusion α=0.80 (mid-range of 0.70–0.95 plateau) selected on BEIR/NFCorpus.
// See src/bin/eval.rs for the evaluation harness.

use crate::db::{CollectionDb, fts, vectors};
use crate::error::Result;
use crate::llm::{
    embedding::Embedder,
    expander::{Expander, SubQueryKind},
    reranker::Reranker,
};
use crate::search::rrf::{self, RankedList};
use crate::types::SearchResult;
use rusqlite::Connection;
use std::collections::HashMap;

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

/// Weight for vector in score-fusion: 0.80·vec + 0.20·bm25.
/// Tuned on BEIR/NFCorpus via grid search (α=0.70–0.95 plateau at nDCG@10≈0.393);
/// 0.80 is the robust mid-range choice. See eval --mode all for reproduction.
const SCORE_FUSION_VEC_ALPHA: f64 = 0.80;

impl HybridSearch {
    pub fn search(&self, dbs: &[CollectionDb], req: &HybridRequest) -> Result<Vec<SearchResult>> {
        // 1. BM25 probe for strong-signal shortcut.
        let probe_results = bm25_across(dbs, req.query, req.limit);
        if is_strong_signal(&probe_results) {
            return Ok(apply_min_score(probe_results, req.min_score, req.limit));
        }

        // 2. Fuse results: use multi-subquery RRF when expander is present,
        //    otherwise use score-based linear fusion (empirically better on BEIR).
        let fused = if let Some(exp) = &self.expander {
            rrf_with_expander(dbs, &self.embedder, exp, req, probe_results)?
        } else {
            score_fusion_two_list(dbs, &self.embedder, req)?
        };

        if fused.is_empty() {
            return Ok(vec![]);
        }

        // 3. Rerank top-20 if reranker available.
        let final_results = match &self.reranker {
            Some(reranker) => rerank(reranker, req.query, fused, dbs, req.limit)?,
            None => fused,
        };

        Ok(apply_min_score(final_results, req.min_score, req.limit))
    }
}

// ── Fusion strategies ─────────────────────────────────────────────────────────

/// Score-based linear fusion: combined = α·vec_score + (1-α)·bm25_score.
/// Retrieves limit*3 candidates from each list to improve recall before re-ranking.
fn score_fusion_two_list(
    dbs: &[CollectionDb],
    embedder: &Embedder,
    req: &HybridRequest,
) -> Result<Vec<SearchResult>> {
    let fetch_n = req.limit * 3;
    let bm25_list = bm25_across(dbs, req.query, fetch_n);
    let emb = embedder.embed_query(req.query)?;
    let vec_list = vec_across(dbs, &emb, fetch_n);

    // Union of both lists keyed by (collection, path).
    let mut scores: HashMap<(String, String), (f64, f64, SearchResult)> = HashMap::new();
    for r in &bm25_list {
        scores
            .entry((r.collection.clone(), r.path.clone()))
            .or_insert((0.0, 0.0, r.clone()))
            .0 = r.score;
    }
    for r in &vec_list {
        let entry = scores
            .entry((r.collection.clone(), r.path.clone()))
            .or_insert((0.0, 0.0, r.clone()));
        entry.1 = r.score;
    }

    let alpha = SCORE_FUSION_VEC_ALPHA;
    let mut merged: Vec<SearchResult> = scores
        .into_values()
        .map(|(bm25_score, vec_score, mut result)| {
            result.score = alpha * vec_score + (1.0 - alpha) * bm25_score;
            result
        })
        .collect();

    SearchResult::sort_desc(&mut merged);
    merged.truncate(req.limit * 2);
    Ok(merged)
}

/// Multi-subquery RRF fusion for use with the query expander.
/// Weights: lex=1.0, vec=1.5, hyde=1.0 — vector weighted higher.
/// Probe BM25 results are NOT added again if a lex sub-query already covers the same query.
fn rrf_with_expander(
    dbs: &[CollectionDb],
    embedder: &Embedder,
    expander: &Expander,
    req: &HybridRequest,
    probe_results: Vec<SearchResult>,
) -> Result<Vec<SearchResult>> {
    let sub_queries = expander
        .expand(req.query)
        .unwrap_or_else(|_| crate::llm::expander::fallback(req.query));

    let mut ranked_lists: Vec<RankedList> = Vec::new();

    for sub in &sub_queries {
        let weight = match sub.kind {
            SubQueryKind::Lex => 1.0,
            SubQueryKind::Vec => 1.5,
            SubQueryKind::Hyde => 1.0,
        };

        match sub.kind {
            SubQueryKind::Lex => {
                let results = bm25_across(dbs, &sub.text, req.limit * 2);
                if !results.is_empty() {
                    ranked_lists.push(RankedList { results, weight });
                }
            }
            SubQueryKind::Vec | SubQueryKind::Hyde => {
                if let Ok(emb) = embedder.embed_query(&sub.text) {
                    let results = vec_across(dbs, &emb, req.limit * 2);
                    if !results.is_empty() {
                        ranked_lists.push(RankedList { results, weight });
                    }
                }
            }
        }
    }

    // Include probe only if no lex sub-query was generated (guards against double-counting).
    let has_lex = sub_queries.iter().any(|s| s.kind == SubQueryKind::Lex);
    if !has_lex && !probe_results.is_empty() {
        ranked_lists.push(RankedList {
            results: probe_results,
            weight: 1.0,
        });
    }

    if ranked_lists.is_empty() {
        return Ok(vec![]);
    }

    Ok(rrf::fuse(&ranked_lists, req.limit * 2))
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn bm25_across(dbs: &[CollectionDb], query: &str, limit: usize) -> Vec<SearchResult> {
    let fts_query = fts::build_query(query);
    if fts_query.is_empty() {
        return vec![];
    }
    dbs.iter()
        .flat_map(|db| {
            let q = fts::BM25Query {
                fts_query: fts_query.clone(),
                collection: &db.name,
                limit,
                title_weight: None,
            };
            fts::search(db.conn(), &q).unwrap_or_else(|e| {
                eprintln!("warn: bm25 search on '{}' failed: {e}", db.name);
                vec![]
            })
        })
        .collect()
}

fn vec_across(dbs: &[CollectionDb], embedding: &[f32], limit: usize) -> Vec<SearchResult> {
    dbs.iter()
        .flat_map(|db| {
            vectors::search(db.conn(), embedding, &db.name, limit).unwrap_or_else(|e| {
                eprintln!("warn: vector search on '{}' failed: {e}", db.name);
                vec![]
            })
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

/// Rerank top-20 using LLM cross-encoder, blend with fusion scores.
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
        let doc_text = fetch_doc_text(dbs, &result.hash, &result.collection);
        if let Some(text) = doc_text {
            let fusion_score = result.score;
            match reranker.score(query, &text) {
                Ok(rerank_score) => {
                    result.score = fusion_score * 0.4 + rerank_score * 0.6;
                }
                Err(_) => {} // keep fusion score on error
            }
        }
    }

    let mut all: Vec<SearchResult> = to_rerank
        .iter()
        .cloned()
        .chain(rest.iter().cloned())
        .collect();
    SearchResult::sort_desc(&mut all);
    all.truncate(limit);
    Ok(all)
}

fn fetch_doc_text(dbs: &[CollectionDb], hash: &str, collection: &str) -> Option<String> {
    let db = dbs.iter().find(|d| d.name == collection)?;
    fetch_content(db.conn(), hash)
}

fn fetch_content(conn: &Connection, hash: &str) -> Option<String> {
    conn.query_row("SELECT doc FROM content WHERE hash = ?1", [hash], |row| {
        row.get(0)
    })
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
        assert!(
            !is_strong_signal(&r1),
            "gap of 0.08 should not be strong signal"
        );

        // Top ≥ 0.85 and gap ≥ 0.15 → strong
        let r2 = vec![make(0.92), make(0.70)];
        assert!(is_strong_signal(&r2), "gap of 0.22 should be strong signal");

        // Top < 0.85 → not strong
        let r3 = vec![make(0.80), make(0.50)];
        assert!(
            !is_strong_signal(&r3),
            "score below 0.85 should not be strong signal"
        );
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
