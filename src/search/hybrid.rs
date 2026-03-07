// Hybrid search pipeline:
//   1. BM25 probe → strong-signal shortcut (skip expansion if top score ≥ 0.85, gap ≥ 0.10)
//   2a. With query expander: lex/vec/hyde sub-queries → RRF fusion (k=60)
//   2b. Without expander: score-based fusion (0.80·vec + 0.20·bm25) — empirically optimal
//       on NFCorpus (nDCG@10: score-fusion 0.393 > vector-only 0.387 > RRF 0.372)
//   3. LLM reranking (top 20) → final score = fused×0.4 + rerank×0.6
//
// Score-fusion α=0.80 (mid-range of 0.70–0.95 plateau) selected on BEIR/NFCorpus.
// See src/bin/eval.rs for the evaluation harness.

use crate::db::{self, CollectionDb, expander_cache::ExpanderCache, fts, vectors};
use crate::error::Result;
use crate::index::hasher;
use crate::llm::{
    embedding::Embedder,
    expander::{QueryExpander, SubQuery, SubQueryKind, fallback},
    scoring::Scorer,
};
use crate::search::rrf::{self, RankedList};
use crate::types::SearchResult;
use rusqlite::Connection;
use std::collections::HashMap;
use std::time::Instant;

pub struct HybridRequest<'a> {
    pub query: &'a str,
    pub limit: usize,
    pub min_score: Option<f64>,
    pub verbose: bool,
}

pub struct SearchOutput {
    pub results: Vec<SearchResult>,
    /// Pipeline log: always contains decision messages; timing lines only when verbose=true.
    pub log: Vec<String>,
}

/// Collects pipeline log lines; timing lines gated on verbose flag.
struct Logger {
    log: Vec<String>,
    verbose: bool,
}

impl Logger {
    fn new(verbose: bool) -> Self {
        Self { log: Vec::new(), verbose }
    }
    fn info(&mut self, msg: impl Into<String>) {
        self.log.push(msg.into());
    }
    fn timing(&mut self, stage: &str, d: std::time::Duration) {
        if self.verbose {
            self.log.push(format!("[timing] {:<14} {}ms", stage, d.as_millis()));
        }
    }
}

pub struct HybridSearch {
    pub embedder: Embedder,
    pub expander: Option<Box<dyn QueryExpander>>,
    pub scorer: Option<Box<dyn Scorer>>,
    pub expander_cache: Option<ExpanderCache>,
}

/// Weight for vector in score-fusion: 0.80·vec + 0.20·bm25.
/// Tuned on BEIR/NFCorpus via grid search (α=0.70–0.95 plateau at nDCG@10≈0.393);
/// 0.80 is the robust mid-range choice. See eval --mode all for reproduction.
const SCORE_FUSION_VEC_ALPHA: f64 = 0.80;

impl HybridSearch {
    pub fn search(&self, dbs: &[CollectionDb], req: &HybridRequest) -> Result<SearchOutput> {
        let mut log = Logger::new(req.verbose);
        let t_total = Instant::now();

        // 1. BM25 probe for strong-signal shortcut.
        let t0 = Instant::now();
        let probe_results = bm25_across(dbs, req.query, req.limit)?;
        log.timing("bm25_probe", t0.elapsed());

        if is_strong_signal(&probe_results) {
            let top = probe_results.first().map(|r| r.score).unwrap_or(0.0);
            log.info(format!("Strong BM25 signal ({top:.2}) — skipping expansion"));
            log.timing("total", t_total.elapsed());
            return Ok(SearchOutput {
                results: apply_min_score(probe_results, req.min_score, req.limit),
                log: log.log,
            });
        }

        // 2. Fuse: expander → RRF; else score-fusion.
        let fused = if let Some(exp) = &self.expander {
            let t0 = Instant::now();
            let cached = self.expander_cache.as_ref()
                .and_then(|c| c.get(exp.model_id(), req.query));
            let subs = if let Some(subs) = cached {
                log.info("Expanding query (cached)...");
                log.timing("expand", t0.elapsed());
                subs
            } else {
                log.info("Expanding query...");
                let subs = exp
                    .expand_query(req.query)
                    .unwrap_or_else(|_| fallback(req.query));
                log.timing("expand", t0.elapsed());
                if let Some(cache) = &self.expander_cache {
                    cache.put(exp.model_id(), req.query, &subs);
                }
                subs
            };

            let n_vec = subs.iter().filter(|s| matches!(s.kind, SubQueryKind::Vec | SubQueryKind::Hyde)).count();
            let n_lex = subs.iter().filter(|s| s.kind == SubQueryKind::Lex).count();
            log.info(format!("Searching {} sub-queries ({} lex, {} vec/hyde)...", subs.len(), n_lex, n_vec));

            rrf_from_subqueries(dbs, &self.embedder, &subs, req, probe_results, &mut log)?
        } else {
            log.info("Score fusion (no expander)...");
            score_fusion_two_list(dbs, &self.embedder, req, &mut log)?
        };

        if fused.is_empty() {
            log.timing("total", t_total.elapsed());
            return Ok(SearchOutput { results: vec![], log: log.log });
        }

        // 3. Rerank top-20 if scorer available.
        let final_results = if let Some(scorer) = &self.scorer {
            let n = fused.len().min(20);
            log.info(format!("Reranking {n} chunks..."));
            let t0 = Instant::now();
            let result = rerank(scorer.as_ref(), req.query, fused, dbs, req.limit, &mut log)?;
            log.timing("rerank", t0.elapsed());
            result
        } else {
            fused
        };

        log.timing("total", t_total.elapsed());
        Ok(SearchOutput {
            results: apply_min_score(final_results, req.min_score, req.limit),
            log: log.log,
        })
    }
}

// ── Fusion strategies ─────────────────────────────────────────────────────────

/// Score-based linear fusion: combined = α·vec_score + (1-α)·bm25_score.
/// Retrieves limit*3 candidates from each list to improve recall before re-ranking.
fn score_fusion_two_list(
    dbs: &[CollectionDb],
    embedder: &Embedder,
    req: &HybridRequest,
    log: &mut Logger,
) -> Result<Vec<SearchResult>> {
    let fetch_n = req.limit * 3;
    let bm25_list = bm25_across(dbs, req.query, fetch_n)?;
    let t0 = Instant::now();
    let emb = embedder.embed_query(req.query)?;
    log.timing("embed", t0.elapsed());
    let t0 = Instant::now();
    let vec_list = vec_across(dbs, &emb, fetch_n)?;
    log.timing("knn", t0.elapsed());

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

    let t0 = Instant::now();
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
    log.timing("fusion", t0.elapsed());
    Ok(merged)
}

/// Multi-subquery RRF fusion.
/// Weights: lex=1.0, vec=1.5, hyde=1.0 — vector weighted higher.
/// Probe BM25 results are NOT added again if a lex sub-query already covers the same query.
fn rrf_from_subqueries(
    dbs: &[CollectionDb],
    embedder: &Embedder,
    sub_queries: &[SubQuery],
    req: &HybridRequest,
    probe_results: Vec<SearchResult>,
    log: &mut Logger,
) -> Result<Vec<SearchResult>> {
    let mut ranked_lists: Vec<RankedList> = Vec::new();

    // Partition sub-queries: lex vs vec/hyde
    let vec_subs: Vec<(usize, f64)> = sub_queries
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s.kind, SubQueryKind::Vec | SubQueryKind::Hyde))
        .map(|(i, s)| {
            let weight = match s.kind {
                SubQueryKind::Vec => 1.5,
                SubQueryKind::Hyde => 1.0,
                SubQueryKind::Lex => unreachable!(),
            };
            (i, weight)
        })
        .collect();

    // BM25 for lex sub-queries
    for sub in sub_queries.iter().filter(|s| s.kind == SubQueryKind::Lex) {
        let results = bm25_across(dbs, &sub.text, req.limit * 2)?;
        if !results.is_empty() {
            ranked_lists.push(RankedList {
                results,
                weight: 1.0,
            });
        }
    }

    // Batch-embed all vec/hyde texts at once
    if !vec_subs.is_empty() {
        let texts: Vec<String> = vec_subs
            .iter()
            .map(|&(i, _)| sub_queries[i].text.clone())
            .collect();

        let t0 = Instant::now();
        let embeddings = embedder.embed_query_batch(&texts)?;
        log.timing("embed", t0.elapsed());

        let t0 = Instant::now();
        for (emb, &(_, weight)) in embeddings.iter().zip(&vec_subs) {
            let results = vec_across(dbs, emb, req.limit * 2)?;
            if !results.is_empty() {
                ranked_lists.push(RankedList { results, weight });
            }
        }
        log.timing("knn", t0.elapsed());
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

    let t0 = Instant::now();
    let result = rrf::fuse(&ranked_lists, req.limit * 2);
    log.timing("fusion", t0.elapsed());
    Ok(result)
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn bm25_across(dbs: &[CollectionDb], query: &str, limit: usize) -> Result<Vec<SearchResult>> {
    let fts_query = fts::build_query(query);
    if fts_query.is_empty() {
        return Ok(vec![]);
    }
    dbs.iter()
        .map(|db| {
            let q = fts::BM25Query {
                fts_query: fts_query.clone(),
                collection: &db.name,
                limit,
                title_weight: None,
            };
            fts::search(db.conn(), &q)
        })
        .collect::<Result<Vec<Vec<_>>>>()
        .map(|vv| vv.into_iter().flatten().collect())
}

fn vec_across(dbs: &[CollectionDb], embedding: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
    dbs.iter()
        .map(|db| vectors::search(db.conn(), embedding, &db.name, limit))
        .collect::<Result<Vec<Vec<_>>>>()
        .map(|vv| vv.into_iter().flatten().collect())
}

/// Strong-signal shortcut: top BM25 score ≥ 0.85 AND gap to second ≥ 0.15.
fn is_strong_signal(results: &[SearchResult]) -> bool {
    if results.len() < 2 {
        return results.first().map(|r| r.score >= 0.85).unwrap_or(false);
    }
    results[0].score >= 0.85 && (results[0].score - results[1].score) >= 0.10
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

/// Rerank top-20 using LLM scorer; blend with fusion scores (fused×0.4 + rerank×0.6).
/// Checks llm_cache before inference and writes new scores back.
fn rerank(
    scorer: &dyn Scorer,
    query: &str,
    mut candidates: Vec<SearchResult>,
    dbs: &[CollectionDb],
    limit: usize,
    log: &mut Logger,
) -> Result<Vec<SearchResult>> {
    let top_n = candidates.len().min(20);
    let (to_rerank, rest) = candidates.split_at_mut(top_n);

    // Build cache keys: sha256(model_id + "\0" + query + "\0" + content_hash)
    let mid = scorer.model_id();
    let q_norm = query.trim().to_lowercase();
    let cache_keys: Vec<String> = to_rerank
        .iter()
        .map(|r| hasher::hash_bytes(format!("{}\0{}\0{}", mid, q_norm, r.hash).as_bytes()))
        .collect();

    // Batch-lookup cached scores (one query per collection DB)
    let mut cached_scores: HashMap<String, f64> = HashMap::new();
    for db in dbs {
        let keys_for_db: Vec<String> = to_rerank
            .iter()
            .zip(&cache_keys)
            .filter(|(r, _)| r.collection == db.name)
            .map(|(_, k)| k.clone())
            .collect();
        if !keys_for_db.is_empty() {
            cached_scores.extend(db::get_rerank_scores(db.conn(), &keys_for_db));
        }
    }

    // Split into cached hits and uncached misses
    let mut uncached_indices: Vec<usize> = Vec::new();
    let mut rerank_scores: Vec<Option<f64>> = vec![None; top_n];

    for (i, key) in cache_keys.iter().enumerate() {
        if let Some(&score) = cached_scores.get(key) {
            rerank_scores[i] = Some(score);
        } else {
            uncached_indices.push(i);
        }
    }

    let n_cached = top_n - uncached_indices.len();
    if n_cached > 0 && log.verbose {
        log.log.push(format!("[timing] rerank_cached  {n_cached}/{top_n} hits"));
    }

    // Score only uncached candidates
    if !uncached_indices.is_empty() {
        let texts: Vec<Option<String>> = uncached_indices
            .iter()
            .map(|&i| fetch_doc_text(dbs, &to_rerank[i].hash, &to_rerank[i].collection))
            .collect();
        let doc_refs: Vec<&str> = texts
            .iter()
            .map(|t| t.as_deref().unwrap_or(""))
            .collect();
        let scores = scorer.score_batch(query, &doc_refs).unwrap_or_default();

        // Collect new entries to write to cache, grouped by collection
        let mut new_entries: HashMap<&str, Vec<(String, f64)>> = HashMap::new();

        for (j, &i) in uncached_indices.iter().enumerate() {
            if texts[j].is_some() {
                if let Some(&score) = scores.get(j) {
                    rerank_scores[i] = Some(score);
                    new_entries
                        .entry(to_rerank[i].collection.as_str())
                        .or_default()
                        .push((cache_keys[i].clone(), score));
                }
            }
        }

        // Write new scores to cache
        for db in dbs {
            if let Some(entries) = new_entries.get(db.name.as_str()) {
                db::put_rerank_scores(db.conn(), entries);
            }
        }
    }

    // Blend scores
    for (i, result) in to_rerank.iter_mut().enumerate() {
        if let Some(rerank_score) = rerank_scores[i] {
            result.score = result.score * 0.4 + rerank_score * 0.6;
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

        // Top ≥ 0.85 but gap < 0.10 → not strong
        let r1 = vec![make(0.90), make(0.87)];
        assert!(
            !is_strong_signal(&r1),
            "gap of 0.03 should not be strong signal"
        );

        // Top ≥ 0.85 but gap just under threshold → not strong
        let r1b = vec![make(0.90), make(0.81)];
        assert!(
            !is_strong_signal(&r1b),
            "gap of 0.09 should not be strong signal"
        );

        // Top ≥ 0.85 and gap ≥ 0.10 → strong
        let r2 = vec![make(0.92), make(0.70)];
        assert!(is_strong_signal(&r2), "gap of 0.22 should be strong signal");

        // Gap just above threshold → strong
        let r2b = vec![make(0.92), make(0.80)];
        assert!(is_strong_signal(&r2b), "gap of 0.12 should be strong signal");

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
