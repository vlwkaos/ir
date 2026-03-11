// Hybrid search pipeline:
//   1. BM25 + vector + score fusion (~20ms) — always runs, cheap
//   2. is_strong_signal(fused) → return fused (skip LLM enhancement)
//   3a. With query expander: expand → RRF with fused as base ranked list
//   3b. Without expander: fused results pass directly to reranking
//   4. LLM reranking (top 20) → final score = fused×0.4 + rerank×0.6
//
// Score-fusion α=0.80 (mid-range of 0.70–0.95 plateau) selected on BEIR/NFCorpus.
// Strong-signal: top*gap >= STRONG_SIGNAL_PRODUCT && top >= STRONG_SIGNAL_FLOOR.
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

/// Shortcut fires when top*gap >= product AND top >= floor.
/// Conservative defaults — calibrate against real query distributions with -v logging.
pub(crate) const STRONG_SIGNAL_PRODUCT: f64 = 0.06;
pub(crate) const STRONG_SIGNAL_FLOOR: f64 = 0.40;

impl HybridSearch {
    pub fn search(&self, dbs: &[CollectionDb], req: &HybridRequest) -> Result<SearchOutput> {
        let mut log = Logger::new(req.verbose);
        let t_total = Instant::now();

        // 1. Fast retrieval: BM25 + vector + score fusion (~20ms).
        let fused = score_fusion_two_list(dbs, &self.embedder, req, &mut log)?;

        if fused.is_empty() {
            log.timing("total", t_total.elapsed());
            return Ok(SearchOutput { results: vec![], log: log.log });
        }

        // Log fused score distribution for threshold calibration.
        if log.verbose {
            let scores: Vec<String> = fused.iter().take(5).map(|r| format!("{:.3}", r.score)).collect();
            log.log.push(format!("[fused] top-5 scores: [{}]", scores.join(", ")));
        }

        // 2. Shortcut: fused results show clear winner → skip LLM enhancement.
        if is_strong_signal(&fused) {
            let top = fused[0].score;
            let gap = fused.get(1).map(|r| top - r.score).unwrap_or(top);
            log.info(format!(
                "Strong signal (score={top:.3}, gap={gap:.3}, product={:.3}) — skipping expansion+reranking",
                top * gap
            ));
            log.timing("total", t_total.elapsed());
            return Ok(SearchOutput {
                results: apply_min_score(fused, req.min_score, req.limit),
                log: log.log,
            });
        }

        // 3. LLM enhancement: expand only when reranker is also available.
        // ! Expansion without reranking is harmful (p<0.05 on NFCorpus, -0.53% nDCG).
        let enhanced = if self.scorer.is_some() {
            if let Some(exp) = &self.expander {
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

                rrf_from_subqueries(dbs, &self.embedder, &subs, req, fused, &mut log)?
            } else {
                fused
            }
        } else {
            if self.expander.is_some() {
                log.info("Skipping expansion (no reranker)");
            }
            fused
        };

        if enhanced.is_empty() {
            log.timing("total", t_total.elapsed());
            return Ok(SearchOutput { results: vec![], log: log.log });
        }

        // 4. Rerank top-20 if scorer available.
        let final_results = if let Some(scorer) = &self.scorer {
            let n = enhanced.len().min(20);
            log.info(format!("Reranking {n} chunks..."));
            let t0 = Instant::now();
            let result = rerank(scorer.as_ref(), req.query, enhanced, dbs, req.limit, &mut log)?;
            log.timing("rerank", t0.elapsed());
            result
        } else {
            enhanced
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
/// base_results (fused BM25+vector) are always included: vector signal is not
/// ! duplicated by lex sub-queries which only run BM25, so always folding in is correct.
fn rrf_from_subqueries(
    dbs: &[CollectionDb],
    embedder: &Embedder,
    sub_queries: &[SubQuery],
    req: &HybridRequest,
    base_results: Vec<SearchResult>,
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

    // Always include fused base results (BM25+vector): adds vector signal not present in lex lists.
    if !base_results.is_empty() {
        ranked_lists.push(RankedList {
            results: base_results,
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

/// Strong-signal shortcut on fused BM25+vector scores.
/// Fires when top*gap >= STRONG_SIGNAL_PRODUCT and top >= STRONG_SIGNAL_FLOOR.
/// Higher scores tolerate smaller gaps; lower scores need proportionally larger gaps.
pub(crate) fn is_strong_signal(results: &[SearchResult]) -> bool {
    let top = match results.first() {
        Some(r) if r.score >= STRONG_SIGNAL_FLOOR => r.score,
        _ => return false,
    };
    if results.len() < 2 {
        return true;
    }
    let gap = top - results[1].score;
    top * gap >= STRONG_SIGNAL_PRODUCT
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
    fn strong_signal_product_boundary() {
        let make = |score: f64| SearchResult {
            collection: "c".into(),
            path: "p".into(),
            title: "t".into(),
            score,
            snippet: None,
            hash: "h".into(),
            doc_id: "#h".into(),
        };

        // Below floor → not strong
        let r = vec![make(0.39), make(0.10)];
        assert!(!is_strong_signal(&r), "score below floor should not be strong");

        // At floor, product below threshold (0.40 * 0.10 = 0.04 < 0.06) → not strong
        let r = vec![make(0.40), make(0.30)];
        assert!(!is_strong_signal(&r), "product 0.04 should not be strong");

        // At floor, product at threshold (0.40 * 0.15 = 0.06) → strong
        let r = vec![make(0.40), make(0.25)];
        assert!(is_strong_signal(&r), "product 0.06 should be strong");

        // High score, product above threshold (0.80 * 0.08 = 0.064 >= 0.06) → strong
        let r = vec![make(0.80), make(0.72)];
        assert!(is_strong_signal(&r), "product 0.064 should be strong");

        // High score, tiny gap (0.80 * 0.04 = 0.032 < 0.06) → not strong
        let r = vec![make(0.80), make(0.76)];
        assert!(!is_strong_signal(&r), "product 0.032 should not be strong");

        // Single result above floor → strong
        let r = vec![make(0.50)];
        assert!(is_strong_signal(&r), "single result above floor should be strong");
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
