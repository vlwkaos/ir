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

fn env_override_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|raw| {
        matches!(
            raw.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

pub struct HybridRequest<'a> {
    pub query: &'a str,
    pub limit: usize,
    pub min_score: Option<f64>,
    pub verbose: bool,
    pub filter: &'a crate::types::Filter,
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
        Self {
            log: Vec::new(),
            verbose,
        }
    }
    fn info(&mut self, msg: impl Into<String>) {
        self.log.push(msg.into());
    }
    fn timing(&mut self, stage: &str, d: std::time::Duration) {
        if self.verbose {
            self.log
                .push(format!("[timing] {:<14} {}ms", stage, d.as_millis()));
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
pub const STRONG_SIGNAL_PRODUCT: f64 = 0.06;
pub const STRONG_SIGNAL_FLOOR: f64 = 0.40;

/// Tier-0 shortcut on raw BM25 scores (pos/(1+pos) normalization).
/// Higher floor than fused thresholds — raw BM25 at 0.40 is a moderate match, not a strong one.
pub const BM25_STRONG_FLOOR: f64 = 0.75;
pub const BM25_STRONG_GAP: f64 = 0.10;

#[derive(Debug, Clone, Copy, PartialEq)]
struct RoutingThresholds {
    fused_floor: f64,
    fused_product: f64,
    bm25_floor: f64,
    bm25_gap: f64,
}

pub fn bm25_strong_signal_thresholds() -> (f64, f64) {
    (
        env_override_f64("IR_BM25_STRONG_FLOOR_OVERRIDE").unwrap_or(BM25_STRONG_FLOOR),
        env_override_f64("IR_BM25_STRONG_GAP_OVERRIDE").unwrap_or(BM25_STRONG_GAP),
    )
}

fn default_routing_thresholds(all_preprocessed: bool) -> RoutingThresholds {
    let (fused_floor, fused_product) =
        strong_signal_thresholds_for_all_preprocessed(all_preprocessed);
    let (bm25_floor, bm25_gap) = bm25_strong_signal_thresholds();
    RoutingThresholds {
        fused_floor,
        fused_product,
        bm25_floor,
        bm25_gap,
    }
}

fn agreed_override(mut values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let first = values.next().flatten()?;
    if values.all(|v| v == Some(first)) {
        Some(first)
    } else {
        None
    }
}

fn routing_thresholds_from_overrides(
    all_preprocessed: bool,
    routings: impl Iterator<Item = Option<crate::types::RoutingConfig>>,
) -> RoutingThresholds {
    let routings: Vec<Option<crate::types::RoutingConfig>> = routings.collect();
    let mut thresholds = default_routing_thresholds(all_preprocessed);
    if let Some(v) = agreed_override(
        routings
            .iter()
            .map(|r| r.as_ref().and_then(|r| r.fused_strong_floor)),
    ) {
        thresholds.fused_floor = v;
    }
    if let Some(v) = agreed_override(
        routings
            .iter()
            .map(|r| r.as_ref().and_then(|r| r.fused_strong_product)),
    ) {
        thresholds.fused_product = v;
    }
    if let Some(v) = agreed_override(
        routings
            .iter()
            .map(|r| r.as_ref().and_then(|r| r.bm25_strong_floor)),
    ) {
        thresholds.bm25_floor = v;
    }
    if let Some(v) = agreed_override(
        routings
            .iter()
            .map(|r| r.as_ref().and_then(|r| r.bm25_strong_gap)),
    ) {
        thresholds.bm25_gap = v;
    }
    thresholds
}

impl HybridSearch {
    pub fn search(&self, dbs: &[CollectionDb], req: &HybridRequest) -> Result<SearchOutput> {
        let mut log = Logger::new(req.verbose);
        let t_total = Instant::now();
        let allow_expand_without_scorer = env_flag("IR_ALLOW_EXPANSION_WITHOUT_SCORER");
        let force_tier1_only = env_flag("IR_FORCE_TIER1_ONLY");

        // Resolve the pipeline profile once from the searched collections
        // (config > env > default) — drives ANN, rerank window, and keep-window.
        let profile = super::profile::resolve_for_query(dbs.iter().map(|d| d.retrieval.as_ref()));

        // 1. Fast retrieval: BM25 + vector + score fusion (~20ms).
        let fused = score_fusion_two_list(dbs, &self.embedder, req, profile.ann, &mut log)?;

        if fused.is_empty() {
            log.timing("total", t_total.elapsed());
            return Ok(SearchOutput {
                results: vec![],
                log: log.log,
            });
        }

        // Log fused score distribution and coverage ratio for threshold calibration.
        if log.verbose {
            let scores: Vec<String> = fused
                .iter()
                .take(5)
                .map(|r| format!("{:.3}", r.score))
                .collect();
            log.log
                .push(format!("[fused] top-5 scores: [{}]", scores.join(", ")));
            let doc_count: usize = dbs.iter().map(|db| db.active_doc_count()).sum();
            let fetch_n = req.limit * 3; // ^ matches score_fusion_two_list no-filter path
            let coverage = if doc_count > 0 {
                (fetch_n * 2) as f64 / doc_count as f64
            } else {
                1.0
            };
            log.log.push(format!(
                "[coverage] fusion_candidates={} corpus={} coverage={:.3} (threshold TBD)",
                fetch_n * 2,
                doc_count,
                coverage
            ));
        }

        // Tier-1 filter: apply to fused list before strong-signal shortcut.
        let mut fused = fused;
        super::filter::apply(&mut fused, req.filter, dbs)?;

        // Research: kNN-graph neighborhood consensus boost (IR_GRAPH_T1_CONSENSUS=1).
        // Rescores in place before signal emission and the strong-signal check,
        // so both see the boosted distribution.
        if super::graph::t1_consensus_enabled() {
            let t0 = Instant::now();
            super::graph::maybe_consensus_t1(dbs, &mut fused);
            log.timing("graph_boost", t0.elapsed());
        }

        // Research: kNN-graph cap injection into the fused list (IR_GRAPH_T1_EXPAND=1).
        // Injected docs score below their seeds, so the strong-signal top is unchanged.
        if super::graph::t1_expand_enabled() {
            let t0 = Instant::now();
            super::graph::maybe_expand_t1(dbs, &mut fused);
            log.timing("graph_pool_t1", t0.elapsed());
        }

        if fused.is_empty() {
            log.timing("total", t_total.elapsed());
            return Ok(SearchOutput {
                results: vec![],
                log: log.log,
            });
        }

        // Research instrumentation: emit fused signal via log (routed back to client stderr).
        // Activated by IR_BENCH_SIGNALS=1; no-op in normal use.
        // Uses log.info (not eprintln) because daemon stderr goes to a log file.
        if std::env::var("IR_BENCH_SIGNALS").is_ok() {
            let top = fused[0].score;
            let gap = if fused.len() >= 2 {
                top - fused[1].score
            } else {
                top
            };
            log.info(format!("SIGNAL_FUSED\t{top:.6}\t{gap:.6}"));
        }
        if force_tier1_only {
            log.info("Research override: returning tier-1 fused results only");
            log.timing("total", t_total.elapsed());
            return Ok(SearchOutput {
                results: apply_min_score(fused, req.min_score, req.limit),
                log: log.log,
            });
        }
        let disable_shortcuts = std::env::var("IR_DISABLE_SHORTCUTS").is_ok();

        // 2. Shortcut: fused results show clear winner AND post-filter count meets limit.
        //    If filter reduced count below limit, escalate to get more candidates from expansion.
        let thresholds = routing_thresholds_for_dbs(dbs);
        let strong_signal_floor = thresholds.fused_floor;
        let strong_signal_product = thresholds.fused_product;
        if !disable_shortcuts
            && is_strong_signal(&fused, strong_signal_floor, strong_signal_product)
            && (req.filter.is_empty() || fused.len() >= req.limit)
        {
            let top = fused[0].score;
            let gap = fused.get(1).map(|r| top - r.score).unwrap_or(top);
            log.info(format!(
                "Strong signal (score={top:.3}, gap={gap:.3}, product={:.3}, threshold={strong_signal_product:.3}) — skipping expansion+reranking",
                top * gap,
            ));
            log.timing("total", t_total.elapsed());
            return Ok(SearchOutput {
                results: apply_min_score(fused, req.min_score, req.limit),
                log: log.log,
            });
        }

        // 3. LLM enhancement: expand only when reranker is also available.
        // ! Expansion without reranking is harmful (p<0.05 on NFCorpus, -0.53% nDCG).
        // Research: IR_GRAPH_AS_EXPANDER=1 skips the LLM expander (its ~3.5s is the
        // dominant tier-2 cost) and lets graph injection below supply the extra
        // candidates instead — LADR-style: the index-time graph IS the expansion.
        let graph_as_expander = super::graph::graph_as_expander_enabled() && self.scorer.is_some();
        if graph_as_expander && self.expander.is_some() {
            log.info("Graph-as-expander: skipping LLM expansion (research)");
        }
        let (enhanced, expansion_ran) = if self.scorer.is_some() || allow_expand_without_scorer {
            if let Some(exp) = self.expander.as_ref().filter(|_| !graph_as_expander) {
                let t0 = Instant::now();
                let cached = self
                    .expander_cache
                    .as_ref()
                    .and_then(|c| c.get(exp.model_id(), req.query));
                if self.scorer.is_none() {
                    log.info("Expanding without reranker (research override)...");
                }
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

                let n_vec = subs
                    .iter()
                    .filter(|s| matches!(s.kind, SubQueryKind::Vec | SubQueryKind::Hyde))
                    .count();
                let n_lex = subs.iter().filter(|s| s.kind == SubQueryKind::Lex).count();
                log.info(format!(
                    "Searching {} sub-queries ({} lex, {} vec/hyde)...",
                    subs.len(),
                    n_lex,
                    n_vec
                ));

                (
                    rrf_from_subqueries(
                        dbs,
                        &self.embedder,
                        &subs,
                        req,
                        fused,
                        profile.ann,
                        &mut log,
                    )?,
                    true,
                )
            } else {
                (fused, false)
            }
        } else {
            if self.expander.is_some() {
                log.info("Skipping expansion (no reranker)");
            }
            (fused, false)
        };

        // Research: GAR-style rerank-pool expansion (IR_GRAPH_T2_EXPAND=1).
        // Graph proposes candidates; the reranker below is the query-aware judge.
        let mut enhanced = enhanced;
        let graph_injected = super::graph::t2_expand_enabled() && self.scorer.is_some();
        if graph_injected {
            let t0 = Instant::now();
            super::graph::maybe_expand_t2(dbs, &mut enhanced);
            log.timing("graph_pool", t0.elapsed());
        }

        // Tier-2 filter: apply AFTER both expansion RRF and graph injection, since
        // each introduces candidates not seen by the tier-1 filter. Graph-injected
        // docs must not bypass metadata filters (`-f meta.tags=…`, `path~…`).
        // When neither ran, enhanced == fused, already filtered at tier-1 (no-op).
        if expansion_ran || graph_injected {
            super::filter::apply(&mut enhanced, req.filter, dbs)?;
        }

        if enhanced.is_empty() {
            log.timing("total", t_total.elapsed());
            return Ok(SearchOutput {
                results: vec![],
                log: log.log,
            });
        }

        // 4. Rerank top window if scorer available (window/keep-window from profile).
        let final_results = if let Some(scorer) = &self.scorer {
            let n = enhanced.len().min(profile.rerank_window);
            log.info(format!("Reranking {n} chunks..."));
            let t0 = Instant::now();
            let result = rerank(
                scorer.as_ref(),
                req.query,
                enhanced,
                dbs,
                req.limit,
                &profile,
                &mut log,
            )?;
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
/// Retrieves limit*fetch_multiplier candidates to improve recall; over-fetches when filter active.
fn score_fusion_two_list(
    dbs: &[CollectionDb],
    embedder: &Embedder,
    req: &HybridRequest,
    use_ann: bool,
    log: &mut Logger,
) -> Result<Vec<SearchResult>> {
    let fetch_n = if req.filter.is_empty() {
        req.limit * 3
    } else {
        (req.limit * super::filter::over_fetch_multiplier(req.filter)).clamp(50, 500)
    };
    if log.verbose {
        log.log.push(format!("[filter] prefetch={fetch_n}"));
    }
    // Log preprocessor chain usage for each collection.
    if log.verbose {
        for db in dbs {
            if db.preprocessor_commands.is_empty() {
                log.log.push(format!("[preprocessor] {}: none", db.name));
            } else {
                log.log.push(format!(
                    "[preprocessor] {}: {}",
                    db.name,
                    db.preprocessor_commands.join(" | ")
                ));
            }
        }
    }
    let bm25_list = bm25_across(dbs, req.query, fetch_n)?;
    let t0 = Instant::now();
    let emb = embedder.embed_query(req.query)?;
    log.timing("embed", t0.elapsed());
    let t0 = Instant::now();
    let vec_list = vec_across(dbs, &emb, fetch_n, use_ann)?;
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
    merged.truncate(fetch_n); // keep over-fetched count; tier-1 filter will reduce further
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
    use_ann: bool,
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

    let fetch_n = if req.filter.is_empty() {
        req.limit * 2
    } else {
        (req.limit * super::filter::over_fetch_multiplier(req.filter)).clamp(50, 500)
    };

    // BM25 for lex sub-queries
    for sub in sub_queries.iter().filter(|s| s.kind == SubQueryKind::Lex) {
        let results = bm25_across(dbs, &sub.text, fetch_n)?;
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
            let results = vec_across(dbs, emb, fetch_n, use_ann)?;
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
    let result = rrf::fuse(&ranked_lists, fetch_n);
    log.timing("fusion", t0.elapsed());
    Ok(result)
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn bm25_across(dbs: &[CollectionDb], query: &str, limit: usize) -> Result<Vec<SearchResult>> {
    dbs.iter()
        .map(|db| {
            let preprocessed = db.preprocess_query(query);
            let fts_query = fts::build_query_natural(&preprocessed);
            if fts_query.is_empty() {
                return Ok(vec![]);
            }
            let q = fts::BM25Query {
                fts_query,
                collection: &db.name,
                limit,
                title_weight: None,
            };
            fts::search(db.conn(), &q)
        })
        .collect::<Result<Vec<Vec<_>>>>()
        .map(|vv| vv.into_iter().flatten().collect())
}

fn vec_across(
    dbs: &[CollectionDb],
    embedding: &[f32],
    limit: usize,
    use_ann: bool,
) -> Result<Vec<SearchResult>> {
    dbs.iter()
        .map(|db| vectors::search(db.conn(), embedding, &db.name, limit, use_ann))
        .collect::<Result<Vec<Vec<_>>>>()
        .map(|vv| vv.into_iter().flatten().collect())
}

/// Strong-signal shortcut on fused BM25+vector scores.
/// Fires when top*gap >= STRONG_SIGNAL_PRODUCT and top >= STRONG_SIGNAL_FLOOR.
/// Higher scores tolerate smaller gaps; lower scores need proportionally larger gaps.
pub(crate) fn is_strong_signal(
    results: &[SearchResult],
    floor_threshold: f64,
    product_threshold: f64,
) -> bool {
    let top = match results.first() {
        Some(r) if r.score >= floor_threshold => r.score,
        _ => return false,
    };
    if results.len() < 2 {
        return true;
    }
    let gap = top - results[1].score;
    top * gap >= product_threshold
}

pub fn strong_signal_thresholds_for_all_preprocessed(all_preprocessed: bool) -> (f64, f64) {
    let floor = env_override_f64("IR_STRONG_SIGNAL_FLOOR_OVERRIDE").unwrap_or(STRONG_SIGNAL_FLOOR);
    let product = if all_preprocessed
        && let Some(v) = env_override_f64("IR_STRONG_SIGNAL_PRODUCT_PREPROCESSED_OVERRIDE")
    {
        v
    } else {
        env_override_f64("IR_STRONG_SIGNAL_PRODUCT_OVERRIDE").unwrap_or(STRONG_SIGNAL_PRODUCT)
    };
    (floor, product)
}

pub fn strong_signal_thresholds_for_collections(cols: &[&crate::types::Collection]) -> (f64, f64) {
    let thresholds = routing_thresholds_from_overrides(
        !cols.is_empty()
            && cols
                .iter()
                .all(|c| c.preprocessor.as_ref().is_some_and(|pp| !pp.is_empty())),
        cols.iter().map(|c| c.routing.clone()),
    );
    (thresholds.fused_floor, thresholds.fused_product)
}

fn routing_thresholds_for_dbs(dbs: &[CollectionDb]) -> RoutingThresholds {
    routing_thresholds_from_overrides(
        !dbs.is_empty() && dbs.iter().all(|db| !db.preprocessor_commands.is_empty()),
        dbs.iter().map(|db| db.routing.clone()),
    )
}

pub fn bm25_strong_signal_thresholds_for_collections(
    cols: &[&crate::types::Collection],
) -> (f64, f64) {
    let thresholds = routing_thresholds_from_overrides(
        !cols.is_empty()
            && cols
                .iter()
                .all(|c| c.preprocessor.as_ref().is_some_and(|pp| !pp.is_empty())),
        cols.iter().map(|c| c.routing.clone()),
    );
    (thresholds.bm25_floor, thresholds.bm25_gap)
}

/// Tier-0 shortcut on raw BM25 scores before any vector retrieval.
/// Higher thresholds than fused shortcut — raw BM25 at 0.40 is a moderate match.
pub fn is_bm25_strong_signal(
    results: &[SearchResult],
    floor_threshold: f64,
    gap_threshold: f64,
) -> bool {
    let top = match results.first() {
        Some(r) if r.score >= floor_threshold => r.score,
        _ => return false,
    };
    if results.len() < 2 {
        return true;
    }
    (top - results[1].score) >= gap_threshold
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

/// Rerank the top window using LLM scorer; blend with fusion scores (fused×0.4 + rerank×0.6).
/// Checks llm_cache before inference and writes new scores back.
fn rerank(
    scorer: &dyn Scorer,
    query: &str,
    mut candidates: Vec<SearchResult>,
    dbs: &[CollectionDb],
    limit: usize,
    profile: &super::profile::RetrievalProfile,
    log: &mut Logger,
) -> Result<Vec<SearchResult>> {
    let top_n = candidates.len().min(profile.rerank_window);
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
        log.log
            .push(format!("[timing] rerank_cached  {n_cached}/{top_n} hits"));
    }

    // Score only uncached candidates
    if !uncached_indices.is_empty() {
        let texts: Vec<Option<String>> = uncached_indices
            .iter()
            .map(|&i| fetch_doc_text(dbs, &to_rerank[i].hash, &to_rerank[i].collection))
            .collect();
        let doc_refs: Vec<&str> = texts.iter().map(|t| t.as_deref().unwrap_or("")).collect();
        let scores = scorer.score_batch(query, &doc_refs).unwrap_or_default();

        // Collect new entries to write to cache, grouped by collection
        let mut new_entries: HashMap<&str, Vec<(String, f64)>> = HashMap::new();

        for (j, &i) in uncached_indices.iter().enumerate() {
            if texts[j].is_some()
                && let Some(&score) = scores.get(j)
            {
                rerank_scores[i] = Some(score);
                new_entries
                    .entry(to_rerank[i].collection.as_str())
                    .or_default()
                    .push((cache_keys[i].clone(), score));
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

    let mut all: Vec<SearchResult>;
    if profile.rerank_keep_window {
        // keep-window (profile; env override IR_RERANK_KEEP_WINDOW): judged docs
        // always outrank the un-judged tail; the blend only orders WITHIN the
        // window. Avoids
        // comparing 0.4-shrunk blended scores against raw fused tail scores —
        // without RRF's flat score scale that mismatch demotes every judged
        // doc whose rerank P isn't high (measured: R@100 0.35→0.24 on
        // nfcorpus rerank-without-expansion).
        let mut win: Vec<SearchResult> = to_rerank.to_vec();
        SearchResult::sort_desc(&mut win);
        all = win;
        all.extend(rest.iter().cloned());
    } else {
        all = to_rerank
            .iter()
            .cloned()
            .chain(rest.iter().cloned())
            .collect();
        SearchResult::sort_desc(&mut all);
    }
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
    use crate::types::{Collection, RoutingConfig};
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_lock() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

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
            content: None,
            chunk_seq: None,
        };

        // Below floor → not strong
        let r = vec![make(0.39), make(0.10)];
        assert!(
            !is_strong_signal(&r, STRONG_SIGNAL_FLOOR, STRONG_SIGNAL_PRODUCT),
            "score below floor should not be strong"
        );

        // At floor, product below threshold (0.40 * 0.10 = 0.04 < 0.06) → not strong
        let r = vec![make(0.40), make(0.30)];
        assert!(
            !is_strong_signal(&r, STRONG_SIGNAL_FLOOR, STRONG_SIGNAL_PRODUCT),
            "product 0.04 should not be strong"
        );

        // At floor, product at threshold (0.40 * 0.15 = 0.06) → strong
        let r = vec![make(0.40), make(0.25)];
        assert!(
            is_strong_signal(&r, STRONG_SIGNAL_FLOOR, STRONG_SIGNAL_PRODUCT),
            "product 0.06 should be strong"
        );

        // High score, product above threshold (0.80 * 0.08 = 0.064 >= 0.06) → strong
        let r = vec![make(0.80), make(0.72)];
        assert!(
            is_strong_signal(&r, STRONG_SIGNAL_FLOOR, STRONG_SIGNAL_PRODUCT),
            "product 0.064 should be strong"
        );

        // High score, tiny gap (0.80 * 0.04 = 0.032 < 0.06) → not strong
        let r = vec![make(0.80), make(0.76)];
        assert!(
            !is_strong_signal(&r, STRONG_SIGNAL_FLOOR, STRONG_SIGNAL_PRODUCT),
            "product 0.032 should not be strong"
        );

        // Single result above floor → strong
        let r = vec![make(0.50)];
        assert!(
            is_strong_signal(&r, STRONG_SIGNAL_FLOOR, STRONG_SIGNAL_PRODUCT),
            "single result above floor should be strong"
        );
    }

    #[test]
    fn strong_signal_product_uses_env_overrides() {
        let _guard = env_lock();
        unsafe {
            std::env::remove_var("IR_STRONG_SIGNAL_PRODUCT_OVERRIDE");
            std::env::remove_var("IR_STRONG_SIGNAL_PRODUCT_PREPROCESSED_OVERRIDE");
            std::env::remove_var("IR_STRONG_SIGNAL_FLOOR_OVERRIDE");
        }

        assert_eq!(
            strong_signal_thresholds_for_all_preprocessed(false),
            (STRONG_SIGNAL_FLOOR, STRONG_SIGNAL_PRODUCT)
        );
        assert_eq!(
            strong_signal_thresholds_for_all_preprocessed(true),
            (STRONG_SIGNAL_FLOOR, STRONG_SIGNAL_PRODUCT)
        );

        unsafe { std::env::set_var("IR_STRONG_SIGNAL_PRODUCT_OVERRIDE", "0.08") };
        assert_eq!(
            strong_signal_thresholds_for_all_preprocessed(false),
            (STRONG_SIGNAL_FLOOR, 0.08)
        );
        assert_eq!(
            strong_signal_thresholds_for_all_preprocessed(true),
            (STRONG_SIGNAL_FLOOR, 0.08)
        );

        unsafe { std::env::set_var("IR_STRONG_SIGNAL_PRODUCT_PREPROCESSED_OVERRIDE", "0.05") };
        assert_eq!(
            strong_signal_thresholds_for_all_preprocessed(true),
            (STRONG_SIGNAL_FLOOR, 0.05)
        );

        unsafe { std::env::set_var("IR_STRONG_SIGNAL_FLOOR_OVERRIDE", "0.45") };
        assert_eq!(
            strong_signal_thresholds_for_all_preprocessed(true),
            (0.45, 0.05)
        );

        unsafe {
            std::env::remove_var("IR_STRONG_SIGNAL_PRODUCT_OVERRIDE");
            std::env::remove_var("IR_STRONG_SIGNAL_PRODUCT_PREPROCESSED_OVERRIDE");
            std::env::remove_var("IR_STRONG_SIGNAL_FLOOR_OVERRIDE");
        }
    }

    #[test]
    fn bm25_strong_signal_uses_explicit_thresholds() {
        let make = |score: f64| SearchResult {
            collection: "c".into(),
            path: "p".into(),
            title: "t".into(),
            score,
            snippet: None,
            hash: "h".into(),
            doc_id: "#h".into(),
            content: None,
            chunk_seq: None,
        };

        let r = vec![make(0.70), make(0.60)];
        assert!(!is_bm25_strong_signal(
            &r,
            BM25_STRONG_FLOOR,
            BM25_STRONG_GAP
        ));
        assert!(is_bm25_strong_signal(&r, 0.70, 0.09));
    }

    #[test]
    fn env_flag_recognizes_truthy_values() {
        let _guard = env_lock();
        unsafe { std::env::set_var("IR_ALLOW_EXPANSION_WITHOUT_SCORER", "1") };
        assert!(env_flag("IR_ALLOW_EXPANSION_WITHOUT_SCORER"));
        unsafe { std::env::set_var("IR_ALLOW_EXPANSION_WITHOUT_SCORER", "true") };
        assert!(env_flag("IR_ALLOW_EXPANSION_WITHOUT_SCORER"));
        unsafe { std::env::set_var("IR_ALLOW_EXPANSION_WITHOUT_SCORER", "0") };
        assert!(!env_flag("IR_ALLOW_EXPANSION_WITHOUT_SCORER"));
        unsafe { std::env::remove_var("IR_ALLOW_EXPANSION_WITHOUT_SCORER") };
        assert!(!env_flag("IR_ALLOW_EXPANSION_WITHOUT_SCORER"));
    }

    #[test]
    fn bm25_strong_signal_thresholds() {
        let make = |score: f64| SearchResult {
            collection: "c".into(),
            path: "p".into(),
            title: "t".into(),
            score,
            snippet: None,
            hash: "h".into(),
            doc_id: "#h".into(),
            content: None,
            chunk_seq: None,
        };

        // Below floor → not strong
        let r = vec![make(0.74), make(0.60)];
        assert!(
            !is_bm25_strong_signal(&r, BM25_STRONG_FLOOR, BM25_STRONG_GAP),
            "score below BM25 floor should not be strong"
        );

        // At floor, gap below threshold → not strong
        let r = vec![make(0.75), make(0.66)];
        assert!(
            !is_bm25_strong_signal(&r, BM25_STRONG_FLOOR, BM25_STRONG_GAP),
            "gap 0.09 should not be strong"
        );

        // At floor, gap at threshold → strong
        let r = vec![make(0.75), make(0.64)];
        assert!(
            is_bm25_strong_signal(&r, BM25_STRONG_FLOOR, BM25_STRONG_GAP),
            "gap 0.11 should be strong"
        );

        // High score, large gap → strong
        let r = vec![make(0.90), make(0.70)];
        assert!(
            is_bm25_strong_signal(&r, BM25_STRONG_FLOOR, BM25_STRONG_GAP),
            "high score + large gap should be strong"
        );

        // Single result above floor → strong
        let r = vec![make(0.80)];
        assert!(
            is_bm25_strong_signal(&r, BM25_STRONG_FLOOR, BM25_STRONG_GAP),
            "single result above floor should be strong"
        );
    }

    #[test]
    fn collection_routing_override_applies_when_all_collections_agree() {
        let make = |name: &str| Collection {
            name: name.into(),
            path: "/tmp".into(),
            globs: vec![],
            excludes: vec![],
            description: None,
            preprocessor: Some(vec!["ko".into()]),
            routing: Some(RoutingConfig {
                fused_strong_floor: None,
                fused_strong_product: Some(0.05),
                bm25_strong_floor: None,
                bm25_strong_gap: None,
            }),
            retrieval: None,
        };
        let a = make("a");
        let b = make("b");
        let cols = vec![&a, &b];
        assert_eq!(
            strong_signal_thresholds_for_collections(&cols),
            (STRONG_SIGNAL_FLOOR, 0.05)
        );
    }

    #[test]
    fn collection_routing_override_falls_back_on_conflict() {
        let a = Collection {
            name: "a".into(),
            path: "/tmp".into(),
            globs: vec![],
            excludes: vec![],
            description: None,
            preprocessor: Some(vec!["ko".into()]),
            routing: Some(RoutingConfig {
                fused_strong_floor: None,
                fused_strong_product: Some(0.05),
                bm25_strong_floor: None,
                bm25_strong_gap: None,
            }),
            retrieval: None,
        };
        let b = Collection {
            name: "b".into(),
            path: "/tmp".into(),
            globs: vec![],
            excludes: vec![],
            description: None,
            preprocessor: None,
            routing: None,
            retrieval: None,
        };
        let cols = vec![&a, &b];
        assert_eq!(
            strong_signal_thresholds_for_collections(&cols),
            (STRONG_SIGNAL_FLOOR, STRONG_SIGNAL_PRODUCT)
        );
        assert_eq!(
            bm25_strong_signal_thresholds_for_collections(&cols),
            (BM25_STRONG_FLOOR, BM25_STRONG_GAP)
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
            content: None,
            chunk_seq: None,
        };
        let results = vec![make(0.9), make(0.5), make(0.3)];
        let filtered = apply_min_score(results, Some(0.6), 10);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].score, 0.9);
    }
}
