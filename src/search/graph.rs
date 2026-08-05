// Graph-expanded retrieval over the kNN document graph (db::graph).
//
// Two research variants, both deterministic and model-free at query time:
//   T0-expand    (IR_GRAPH_T0_EXPAND=1)    — BM25 seeds pull graph neighbors into
//     the candidate list. Neighbor score = max(seed_score · edge_weight) · γ, so
//     seeds and graph arrivals live on one flat scale and a neighbor can never
//     outrank the seed that brought it in.
//   T1-consensus (IR_GRAPH_T1_CONSENSUS=1) — fused scores blended with the
//     weighted mean of in-list neighbor scores: docs whose semantic neighborhood
//     also matched the query rise; lone high-scorers are damped.
//
// Tuning (research env vars): IR_GRAPH_DECAY (γ, default 0.8),
// IR_GRAPH_SEEDS (default 10), IR_GRAPH_LAMBDA (λ, default 0.2).

use crate::db::{CollectionDb, graph};
use crate::types::SearchResult;
use std::collections::HashMap;

const DEFAULT_DECAY: f64 = 0.8;
const DEFAULT_SEEDS: usize = 10;
const DEFAULT_LAMBDA: f64 = 0.2;

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        // Reject NaN/±inf: a non-finite γ/λ propagates into scores and
        // SearchResult::sort_desc treats NaN compares as Equal, silently
        // scrambling rank order instead of failing a sweep loudly.
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

use crate::config::env_flag;

/// T0 ranking mode: "cap" (default) = score-capped injection (zero-harm; inert
/// in saturated lists), "rrf" = rank-based fusion of BM25 rank with graph
/// activation mass (can displace; effective in dense lists where scores are
/// compressed and the cap can never crack top-k).
fn t0_rrf_mode() -> bool {
    std::env::var("IR_GRAPH_T0_MODE")
        .map(|v| v.eq_ignore_ascii_case("rrf"))
        .unwrap_or(false)
}

const RRF_K: f64 = 60.0;
const DEFAULT_RRF_WEIGHT: f64 = 0.5;

/// Weighted-RRF rescore: every doc gets 1/(k+own_rank) if present in the list,
/// plus λ·Σ over seeds s linking to it of w(s,d)/(k+rank(s)) — graph mass from
/// high-ranked seeds (λ = IR_GRAPH_RRF_WEIGHT, default 0.5). Rank-based, so
/// immune to score compression in dense lists. A doc endorsed by both signals
/// may legitimately outrank a seed endorsed by one — two-signal consensus.
/// `edges`: (seed_key, neighbor_key, weight); `ranks`: key → 0-based rank.
fn rrf_rescore(
    ranks: &HashMap<String, usize>,
    edges: &[(String, String, f64)],
    n_seeds: usize,
    rrf_weight: f64,
) -> HashMap<String, f64> {
    let mut score: HashMap<String, f64> = ranks
        .iter()
        .map(|(k, &r)| (k.clone(), 1.0 / (RRF_K + r as f64 + 1.0)))
        .collect();
    // Mass conservation (PPR-style): each seed distributes its reciprocal-rank
    // mass ACROSS its edges proportionally to weight, rather than emitting full
    // mass per edge — otherwise hub seeds flood the list with their whole cluster.
    let mut out_strength: HashMap<&str, f64> = HashMap::new();
    for (seed, _, w) in edges {
        *out_strength.entry(seed.as_str()).or_insert(0.0) += w.clamp(0.0, 1.0);
    }
    for (seed, neighbor, w) in edges {
        let Some(&sr) = ranks.get(seed) else { continue };
        if sr >= n_seeds {
            continue;
        }
        let total = out_strength.get(seed.as_str()).copied().unwrap_or(1.0);
        if total <= 0.0 {
            continue;
        }
        let mass = rrf_weight * (w.clamp(0.0, 1.0) / total) / (RRF_K + sr as f64 + 1.0);
        *score.entry(neighbor.clone()).or_insert(0.0) += mass;
    }
    score
}

pub fn t1_consensus_enabled() -> bool {
    env_flag("IR_GRAPH_T1_CONSENSUS")
}

// ── pure propagation cores (unit-tested, no DB) ──────────────────────────────

/// T0 activation: for each neighbor key, max over seeds of seed_score·weight·γ.
/// `edges`: (seed_key, neighbor_key, weight).
fn activation_scores(
    seed_scores: &HashMap<String, f64>,
    edges: &[(String, String, f64)],
    gamma: f64,
) -> HashMap<String, f64> {
    let mut activation: HashMap<String, f64> = HashMap::new();
    for (seed, neighbor, weight) in edges {
        let Some(&s) = seed_scores.get(seed) else {
            continue;
        };
        let score = s * weight.clamp(0.0, 1.0) * gamma;
        let entry = activation.entry(neighbor.clone()).or_insert(0.0);
        if score > *entry {
            *entry = score;
        }
    }
    activation
}

/// T1 consensus for one doc: Σ(w·neighbor_score)/Σw over ALL its graph edges.
/// Neighbors absent from the result list contribute 0 to the numerator but keep
/// their weight in the denominator — consensus measures "how much of this doc's
/// semantic neighborhood also matched the query". None when the doc has no edges.
fn weighted_consensus(edges: &[(f64, Option<f64>)]) -> Option<f64> {
    let denom: f64 = edges.iter().map(|(w, _)| w.max(0.0)).sum();
    if denom <= 0.0 {
        return None;
    }
    let num: f64 = edges
        .iter()
        .map(|(w, s)| w.max(0.0) * s.unwrap_or(0.0))
        .sum();
    Some(num / denom)
}

/// Blend a fused score with its neighborhood consensus.
/// Docs without graph edges keep their score unchanged (graph coverage gaps
/// must not penalize a doc relative to covered ones... they compete on raw score).
fn blend_consensus(score: f64, consensus: Option<f64>, lambda: f64) -> f64 {
    match consensus {
        Some(c) => (1.0 - lambda) * score + lambda * c,
        None => score,
    }
}

// ── orchestrators ─────────────────────────────────────────────────────────────

pub fn t2_expand_enabled() -> bool {
    env_flag("IR_GRAPH_T2_EXPAND")
}

pub fn t1_expand_enabled() -> bool {
    env_flag("IR_GRAPH_T1_EXPAND")
}

pub fn graph_as_expander_enabled() -> bool {
    env_flag("IR_GRAPH_AS_EXPANDER")
}

/// T1: cap-inject graph neighbors of the top fused seeds into the fused list.
/// The tier-0 win used cap injection on BM25 seeds; this is the same move on
/// fused (vector+bm25) seeds. Principle-2 prediction: small gains — tier 1
/// already scans all vectors, so cosine neighbors of fused seeds are largely
/// in the pool already. This run closes that matrix slot with a measurement.
pub fn maybe_expand_t1(dbs: &[CollectionDb], results: &mut Vec<SearchResult>) {
    if !t1_expand_enabled() || results.is_empty() {
        return;
    }
    let inject = env_f64("IR_GRAPH_T1_INJECT", 30.0) as usize;
    let keep = results.len() + inject;
    expand_with_activation(dbs, results, keep, /* force_cap */ true);
}

/// T2 (GAR-style): expand the rerank candidate pool with graph neighbors of the
/// top fused docs, BEFORE the cross-encoder judges the top-20. The graph only
/// PROPOSES candidates here — the reranker is the query-aware judge, which is
/// the role division the tier-0 fusion experiments showed is required.
/// Cap-mode activation scoring places proposals just below their seeds so they
/// enter the rerank window without displacing anything above it.
/// The caller re-applies filter::apply after injection, so injected docs are
/// subject to the same metadata filters as the rest of the pool.
pub fn maybe_expand_t2(dbs: &[CollectionDb], results: &mut Vec<SearchResult>) {
    if !t2_expand_enabled() || results.is_empty() {
        return;
    }
    let inject = env_f64("IR_GRAPH_T2_INJECT", 30.0) as usize;
    let keep = results.len() + inject;
    expand_with_activation(dbs, results, keep, /* force_cap */ true);
}

/// T0: expand a BM25 result list with graph neighbors of the top seeds.
/// No-op unless `enabled` (from the retrieval profile) and a graph exists. Injected
/// docs are full SearchResults (snippet-less) scored by activation; existing docs
/// keep max(direct, activation). Caller applies filters/min_score afterwards.
pub fn maybe_expand_t0(
    dbs: &[CollectionDb],
    results: &mut Vec<SearchResult>,
    limit: usize,
    enabled: bool,
) {
    if !enabled || results.is_empty() {
        return;
    }
    expand_with_activation(dbs, results, limit, /* force_cap */ false);
}

fn expand_with_activation(
    dbs: &[CollectionDb],
    results: &mut Vec<SearchResult>,
    limit: usize,
    force_cap: bool,
) {
    let gamma = env_f64("IR_GRAPH_DECAY", DEFAULT_DECAY);
    let n_seeds = env_usize("IR_GRAPH_SEEDS", DEFAULT_SEEDS);
    let rrf = !force_cap && t0_rrf_mode();

    for db in dbs {
        if !graph::has_graph(db.conn()) {
            continue;
        }
        // Seeds: top-n results belonging to this collection.
        let seeds: Vec<&SearchResult> = results
            .iter()
            .filter(|r| r.collection == db.name)
            .take(n_seeds)
            .collect();
        if seeds.is_empty() {
            continue;
        }
        let seed_scores: HashMap<String, f64> =
            seeds.iter().map(|r| (r.path.clone(), r.score)).collect();
        let seed_paths: Vec<&str> = seeds.iter().map(|r| r.path.as_str()).collect();

        let neighbor_map = graph::neighbors_for_paths(db.conn(), &seed_paths);
        // Metadata for injected results + flat edge list for the pure core.
        let mut meta: HashMap<String, (String, String)> = HashMap::new(); // path → (title, hash)
        let mut edges: Vec<(String, String, f64)> = Vec::new();
        for (seed_path, neighbors) in neighbor_map {
            for n in neighbors {
                meta.entry(n.path.clone())
                    .or_insert_with(|| (n.title.clone(), n.hash.clone()));
                edges.push((seed_path.clone(), n.path, n.weight));
            }
        }
        // Deterministic edge order → deterministic activation (max is order-free,
        // but keep sorted for reproducible logs/debugging).
        edges.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));

        let mut present: HashMap<String, usize> = HashMap::new();
        for (i, r) in results.iter().enumerate() {
            if r.collection == db.name {
                present.insert(r.path.clone(), i);
            }
        }

        // Per-path target scores, by mode.
        let new_scores: HashMap<String, f64> = if rrf {
            // ! RRF scores live on a different scale than BM25 — research mode,
            //   meaningful for single-collection benchmarking only.
            let mut ranks: HashMap<String, usize> = HashMap::new();
            for (i, r) in results
                .iter()
                .filter(|r| r.collection == db.name)
                .enumerate()
            {
                ranks.insert(r.path.clone(), i);
            }
            rrf_rescore(
                &ranks,
                &edges,
                n_seeds,
                env_f64("IR_GRAPH_RRF_WEIGHT", DEFAULT_RRF_WEIGHT),
            )
        } else {
            activation_scores(&seed_scores, &edges, gamma)
        };

        let mut injected: Vec<SearchResult> = Vec::new();
        for (path, score) in &new_scores {
            if let Some(&i) = present.get(path.as_str()) {
                // cap mode: only ever raise a score; rrf mode: replace outright.
                if rrf || *score > results[i].score {
                    results[i].score = *score;
                }
            } else if let Some((title, hash)) = meta.get(path) {
                injected.push(SearchResult {
                    collection: db.name.clone(),
                    path: path.clone(),
                    title: title.clone(),
                    score: *score,
                    snippet: None,
                    hash: hash.clone(),
                    doc_id: format!("#{}", &hash[..6.min(hash.len())]),
                    content: None,
                    chunk_seq: None,
                });
            }
        }
        results.extend(injected);
    }

    SearchResult::sort_desc(results);
    results.truncate(limit);
}

/// T1: consensus-boost a fused result list in place (no docs added or removed).
/// No-op unless IR_GRAPH_T1_CONSENSUS is set and a graph exists.
pub fn maybe_consensus_t1(dbs: &[CollectionDb], results: &mut [SearchResult]) {
    if !t1_consensus_enabled() || results.is_empty() {
        return;
    }
    let lambda = env_f64("IR_GRAPH_LAMBDA", DEFAULT_LAMBDA);

    for db in dbs {
        if !graph::has_graph(db.conn()) {
            continue;
        }
        let in_list: HashMap<&str, f64> = results
            .iter()
            .filter(|r| r.collection == db.name)
            .map(|r| (r.path.as_str(), r.score))
            .collect();
        let paths: Vec<&str> = in_list.keys().copied().collect();
        let neighbor_map = graph::neighbors_for_paths(db.conn(), &paths);

        // Compute all boosts against the pre-boost snapshot (in_list), then apply:
        // a doc's boost must not depend on whether its neighbor was boosted first.
        let mut boosted: HashMap<String, f64> = HashMap::new();
        for (path, neighbors) in &neighbor_map {
            let edges: Vec<(f64, Option<f64>)> = neighbors
                .iter()
                .map(|n| (n.weight, in_list.get(n.path.as_str()).copied()))
                .collect();
            let consensus = weighted_consensus(&edges);
            let score = in_list[path.as_str()];
            boosted.insert(path.clone(), blend_consensus(score, consensus, lambda));
        }
        for r in results.iter_mut() {
            if r.collection == db.name
                && let Some(&s) = boosted.get(&r.path)
            {
                r.score = s;
            }
        }
    }

    SearchResult::sort_desc(results);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeds(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn edge(s: &str, n: &str, w: f64) -> (String, String, f64) {
        (s.to_string(), n.to_string(), w)
    }

    #[test]
    fn activation_takes_max_over_seeds() {
        let s = seeds(&[("a", 0.9), ("b", 0.5)]);
        let edges = vec![edge("a", "x", 0.5), edge("b", "x", 0.9)];
        let act = activation_scores(&s, &edges, 1.0);
        // a→x: 0.9·0.5 = 0.45; b→x: 0.5·0.9 = 0.45 — equal; then γ applies
        assert!((act["x"] - 0.45).abs() < 1e-9);

        let act = activation_scores(&s, &edges, 0.8);
        assert!((act["x"] - 0.36).abs() < 1e-9);
    }

    #[test]
    fn activation_never_exceeds_seed_score() {
        let s = seeds(&[("a", 0.7)]);
        let edges = vec![edge("a", "x", 1.5)]; // weight clamped to 1.0
        let act = activation_scores(&s, &edges, 0.9);
        assert!(act["x"] < 0.7, "neighbor must rank below its seed");
        assert!((act["x"] - 0.63).abs() < 1e-9);
    }

    #[test]
    fn activation_ignores_unknown_seeds_and_negative_weights() {
        let s = seeds(&[("a", 0.7)]);
        let edges = vec![edge("ghost", "x", 0.9), edge("a", "y", -0.4)];
        let act = activation_scores(&s, &edges, 0.8);
        assert!(!act.contains_key("x"));
        assert_eq!(act.get("y").copied().unwrap_or(0.0), 0.0);
    }

    #[test]
    fn consensus_weighted_mean_with_missing_neighbors() {
        // Two neighbors in-list (0.8, 0.4), one absent (counts as 0 in numerator).
        let edges = vec![(1.0, Some(0.8)), (1.0, Some(0.4)), (2.0, None)];
        let c = weighted_consensus(&edges).unwrap();
        // (1·0.8 + 1·0.4 + 2·0) / (1+1+2) = 1.2/4 = 0.3
        assert!((c - 0.3).abs() < 1e-9);
    }

    #[test]
    fn consensus_none_without_edges() {
        assert_eq!(weighted_consensus(&[]), None);
        assert_eq!(weighted_consensus(&[(-1.0, Some(0.5))]), None);
    }

    #[test]
    fn blend_boosts_supported_docs_and_damps_lone_ones() {
        // Doc with strong neighborhood support rises above equal-scored lone doc.
        let supported = blend_consensus(0.6, Some(0.7), 0.2);
        let lone_in_graph = blend_consensus(0.6, Some(0.0), 0.2);
        let no_edges = blend_consensus(0.6, None, 0.2);
        assert!(supported > 0.6);
        assert!(lone_in_graph < 0.6);
        assert_eq!(no_edges, 0.6, "docs without edges keep raw score");
    }

    #[test]
    fn rrf_rescore_lifts_neighbor_of_high_seed_in_dense_list() {
        // Dense list: ranks 0..100. Doc at rank 40 is a neighbor of the rank-0 seed.
        let mut ranks = HashMap::new();
        for i in 0..100 {
            ranks.insert(format!("d{i}"), i);
        }
        let edges = vec![("d0".to_string(), "d40".to_string(), 1.0)];
        let s = rrf_rescore(&ranks, &edges, 5, 0.5);
        // d40: own 1/101 + mass 0.5/61 must outrank rank-10's 1/71
        assert!(
            s["d40"] > s["d10"],
            "graph-connected doc must outrank rank-10"
        );
        // untouched docs keep pure rank order
        assert!(s["d1"] > s["d2"]);
    }

    #[test]
    fn rrf_rescore_ignores_seeds_beyond_n() {
        let mut ranks = HashMap::new();
        for i in 0..30 {
            ranks.insert(format!("d{i}"), i);
        }
        let edges = vec![("d25".to_string(), "d29".to_string(), 1.0)];
        let s = rrf_rescore(&ranks, &edges, 5, 0.5); // d25 is rank 25 — not a seed
        assert!(s["d29"] < s["d28"], "no mass from non-seed rank");
    }

    #[test]
    fn rrf_rescore_injects_unranked_neighbor_with_mass_only() {
        let mut ranks = HashMap::new();
        for i in 0..50 {
            ranks.insert(format!("d{i}"), i);
        }
        let edges = vec![("d0".to_string(), "new".to_string(), 1.0)];
        let s = rrf_rescore(&ranks, &edges, 5, 0.5);
        // Injected-only doc: mass 0.5/61 ≈ rank-61-equivalent — lands deep.
        // RRF mode's power is reordering in-list docs, not injection.
        assert!(s["new"] > 0.0);
        assert!(s["new"] < s["d40"], "injection lands below mid-list");
        assert!(s["new"] < s["d0"]);
    }

    #[test]
    fn blend_is_bounded() {
        // Blend of two [0,1] quantities stays in [0,1] — strong-signal
        // thresholds keep their calibrated meaning.
        for &(s, c, l) in &[(1.0, 1.0, 0.3), (0.0, 0.0, 0.3), (0.5, 1.0, 1.0)] {
            let b = blend_consensus(s, Some(c), l);
            assert!((0.0..=1.0).contains(&b), "blend {b} out of bounds");
        }
    }
}
