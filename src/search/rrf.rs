// Reciprocal Rank Fusion.
// Score(doc) = Σ weight_i / (k + rank_i + 1)
// Position bonuses: rank 0 → +0.05, rank 1-2 → +0.02
// Weights: first result set gets 1.0, subsequent 0.8 (FTS weighted higher than vec).

use crate::types::SearchResult;
use std::collections::HashMap;

const K: f64 = 60.0;

#[derive(Debug)]
pub struct RankedList {
    pub results: Vec<SearchResult>,
    /// Weight applied to this list's contribution (1.0 = full, 0.8 = secondary).
    pub weight: f64,
}

/// Merge multiple ranked lists into a single RRF-scored list.
/// Docs are keyed by `(collection, path)`.
pub fn fuse(lists: &[RankedList], limit: usize) -> Vec<SearchResult> {
    // Map (collection, path) → accumulated RRF score + best SearchResult
    let mut scores: HashMap<(String, String), (f64, SearchResult)> = HashMap::new();

    for list in lists {
        for (rank, result) in list.results.iter().enumerate() {
            let base = list.weight / (K + rank as f64 + 1.0);
            let bonus = match rank {
                0 => 0.05,
                1 | 2 => 0.02,
                _ => 0.0,
            };
            let contribution = base + bonus;

            let key = (result.collection.clone(), result.path.clone());
            scores
                .entry(key)
                .and_modify(|(score, _)| *score += contribution)
                .or_insert((contribution, result.clone()));
        }
    }

    let mut merged: Vec<SearchResult> = scores
        .into_values()
        .map(|(rrf_score, mut result)| {
            result.score = rrf_score;
            result
        })
        .collect();

    SearchResult::sort_desc(&mut merged);
    merged.truncate(limit);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SearchResult;

    fn make_result(path: &str, score: f64) -> SearchResult {
        SearchResult {
            collection: "col".into(),
            path: path.into(),
            title: path.into(),
            score,
            snippet: None,
            hash: "abc".into(),
            doc_id: "#abc".into(),
        }
    }

    #[test]
    fn fuse_single_list() {
        let list = RankedList {
            results: vec![make_result("a.md", 0.9), make_result("b.md", 0.5)],
            weight: 1.0,
        };
        let out = fuse(&[list], 10);
        assert_eq!(out.len(), 2);
        // a.md scored at rank 0 → higher RRF score
        assert_eq!(out[0].path, "a.md");
        assert!(out[0].score > out[1].score);
    }

    #[test]
    fn fuse_accumulates_across_lists() {
        // doc appearing in both lists should score higher than doc in only one
        let list1 = RankedList {
            results: vec![make_result("shared.md", 0.9), make_result("only1.md", 0.8)],
            weight: 1.0,
        };
        let list2 = RankedList {
            results: vec![make_result("shared.md", 0.9), make_result("only2.md", 0.7)],
            weight: 0.8,
        };
        let out = fuse(&[list1, list2], 10);
        let shared = out.iter().find(|r| r.path == "shared.md").unwrap();
        let only1 = out.iter().find(|r| r.path == "only1.md").unwrap();
        assert!(
            shared.score > only1.score,
            "shared doc should have higher score"
        );
    }

    #[test]
    fn fuse_respects_limit() {
        let results: Vec<_> = (0..20)
            .map(|i| make_result(&format!("{i}.md"), 0.9))
            .collect();
        let list = RankedList {
            results,
            weight: 1.0,
        };
        let out = fuse(&[list], 5);
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn rank0_position_bonus() {
        // Rank 0 gets +0.05 bonus — verify it's applied
        let list = RankedList {
            results: vec![make_result("top.md", 0.9)],
            weight: 1.0,
        };
        let out = fuse(&[list], 10);
        // Expected: 1.0 / (60 + 0 + 1) + 0.05
        let expected = 1.0 / 61.0 + 0.05;
        assert!((out[0].score - expected).abs() < 1e-10);
    }
}
