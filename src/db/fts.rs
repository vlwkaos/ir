// BM25 full-text search via SQLite FTS5.
// Score normalization: score / (1 + score)  (FTS5 returns negative values; negate first)

use crate::error::Result;
use crate::types::SearchResult;
use rusqlite::Connection;

/// Escape a string for use inside FTS5 double-quoted terms.
/// FTS5 only requires `"` to be doubled; no other escaping is needed.
fn fts5_escape(s: &str) -> String {
    s.replace('"', "\"\"")
}

/// Strip English stop words from a query string.
/// Returns the original string unchanged if all tokens are stop words (avoids empty query).
#[allow(dead_code)] // ^ public utility; tested; no production caller yet
pub fn strip_stopwords(input: &str) -> String {
    let filtered: Vec<&str> = input
        .split_whitespace()
        .filter(|w| !is_stopword(w))
        .collect();
    if filtered.is_empty() {
        input.to_string()
    } else {
        filtered.join(" ")
    }
}

fn is_stopword(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "a" | "an"
            | "the"
            | "and"
            | "or"
            | "but"
            | "in"
            | "on"
            | "at"
            | "to"
            | "for"
            | "of"
            | "with"
            | "from"
            | "by"
            | "as"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "being"
            | "have"
            | "has"
            | "had"
            | "do"
            | "does"
            | "did"
            | "will"
            | "would"
            | "could"
            | "should"
            | "can"
            | "may"
            | "might"
            | "shall"
            | "not"
            | "no"
            | "that"
            | "this"
            | "these"
            | "those"
            | "it"
            | "its"
            | "i"
            | "me"
            | "my"
            | "we"
            | "our"
            | "you"
            | "your"
            | "he"
            | "him"
            | "his"
            | "she"
            | "her"
            | "they"
            | "them"
            | "their"
            | "who"
            | "whom"
            | "which"
            | "what"
            | "when"
            | "where"
            | "why"
            | "how"
            | "about"
            | "into"
            | "than"
            | "then"
            | "there"
            | "also"
            | "such"
            | "so"
            | "if"
            | "use"
            | "used"
            | "using"
    )
}

/// Build FTS5 query adapted for natural-language input.
/// Queries with more than 3 terms (after stop word removal) use OR semantics.
/// Short keyword queries (≤3 non-stop terms) keep AND semantics.
///
/// This handles question-format queries like "what are the symptoms of diabetes"
/// where AND semantics would nearly always return empty results.
pub fn build_query_natural(input: &str) -> String {
    let all_terms: Vec<&str> = input.split_whitespace().collect();
    let content_terms: Vec<&str> = all_terms
        .iter()
        .copied()
        .filter(|w| !is_stopword(w))
        .collect();

    // Short keyword query: keep existing AND semantics
    if content_terms.len() <= 3 && content_terms.len() == all_terms.len() {
        return build_query(input);
    }

    // Natural-language query: strip stop words and use OR
    let cleaned = if content_terms.is_empty() {
        // All stop words (e.g., "what is the") — preserve all terms, use OR
        all_terms.join(" ")
    } else {
        content_terms.join(" ")
    };
    build_query_or(&cleaned)
}

/// Build an FTS5 query for evaluation / recall mode.
/// Identical tokenization to `build_query` but positive terms are ORed, not ANDed.
/// Required for BEIR-style evaluation where queries are full questions — AND semantics
/// force all stop words to match and nearly nothing passes.
pub fn build_query_or(input: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut neg_parts: Vec<String> = Vec::new();

    let mut chars = input.chars().peekable();
    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' => {
                chars.next();
            }
            '-' => {
                chars.next();
                let term = read_term(&mut chars);
                if !term.is_empty() {
                    neg_parts.push(format!("NOT \"{}\"", fts5_escape(&term)));
                }
            }
            '"' => {
                chars.next();
                let phrase: String = chars.by_ref().take_while(|&c| c != '"').collect();
                if !phrase.is_empty() {
                    parts.push(format!("\"{}\"", fts5_escape(&phrase)));
                }
            }
            _ => {
                let term = read_term(&mut chars);
                if !term.is_empty() {
                    parts.push(format!("\"{}\"*", fts5_escape(&term)));
                }
            }
        }
    }

    let pos = parts.join(" OR ");
    if neg_parts.is_empty() {
        pos
    } else {
        let neg = neg_parts.join(" ");
        if pos.is_empty() {
            neg
        } else {
            format!("{pos} {neg}")
        }
    }
}

/// Build an FTS5 query from user input.
/// - bare terms become prefix matches: `"term"*`
/// - "quoted phrases" stay as exact: `"phrase"`
/// - -negation becomes `NOT "term"`
/// - all positive terms are ANDed
pub fn build_query(input: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut neg_parts: Vec<String> = Vec::new();

    let mut chars = input.chars().peekable();
    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' => {
                chars.next();
            }
            '-' => {
                chars.next();
                let term = read_term(&mut chars);
                if !term.is_empty() {
                    let escaped = fts5_escape(&term);
                    neg_parts.push(format!("NOT \"{escaped}\""));
                }
            }
            '"' => {
                chars.next(); // consume opening quote
                let phrase: String = chars.by_ref().take_while(|&c| c != '"').collect();
                if !phrase.is_empty() {
                    let escaped = fts5_escape(&phrase);
                    parts.push(format!("\"{escaped}\""));
                }
            }
            _ => {
                let term = read_term(&mut chars);
                if !term.is_empty() {
                    let escaped = fts5_escape(&term);
                    parts.push(format!("\"{escaped}\"*"));
                }
            }
        }
    }

    let pos = parts.join(" AND ");
    if neg_parts.is_empty() {
        pos
    } else {
        let neg = neg_parts.join(" ");
        if pos.is_empty() {
            neg
        } else {
            format!("{pos} {neg}")
        }
    }
}

fn read_term(chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
    chars
        .by_ref()
        .take_while(|&c| !matches!(c, ' ' | '\t'))
        .collect()
}

/// Normalize FTS5 BM25 score to [0, 1].
/// FTS5 returns negative values (more negative = better match).
fn normalize(raw: f64) -> f64 {
    let pos = -raw; // make positive
    pos / (1.0 + pos)
}

pub struct BM25Query<'a> {
    pub fts_query: String,
    pub collection: &'a str,
    pub limit: usize,
    /// Override title column weight in bm25(). None = use default (1.0).
    pub title_weight: Option<f64>,
}

pub fn search(conn: &Connection, q: &BM25Query) -> Result<Vec<SearchResult>> {
    let fts = &q.fts_query;
    if fts.is_empty() {
        return Ok(vec![]);
    }

    // ! FTS5 bm25() args must be SQL literals — dynamic SQL required for non-default title_weight.
    if let Some(tw) = q.title_weight {
        let sql = format!(
            "SELECT d.path, d.title, bm25(documents_fts, 10.0, {tw}, 1.0) AS score,
               d.hash, snippet(documents_fts, 2, '<b>', '</b>', '...', 32) AS snip
        FROM documents_fts
        JOIN documents d ON documents_fts.rowid = d.id
        WHERE documents_fts MATCH ?1
          AND d.active = 1
        ORDER BY score ASC
        LIMIT ?2"
        );
        let mut stmt = conn.prepare(&sql)?;
        return collect_search_rows(&mut stmt, fts, q);
    }

    // filepath weighted 10x higher than body, title 1x.
    let mut stmt = conn.prepare_cached(
        "SELECT d.path, d.title, bm25(documents_fts, 10.0, 1.0, 1.0) AS score,
               d.hash, snippet(documents_fts, 2, '<b>', '</b>', '...', 32) AS snip
        FROM documents_fts
        JOIN documents d ON documents_fts.rowid = d.id
        WHERE documents_fts MATCH ?1
          AND d.active = 1
        ORDER BY score ASC
        LIMIT ?2",
    )?;
    collect_search_rows(&mut stmt, fts, q)
}

fn collect_search_rows(
    stmt: &mut rusqlite::Statement<'_>,
    fts: &str,
    q: &BM25Query<'_>,
) -> Result<Vec<SearchResult>> {
    let rows = stmt.query_map(rusqlite::params![fts, q.limit as i64], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, f64>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, Option<String>>(4)?,
        ))
    })?;

    let mut results = Vec::new();
    for row in rows {
        let (path, title, raw_score, hash, snippet) = row?;
        let doc_id = format!("#{}", &hash[..6.min(hash.len())]);
        results.push(SearchResult {
            collection: q.collection.to_string(),
            path,
            title,
            score: normalize(raw_score),
            snippet,
            hash,
            doc_id,
            content: None,
            chunk_seq: None,
            unit_seq: None,
            unit_kind: None,
            language: None,
            symbol: None,
            start_line: None,
            end_line: None,
            start_byte: None,
            end_byte: None,
            indexed_hash: None,
            indexed_at: None,
            markers: Vec::new(),
            related: Vec::new(),
        });
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_terms_become_prefix_and() {
        let q = build_query("rust memory");
        assert!(q.contains("\"rust\"*"), "expected prefix match for 'rust'");
        assert!(
            q.contains("\"memory\"*"),
            "expected prefix match for 'memory'"
        );
        assert!(q.contains(" AND "), "terms should be ANDed");
    }

    #[test]
    fn quoted_phrase_stays_exact() {
        assert_eq!(build_query("\"exact phrase\""), "\"exact phrase\"");
    }

    #[test]
    fn negation_produces_not() {
        let q = build_query("good -bad");
        assert!(q.contains("\"good\"*"));
        assert!(q.contains("NOT \"bad\""));
    }

    #[test]
    fn empty_and_whitespace_return_empty() {
        assert_eq!(build_query(""), "");
        assert_eq!(build_query("   "), "");
    }

    #[test]
    fn embedded_quote_is_escaped() {
        // A bare term containing `"` must not break FTS5 query syntax.
        // read_term stops at whitespace, so `rust"lang` becomes a single token.
        let q = build_query("rust\"lang");
        assert!(
            q.contains("\"rust\"\"lang\"*"),
            "inner quote must be doubled: {q}"
        );

        // Same for negated terms.
        let q2 = build_query("-bad\"actor");
        assert!(
            q2.contains("NOT \"bad\"\"actor\""),
            "negated inner quote must be doubled: {q2}"
        );
    }

    #[test]
    fn normalize_maps_negative_fts5_scores() {
        // FTS5 raw score -1.0 → pos=1.0 → 1/(1+1) = 0.5
        assert!((normalize(-1.0) - 0.5).abs() < 1e-10);
        // Large negative score → approaches 1.0
        assert!(normalize(-1000.0) > 0.999);
        // Zero → 0.0
        assert_eq!(normalize(0.0), 0.0);
    }

    // ── build_query_natural tests ────────────────────────────────────────────

    #[test]
    fn natural_short_keyword_query_stays_and() {
        // ≤3 terms, no stop words → unchanged AND behavior
        let q = build_query_natural("rust memory");
        assert!(q.contains("AND"), "short query should keep AND: {q}");
    }

    #[test]
    fn natural_long_question_becomes_or() {
        // Question-format query: stop words stripped, remaining joined with OR
        let q = build_query_natural("what is the best way to invest money");
        assert!(q.contains("OR"), "long question should use OR: {q}");
        assert!(
            !q.contains("\"what\""),
            "stop word 'what' should be stripped: {q}"
        );
        assert!(
            !q.contains("\"is\""),
            "stop word 'is' should be stripped: {q}"
        );
        assert!(
            !q.contains("\"the\""),
            "stop word 'the' should be stripped: {q}"
        );
        assert!(
            q.contains("\"best\""),
            "content term 'best' should remain: {q}"
        );
        assert!(
            q.contains("\"invest\""),
            "content term 'invest' should remain: {q}"
        );
    }

    #[test]
    fn natural_all_stopwords_fallback() {
        // All stop words → don't produce empty query; use OR of originals
        let q = build_query_natural("what is the");
        assert!(!q.is_empty(), "all-stopword query must not be empty: {q}");
        assert!(q.contains("OR"), "all-stopword fallback should use OR: {q}");
    }

    #[test]
    fn natural_negation_preserved() {
        // Negated terms survive regardless of stop word list
        let q = build_query_natural("what is the best way -spam to do something useful");
        assert!(
            q.contains("NOT \"spam\""),
            "negation must be preserved: {q}"
        );
        assert!(q.contains("OR"), "long query should use OR: {q}");
    }

    #[test]
    fn natural_short_with_stopwords_uses_or() {
        // 4+ terms even with stop words → switch to OR
        let q = build_query_natural("what is diabetes symptoms treatment");
        assert!(
            q.contains("OR"),
            "mixed query with 5 terms should use OR: {q}"
        );
    }

    // ── strip_stopwords tests ────────────────────────────────────────────────

    #[test]
    fn strip_removes_stop_words() {
        assert_eq!(strip_stopwords("what is the best way"), "best way");
    }

    #[test]
    fn strip_preserves_content_words() {
        assert_eq!(strip_stopwords("rust memory safety"), "rust memory safety");
    }

    #[test]
    fn strip_all_stopwords_returns_original() {
        // Must not return empty string
        assert_eq!(strip_stopwords("what is the"), "what is the");
    }
}
