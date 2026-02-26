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
}

pub fn search(conn: &Connection, q: &BM25Query) -> Result<Vec<SearchResult>> {
    let fts = &q.fts_query;
    if fts.is_empty() {
        return Ok(vec![]);
    }

    // filepath weighted 10x higher than body, per qmd's bm25(documents_fts, 10.0, 1.0, 1.0)
    let sql = "
        SELECT d.path, d.title, bm25(documents_fts, 10.0, 1.0, 1.0) AS score,
               d.hash, snippet(documents_fts, 2, '<b>', '</b>', '...', 32) AS snip
        FROM documents_fts
        JOIN documents d ON documents_fts.rowid = d.id
        WHERE documents_fts MATCH ?1
          AND d.active = 1
        ORDER BY score ASC
        LIMIT ?2
    ";

    let mut stmt = conn.prepare_cached(sql)?;
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
        let doc_id = format!("#{}", &hash[..6]);
        results.push(SearchResult {
            collection: q.collection.to_string(),
            path,
            title,
            score: normalize(raw_score),
            snippet,
            hash,
            doc_id,
        });
    }
    Ok(results)
}
