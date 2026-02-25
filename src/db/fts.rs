// BM25 full-text search via SQLite FTS5.
// Score normalization: score / (1 + score)  (FTS5 returns negative values; negate first)

use crate::error::Result;
use crate::types::SearchResult;
use rusqlite::Connection;

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
                    neg_parts.push(format!("NOT \"{term}\""));
                }
            }
            '"' => {
                chars.next(); // consume opening quote
                let phrase: String = chars.by_ref().take_while(|&c| c != '"').collect();
                if !phrase.is_empty() {
                    parts.push(format!("\"{phrase}\""));
                }
            }
            _ => {
                let term = read_term(&mut chars);
                if !term.is_empty() {
                    parts.push(format!("\"{term}\"*"));
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
    let rows = stmt.query_map(
        rusqlite::params![fts, q.limit as i64],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
            ))
        },
    )?;

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
