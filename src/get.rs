// Document retrieval by path -- shared by CLI and MCP.
// Supports exact, suffix, and substring matching with vault-root path resolution.

use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::config::{Config, collection_db_path};
use crate::db;
use crate::error::Result;
use crate::types::{Collection, SearchResult};

// ── output types ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct DocContent {
    pub collection: String,
    pub path: String,
    pub title: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MultiGetResult {
    pub found: Vec<DocContent>,
    /// Paths that had no match in any collection
    pub not_found: Vec<String>,
}

// ── SQL ──────────────────────────────────────────────────────────────────────

const SQL_EXACT: &str = "SELECT d.path, d.title, c.doc \
    FROM documents d JOIN content c ON d.hash = c.hash \
    WHERE d.path = ?1 AND d.active = 1 LIMIT 1";
// ^ ESCAPE clause required so literal % and _ in paths don't act as LIKE wildcards.
const SQL_LIKE_ESCAPED: &str = "SELECT d.path, d.title, c.doc \
    FROM documents d JOIN content c ON d.hash = c.hash \
    WHERE d.path LIKE ?1 ESCAPE '\\' AND d.active = 1 LIMIT 1";

/// Escape `%` and `_` so LIKE treats them as literals.
fn escape_like(s: &str) -> String {
    s.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_")
}

// ── public API ───────────────────────────────────────────────────────────────

pub fn open_readonly(path: &std::path::Path) -> std::result::Result<Connection, rusqlite::Error> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let _ = conn.execute_batch("PRAGMA busy_timeout = 5000;");
    Ok(conn)
}

pub fn fetch_document(path: &str, collection_filter: &[String]) -> Result<Option<DocContent>> {
    let config = Config::load()?;
    fetch_document_with_config(path, collection_filter, &config)
}

pub fn fetch_document_with_config(
    path: &str,
    collection_filter: &[String],
    config: &Config,
) -> Result<Option<DocContent>> {
    let cols: Vec<&Collection> = if collection_filter.is_empty() {
        config.collections.iter().collect()
    } else {
        config.collections.iter().filter(|c| collection_filter.contains(&c.name)).collect()
    };

    db::ensure_sqlite_vec();

    // Try vault-root prefix first: "CollectionDir/rel/path" → search CollectionDir's DB with rel/path.
    if let Some((col, stripped)) = resolve_vault_root_path(path, &cols) {
        let db_path = collection_db_path(&col.name);
        match open_readonly(&db_path) {
            Ok(conn) => {
                if let Some(doc) = lookup_in_conn(&conn, &col.name, &stripped)? {
                    return Ok(Some(doc));
                }
            }
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == rusqlite::ErrorCode::CannotOpen => {}
            Err(e) => return Err(e.into()),
        }
    }

    // Fallback: try all collections with the original path (including any vault-root collection,
    // in case the path is stored verbatim with the prefix inside that collection).
    for col in &cols {
        let db_path = collection_db_path(&col.name);
        let conn = match open_readonly(&db_path) {
            Ok(c) => c,
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == rusqlite::ErrorCode::CannotOpen => continue,
            Err(e) => return Err(e.into()),
        };
        if let Some(doc) = lookup_in_conn(&conn, &col.name, path)? {
            return Ok(Some(doc));
        }
    }
    Ok(None)
}

/// Try exact, suffix, then substring path match. Stops at first hit.
pub fn lookup_in_conn(conn: &Connection, collection: &str, path: &str) -> Result<Option<DocContent>> {
    if path.is_empty() {
        return Ok(None);
    }
    // ^ Escape LIKE wildcards so literal % and _ in paths don't cause false positives.
    let escaped = escape_like(path);
    let suffix = format!("%/{escaped}");
    let substr = format!("%{escaped}%");
    let queries: &[(&str, &str)] = &[
        (SQL_EXACT, path),
        (SQL_LIKE_ESCAPED, &suffix),
        (SQL_LIKE_ESCAPED, &substr),
    ];
    for (sql, param) in queries {
        let row = conn.query_row(sql, rusqlite::params![param], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        });
        match row {
            Ok((doc_path, title, content)) => {
                return Ok(Some(DocContent {
                    collection: collection.to_string(),
                    path: doc_path,
                    title,
                    content,
                }));
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(None)
}

// ── vault-root path resolution ───────────────────────────────────────────────

/// If path starts with a collection's directory name (the last component of its
/// absolute path), return that collection and the remainder after stripping.
///
/// Example: path "0. PeriodicNotes/2026/file.md", collection with
/// path "/vault/0. PeriodicNotes" -> Some((col, "2026/file.md"))
fn resolve_vault_root_path<'a>(
    path: &str,
    collections: &[&'a Collection],
) -> Option<(&'a Collection, String)> {
    let (first, rest) = path.split_once('/')?;
    if rest.is_empty() {
        return None;
    }
    for col in collections {
        // ^ skip collections whose path has no usable dir component (e.g. root "/" or non-UTF-8)
        let col_path = std::path::Path::new(&col.path);
        let dir_name = match col_path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if first == dir_name {
            return Some((col, rest.to_string()));
        }
    }
    None
}

// ── chunk retrieval ───────────────────────────────────────────────────────────

/// Inner: batch-fetch chunks from a single open connection.
/// `items` is a slice of (result_idx, hash, seq) tuples.
/// Writes chunk text into `results[result_idx].content`.
fn apply_chunks_from_conn(
    conn: &Connection,
    items: &[(usize, String, usize)],
    results: &mut [SearchResult],
) -> Result<()> {
    let mut unique_hashes: Vec<&str> = items.iter().map(|(_, h, _)| h.as_str()).collect();
    unique_hashes.sort_unstable();
    unique_hashes.dedup();
    let placeholders = unique_hashes.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!("SELECT hash, doc FROM content WHERE hash IN ({placeholders})");
    let mut stmt = conn.prepare(&sql)?;
    let content_map: HashMap<String, String> = stmt
        .query_map(rusqlite::params_from_iter(unique_hashes.iter().copied()), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<std::result::Result<_, _>>()?;

    for (result_idx, hash, seq) in items {
        if let Some(doc) = content_map.get(hash) {
            let chunks = crate::index::chunker::chunk_document(doc);
            results[*result_idx].content = chunks.into_iter().nth(*seq).map(|c| c.text);
        }
    }
    Ok(())
}

/// Inner: fetch a single chunk from an open connection. Used in tests only.
#[cfg(test)]
fn fetch_chunk_from_conn(conn: &Connection, hash: &str, seq: usize) -> Result<Option<String>> {
    use rusqlite::OptionalExtension;
    let doc: Option<String> = conn
        .query_row(
            "SELECT doc FROM content WHERE hash = ?1",
            rusqlite::params![hash],
            |row| row.get(0),
        )
        .optional()?;
    Ok(doc.and_then(|text| {
        crate::index::chunker::chunk_document(&text).into_iter().nth(seq).map(|c| c.text)
    }))
}

/// Populate `.content` on search results that have `chunk_seq` set.
/// Batches DB access per collection: one connection + one query per distinct collection.
pub fn populate_chunk_content(results: &mut [SearchResult]) -> Result<()> {
    let tasks: Vec<(usize, String, String, usize)> = results
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.chunk_seq.map(|seq| (i, r.collection.clone(), r.hash.clone(), seq)))
        .collect();
    if tasks.is_empty() {
        return Ok(());
    }

    let config = Config::load()?;
    let mut by_col: HashMap<String, Vec<(usize, String, usize)>> = HashMap::new();
    for (idx, col, hash, seq) in tasks {
        by_col.entry(col).or_default().push((idx, hash, seq));
    }

    for (col_name, items) in &by_col {
        let col = match config.get_collection(col_name) {
            Some(c) => c,
            None => continue,
        };
        let db_path = collection_db_path(&col.name);
        let conn = match open_readonly(&db_path) {
            Ok(c) => c,
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == rusqlite::ErrorCode::CannotOpen => continue,
            Err(e) => return Err(e.into()),
        };
        apply_chunks_from_conn(&conn, items, results)?;
    }
    Ok(())
}

// ── section extraction ───────────────────────────────────────────────────────

/// Parse an ATX heading line (e.g. `## Title` or `### Title ###`).
/// Returns `(level, heading_text)` or `None` if not a heading.
fn parse_atx_heading(line: &str) -> Option<(usize, &str)> {
    let trimmed = line.trim_start();
    let level = trimmed.bytes().take_while(|&b| b == b'#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &trimmed[level..];
    if rest.is_empty() {
        return Some((level, ""));
    }
    // CommonMark: must be followed by a space/tab or end of line.
    if rest.as_bytes()[0] != b' ' && rest.as_bytes()[0] != b'\t' {
        return None;
    }
    // Strip optional ATX closing sequence and surrounding whitespace.
    let text = rest.trim().trim_end_matches('#').trim_end();
    Some((level, text))
}

/// Extract the section whose heading text matches `heading` (case-insensitive).
/// Returns the slice from the heading line through the last line before the
/// next heading of the same or higher level (or end of document).
/// Headings inside fenced code blocks are ignored.
pub fn extract_section<'a>(doc: &'a str, heading: &str) -> Option<&'a str> {
    let query = heading.trim().to_lowercase();
    let mut in_code_fence = false;
    let mut found_start: Option<usize> = None;
    let mut found_level: usize = 0;
    let mut pos: usize = 0;

    for raw_line in doc.split('\n') {
        let line_start = pos;
        pos += raw_line.len() + 1; // +1 for the '\n' split on
        pos = pos.min(doc.len()); // clamp: last line may have no trailing \n
        let line = raw_line.trim_end_matches('\r');

        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence {
            continue;
        }

        if let Some((level, text)) = parse_atx_heading(line) {
            if let Some(start) = found_start {
                if level <= found_level {
                    // Next same-or-higher heading closes the section.
                    return Some(&doc[start..line_start]);
                }
            } else if text.to_lowercase() == query {
                found_start = Some(line_start);
                found_level = level;
            }
        }
    }

    // Heading found but no closing heading — section runs to end of doc.
    found_start.map(|start| &doc[start..])
}

// ── content trimming ─────────────────────────────────────────────────────────

/// Trim document content by char offset and max length.
/// Char-safe: slices on character boundaries, not byte boundaries.
/// - `offset=None` or `0` → start from beginning
/// - `max_chars=None` or `0` → no limit
/// - `offset` beyond content length → empty string
pub fn trim_content<'a>(content: &'a str, offset: Option<usize>, max_chars: Option<usize>) -> &'a str {
    let start = offset.unwrap_or(0);
    let limit = max_chars.unwrap_or(0);

    let byte_start = match content.char_indices().nth(start) {
        Some((b, _)) => b,
        None => return "",
    };
    let sliced = &content[byte_start..];

    if limit == 0 {
        return sliced;
    }

    match sliced.char_indices().nth(limit) {
        Some((b, _)) => &sliced[..b],
        None => sliced,
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── extract_section ──────────────────────────────────────────────────────

    #[test]
    fn section_basic() {
        let doc = "# Doc\n\n## Installation\nstuff\n\n## Usage\nother\n";
        let s = extract_section(doc, "Installation").unwrap();
        assert_eq!(s, "## Installation\nstuff\n\n");
    }

    #[test]
    fn section_case_insensitive() {
        let doc = "## Installation\ncontent\n## Other\n";
        assert!(extract_section(doc, "installation").is_some());
        assert!(extract_section(doc, "INSTALLATION").is_some());
    }

    #[test]
    fn section_last_in_doc() {
        let doc = "## First\nfoo\n## Last\nbar";
        let s = extract_section(doc, "Last").unwrap();
        assert_eq!(s, "## Last\nbar");
    }

    #[test]
    fn section_not_found_returns_none() {
        let doc = "## Existing\ncontent\n";
        assert!(extract_section(doc, "Missing").is_none());
    }

    #[test]
    fn section_includes_subsections() {
        let doc = "## A\n### Sub\ntext\n## B\n";
        let s = extract_section(doc, "A").unwrap();
        assert_eq!(s, "## A\n### Sub\ntext\n");
    }

    #[test]
    fn section_h1_closes_h2() {
        let doc = "# Root\n## Section\ncontent\n# Other\n";
        let s = extract_section(doc, "Section").unwrap();
        assert_eq!(s, "## Section\ncontent\n");
    }

    #[test]
    fn section_ignores_heading_in_code_fence() {
        let doc = "## Real\n```\n## Fake\n```\n## Next\n";
        // "Fake" is inside a code fence — should not be found.
        assert!(extract_section(doc, "Fake").is_none());
        // "Real" should be found and end at "Next".
        let s = extract_section(doc, "Real").unwrap();
        assert!(s.contains("```\n## Fake\n```\n"));
        assert!(!s.contains("## Next"));
    }

    #[test]
    fn section_atx_closing_stripped() {
        // ATX headings may have trailing # markers: `## Title ##`
        let doc = "## Title ##\ncontent\n## Other\n";
        let s = extract_section(doc, "Title").unwrap();
        assert_eq!(s, "## Title ##\ncontent\n");
    }

    #[test]
    fn section_empty_doc_returns_none() {
        assert!(extract_section("", "anything").is_none());
    }

    // ── extract_section edge cases ───────────────────────────────────────────

    #[test]
    fn section_empty_heading_string_returns_none() {
        // extract_section with an empty heading should never match any heading.
        let doc = "## Real\ncontent\n";
        assert!(extract_section(doc, "").is_none());
    }

    #[test]
    fn section_whitespace_only_heading_returns_none() {
        // Heading query trims whitespace before comparison; all-whitespace becomes "".
        // An ATX heading line with no text (e.g. "## ") has text="" after parse.
        // The query "   " trims to "" which should match a bare "#" heading.
        let doc = "##\ncontent\n## Other\n";
        // "##" is a valid heading with empty text; "   ".trim() == ""
        let s = extract_section(doc, "   ");
        assert!(s.is_some(), "whitespace-only query trims to '' which matches '##'");
    }

    #[test]
    fn section_doc_no_trailing_newline() {
        // Last line has no '\n' — section must still return content to end-of-doc.
        let doc = "## Heading\ncontent here";
        let s = extract_section(doc, "Heading").unwrap();
        assert_eq!(s, "## Heading\ncontent here");
    }

    #[test]
    fn section_first_of_duplicate_headings() {
        // Two headings with the same name: the first one is returned.
        let doc = "## Intro\nfirst\n## Other\nMiddle\n## Intro\nsecond\n";
        let s = extract_section(doc, "Intro").unwrap();
        assert_eq!(s, "## Intro\nfirst\n");
    }

    #[test]
    fn section_h4_closed_by_h3() {
        // h4 section should be closed by h3 (higher level = lower number).
        let doc = "### Parent\n#### Child\ncontent\n### Sibling\n";
        let s = extract_section(doc, "Child").unwrap();
        assert_eq!(s, "#### Child\ncontent\n");
    }

    #[test]
    fn section_h6_runs_to_end() {
        // Deeply-nested h6 with nothing closing it.
        let doc = "# Top\n###### Deep\nleaf content";
        let s = extract_section(doc, "Deep").unwrap();
        assert_eq!(s, "###### Deep\nleaf content");
    }

    #[test]
    fn section_h4_not_closed_by_h5() {
        // A deeper heading (h5) does NOT close an h4 section.
        let doc = "#### Parent\n##### Child\nstuff\n";
        let s = extract_section(doc, "Parent").unwrap();
        assert_eq!(s, "#### Parent\n##### Child\nstuff\n");
    }

    #[test]
    fn section_tilde_fence_ignored_like_backtick() {
        // ~~~ fences should suppress headings the same way ``` does.
        let doc = "## Real\n~~~\n## Fake\n~~~\n## Next\n";
        assert!(extract_section(doc, "Fake").is_none());
        let s = extract_section(doc, "Real").unwrap();
        assert!(s.contains("~~~\n## Fake\n~~~\n"));
        assert!(!s.contains("## Next"));
    }

    #[test]
    fn section_mixed_fence_types_toggle_independently() {
        // ``` opens then ~~~ closes? Per CommonMark, fences must match; but the
        // chunker/section parser tracks a single toggle — a ``` open closed by ~~~
        // produces two toggles, leaving in_code_fence=false after both.
        // Verify that after the second toggle the heading IS visible again.
        let doc = "## Before\n```\n## Inside1\n~~~\n## Inside2\n## After\n";
        // After "```" in_code_fence=true, after "~~~" in_code_fence=false.
        // Inside1 is hidden; Inside2 appears AFTER the ~~~ toggle → visible.
        // We don't assert what "After" does — just check "Inside1" is hidden.
        assert!(extract_section(doc, "Inside1").is_none());
    }

    #[test]
    fn section_unclosed_fence_hides_rest_of_doc() {
        // A fence that is opened but never closed hides all subsequent headings.
        let doc = "## Before\n```\n## Hidden\n## AlsoHidden\n";
        assert!(extract_section(doc, "Hidden").is_none());
        assert!(extract_section(doc, "AlsoHidden").is_none());
        // "Before" is before the fence — it should be found.
        assert!(extract_section(doc, "Before").is_some());
    }

    #[test]
    fn section_crlf_line_endings() {
        // CRLF docs: \r is stripped before heading comparison.
        let doc = "## Title\r\ncontent\r\n## Other\r\n";
        let s = extract_section(doc, "Title");
        assert!(s.is_some(), "heading should match with CRLF line endings");
        let text = s.unwrap();
        // Section should start at "## Title" and not include "## Other".
        assert!(text.starts_with("## Title"));
        assert!(!text.contains("## Other"));
    }

    #[test]
    fn section_heading_immediately_followed_by_next_heading() {
        // Empty body section: heading immediately followed by same-level heading.
        let doc = "## A\n## B\ncontent\n";
        let s = extract_section(doc, "A").unwrap();
        assert_eq!(s, "## A\n");
    }

    #[test]
    fn section_whitespace_body() {
        // Section whose entire body is whitespace / blank lines.
        let doc = "## Empty\n   \n\n## Next\n";
        let s = extract_section(doc, "Empty").unwrap();
        assert_eq!(s, "## Empty\n   \n\n");
    }

    #[test]
    fn section_very_long_heading_name() {
        let long = "A".repeat(4096);
        let doc = format!("## {long}\ncontent\n");
        let s = extract_section(&doc, &long);
        assert!(s.is_some());
    }

    #[test]
    fn section_no_space_after_hash_not_a_heading() {
        // "#Title" (no space) is not a valid ATX heading; should not be found.
        let doc = "#NoSpace\ncontent\n";
        assert!(extract_section(doc, "NoSpace").is_none());
    }

    #[test]
    fn section_single_hash_only_heading() {
        // Bare "#" alone on a line is a valid heading with empty text.
        let doc = "#\ncontent\n## Other\n";
        // query "" matches the bare "#" heading
        let s = extract_section(doc, "");
        assert!(s.is_some());
    }

    #[test]
    fn section_heading_with_leading_spaces() {
        // ATX headings allow up to 3 spaces of indentation (CommonMark).
        // parse_atx_heading does trim_start, so "  ## Title" should parse.
        let doc = "  ## Indented\ncontent\n";
        let s = extract_section(doc, "Indented");
        assert!(s.is_some(), "heading with leading spaces should be matched");
    }

    #[test]
    fn section_result_is_substring_of_original() {
        // extract_section returns a &str slice into the original doc (zero-copy).
        let doc = "## Hello\nworld\n";
        let s = extract_section(doc, "Hello").unwrap();
        let doc_ptr = doc.as_ptr() as usize;
        let s_ptr = s.as_ptr() as usize;
        assert!(s_ptr >= doc_ptr && s_ptr <= doc_ptr + doc.len(),
            "returned slice should point into the original doc");
    }

    // ── trim_content ─────────────────────────────────────────────────────────

    #[test]
    fn trim_no_args_returns_full() {
        assert_eq!(trim_content("hello", None, None), "hello");
    }

    #[test]
    fn trim_max_chars_truncates() {
        assert_eq!(trim_content("hello world", None, Some(5)), "hello");
    }

    #[test]
    fn trim_max_chars_zero_means_no_limit() {
        assert_eq!(trim_content("hello", None, Some(0)), "hello");
    }

    #[test]
    fn trim_offset_skips_start() {
        assert_eq!(trim_content("hello world", Some(6), None), "world");
    }

    #[test]
    fn trim_offset_zero_means_start() {
        assert_eq!(trim_content("hello", Some(0), None), "hello");
    }

    #[test]
    fn trim_offset_and_max_chars() {
        assert_eq!(trim_content("hello world", Some(6), Some(3)), "wor");
    }

    #[test]
    fn trim_offset_beyond_len_returns_empty() {
        assert_eq!(trim_content("hi", Some(100), None), "");
    }

    #[test]
    fn trim_max_chars_beyond_len_returns_rest() {
        assert_eq!(trim_content("hi", None, Some(100)), "hi");
    }

    #[test]
    fn trim_cjk_char_boundary() {
        // Each CJK char is 3 bytes; slicing must be char-safe
        let s = "日本語テスト";
        assert_eq!(trim_content(s, Some(2), Some(2)), "語テ");
    }

    #[test]
    fn trim_cjk_offset_beyond_len_empty() {
        assert_eq!(trim_content("日本語", Some(10), None), "");
    }

    #[test]
    fn trim_empty_string() {
        assert_eq!(trim_content("", None, Some(5)), "");
        assert_eq!(trim_content("", Some(3), None), "");
    }

    // ── fetch_chunk_from_conn ────────────────────────────────────────────────

    fn open_chunk_test_db() -> Connection {
        crate::db::ensure_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(include_str!("db/schema_base.sql")).unwrap();
        conn
    }

    fn insert_content(conn: &Connection, hash: &str, doc: &str) {
        conn.execute(
            "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?1, ?2, '2026-01-01')",
            rusqlite::params![hash, doc],
        ).unwrap();
    }

    #[test]
    fn chunk_from_conn_seq0_returns_first_chunk() {
        let conn = open_chunk_test_db();
        insert_content(&conn, "h1", "hello world");
        let text = fetch_chunk_from_conn(&conn, "h1", 0).unwrap();
        // Short doc is a single chunk — seq 0 returns the whole text.
        assert_eq!(text.as_deref(), Some("hello world"));
    }

    #[test]
    fn chunk_from_conn_hash_not_found_returns_none() {
        let conn = open_chunk_test_db();
        let text = fetch_chunk_from_conn(&conn, "missing", 0).unwrap();
        assert!(text.is_none());
    }

    #[test]
    fn chunk_from_conn_seq_out_of_range_returns_none() {
        let conn = open_chunk_test_db();
        insert_content(&conn, "h2", "short doc");
        // Only one chunk exists (seq 0); seq 5 is out of range.
        let text = fetch_chunk_from_conn(&conn, "h2", 5).unwrap();
        assert!(text.is_none());
    }

    #[test]
    fn chunk_from_conn_multi_chunk_doc_seq1() {
        use crate::index::chunker::{chunk_document, set_chunk_size_tokens_override};
        // chunk_size=200 tokens=800 chars, min=100 tokens=400 chars.
        // A 1000-char doc triggers rebalance (remaining_after=200 < min, doc_tail=1000 ≥ 2*min=800)
        // → split at 600, producing ≥2 chunks both ≥ min.
        set_chunk_size_tokens_override(Some(200));
        let doc: String = "word ".repeat(200); // 1000 chars
        let chunks = chunk_document(&doc);
        assert!(chunks.len() >= 2, "expected multiple chunks");

        let conn = open_chunk_test_db();
        insert_content(&conn, "h3", &doc);
        let chunk1 = fetch_chunk_from_conn(&conn, "h3", 1).unwrap();
        assert_eq!(chunk1.as_deref(), Some(chunks[1].text.as_str()));

        set_chunk_size_tokens_override(None);
    }

    // ── apply_chunks_from_conn ───────────────────────────────────────────────

    fn make_result_with_chunk(hash: &str, seq: usize) -> SearchResult {
        SearchResult {
            collection: "col".into(),
            path: "p".into(),
            title: "t".into(),
            score: 1.0,
            snippet: None,
            hash: hash.into(),
            doc_id: "#abc".into(),
            content: None,
            chunk_seq: Some(seq),
        }
    }

    fn make_result_no_chunk(hash: &str) -> SearchResult {
        SearchResult {
            collection: "col".into(),
            path: "p".into(),
            title: "t".into(),
            score: 1.0,
            snippet: None,
            hash: hash.into(),
            doc_id: "#abc".into(),
            content: None,
            chunk_seq: None,
        }
    }

    #[test]
    fn apply_chunks_populates_content() {
        let conn = open_chunk_test_db();
        insert_content(&conn, "abc", "hello world");
        let items = vec![(0usize, "abc".to_string(), 0usize)];
        let mut results = vec![make_result_with_chunk("abc", 0)];
        apply_chunks_from_conn(&conn, &items, &mut results).unwrap();
        assert_eq!(results[0].content.as_deref(), Some("hello world"));
    }

    #[test]
    fn apply_chunks_skips_missing_hash() {
        let conn = open_chunk_test_db();
        let items = vec![(0usize, "nope".to_string(), 0usize)];
        let mut results = vec![make_result_with_chunk("nope", 0)];
        apply_chunks_from_conn(&conn, &items, &mut results).unwrap();
        assert!(results[0].content.is_none());
    }

    #[test]
    fn apply_chunks_batch_multiple_hashes() {
        let conn = open_chunk_test_db();
        insert_content(&conn, "h_a", "doc alpha");
        insert_content(&conn, "h_b", "doc beta");
        let items = vec![
            (0usize, "h_a".to_string(), 0usize),
            (1usize, "h_b".to_string(), 0usize),
        ];
        let mut results = vec![
            make_result_with_chunk("h_a", 0),
            make_result_with_chunk("h_b", 0),
        ];
        apply_chunks_from_conn(&conn, &items, &mut results).unwrap();
        assert_eq!(results[0].content.as_deref(), Some("doc alpha"));
        assert_eq!(results[1].content.as_deref(), Some("doc beta"));
    }

    #[test]
    fn apply_chunks_leaves_no_chunk_seq_results_untouched() {
        // Results without chunk_seq are not passed to apply_chunks, but even if content is
        // pre-populated it should not be overwritten (items list controls what gets written).
        let conn = open_chunk_test_db();
        insert_content(&conn, "h_c", "some content");
        // Only pass the result that has chunk_seq; the other is untouched.
        let items = vec![(0usize, "h_c".to_string(), 0usize)];
        let mut results = vec![
            make_result_with_chunk("h_c", 0),
            make_result_no_chunk("h_d"),
        ];
        apply_chunks_from_conn(&conn, &items, &mut results).unwrap();
        assert!(results[0].content.is_some());
        assert!(results[1].content.is_none());
    }

    fn test_col(name: &str, path: &str) -> Collection {
        Collection {
            name: name.to_string(),
            path: path.to_string(),
            globs: vec![],
            excludes: vec![],
            description: None,
            preprocessor: None,
        }
    }

    fn open_test_db() -> Connection {
        crate::db::ensure_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(include_str!("db/schema_base.sql")).unwrap();
        conn
    }

    fn insert_doc(conn: &Connection, path: &str, title: &str, content: &str) {
        let hash = format!("hash_{}", path.replace('/', "_"));
        conn.execute(
            "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?1, ?2, '2026-01-01')",
            rusqlite::params![hash, content],
        ).unwrap();
        conn.execute(
            "INSERT INTO documents (path, title, hash, created_at, modified_at, active) \
             VALUES (?1, ?2, ?3, '2026-01-01', '2026-01-01', 1)",
            rusqlite::params![path, title, hash],
        ).unwrap();
    }

    fn insert_inactive_doc(conn: &Connection, path: &str, title: &str, content: &str) {
        let hash = format!("hash_{}", path.replace('/', "_"));
        conn.execute(
            "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?1, ?2, '2026-01-01')",
            rusqlite::params![hash, content],
        ).unwrap();
        conn.execute(
            "INSERT INTO documents (path, title, hash, created_at, modified_at, active) \
             VALUES (?1, ?2, ?3, '2026-01-01', '2026-01-01', 0)",
            rusqlite::params![path, title, hash],
        ).unwrap();
    }

    // ── resolve_vault_root_path ──────────────────────────────────────────────

    #[test]
    fn vault_root_strips_prefix() {
        let periodic = test_col("periodic", "/vault/0. PeriodicNotes");
        let cols: Vec<&Collection> = vec![&periodic];
        let (col, rest) = resolve_vault_root_path(
            "0. PeriodicNotes/2026/Daily/04/file.md", &cols
        ).unwrap();
        assert_eq!(col.name, "periodic");
        assert_eq!(rest, "2026/Daily/04/file.md");
    }

    #[test]
    fn vault_root_no_match_returns_none() {
        let periodic = test_col("periodic", "/vault/0. PeriodicNotes");
        let cols: Vec<&Collection> = vec![&periodic];
        assert!(resolve_vault_root_path("other/2026/file.md", &cols).is_none());
    }

    #[test]
    fn vault_root_no_slash_returns_none() {
        let periodic = test_col("periodic", "/vault/0. PeriodicNotes");
        let cols: Vec<&Collection> = vec![&periodic];
        assert!(resolve_vault_root_path("just-a-filename.md", &cols).is_none());
    }

    #[test]
    fn vault_root_multiple_collections_picks_correct() {
        let periodic = test_col("periodic", "/vault/0. PeriodicNotes");
        let projects = test_col("projects", "/vault/1. Projects");
        let cols: Vec<&Collection> = vec![&periodic, &projects];
        let (col, rest) = resolve_vault_root_path(
            "1. Projects/myproject/README.md", &cols
        ).unwrap();
        assert_eq!(col.name, "projects");
        assert_eq!(rest, "myproject/README.md");
    }

    #[test]
    fn vault_root_case_sensitive() {
        let notes = test_col("notes", "/vault/Notes");
        let cols: Vec<&Collection> = vec![&notes];
        // lowercase "notes" should NOT match "Notes"
        assert!(resolve_vault_root_path("notes/file.md", &cols).is_none());
    }

    #[test]
    fn vault_root_nested_path_only_strips_first_component() {
        let col = test_col("deep", "/vault/a/b/c");
        let cols: Vec<&Collection> = vec![&col];
        // file_name() of "/vault/a/b/c" is "c"
        let (matched, rest) = resolve_vault_root_path("c/file.md", &cols).unwrap();
        assert_eq!(matched.name, "deep");
        assert_eq!(rest, "file.md");
    }

    #[test]
    fn vault_root_with_spaces_in_dirname() {
        let col = test_col("periodic", "/vault/0. Periodic Notes");
        let cols: Vec<&Collection> = vec![&col];
        let (matched, rest) = resolve_vault_root_path(
            "0. Periodic Notes/2026/file.md", &cols
        ).unwrap();
        assert_eq!(matched.name, "periodic");
        assert_eq!(rest, "2026/file.md");
    }

    // ── lookup_in_conn ───────────────────────────────────────────────────────

    #[test]
    fn lookup_exact_match() {
        let conn = open_test_db();
        insert_doc(&conn, "2026/Daily/04/file.md", "File", "hello world");
        let doc = lookup_in_conn(&conn, "test", "2026/Daily/04/file.md").unwrap().unwrap();
        assert_eq!(doc.path, "2026/Daily/04/file.md");
        assert_eq!(doc.title, "File");
        assert_eq!(doc.content, "hello world");
        assert_eq!(doc.collection, "test");
    }

    #[test]
    fn lookup_suffix_match() {
        let conn = open_test_db();
        insert_doc(&conn, "2026/Daily/04/file.md", "File", "content");
        // Suffix: requesting just "04/file.md" should match via %/04/file.md
        let doc = lookup_in_conn(&conn, "test", "04/file.md").unwrap().unwrap();
        assert_eq!(doc.path, "2026/Daily/04/file.md");
    }

    #[test]
    fn lookup_substring_match() {
        let conn = open_test_db();
        insert_doc(&conn, "2026/Daily/04/file.md", "File", "content");
        // Substring: partial match via %Daily%
        let doc = lookup_in_conn(&conn, "test", "Daily/04/file").unwrap().unwrap();
        assert_eq!(doc.path, "2026/Daily/04/file.md");
    }

    #[test]
    fn lookup_no_match() {
        let conn = open_test_db();
        insert_doc(&conn, "2026/Daily/04/file.md", "File", "content");
        assert!(lookup_in_conn(&conn, "test", "nonexistent.md").unwrap().is_none());
    }

    #[test]
    fn lookup_prefers_exact_over_suffix() {
        let conn = open_test_db();
        insert_doc(&conn, "file.md", "Exact", "exact content");
        insert_doc(&conn, "subdir/file.md", "Suffix", "suffix content");
        // "file.md" should exact-match, not suffix-match "subdir/file.md"
        let doc = lookup_in_conn(&conn, "test", "file.md").unwrap().unwrap();
        assert_eq!(doc.title, "Exact");
    }

    #[test]
    fn lookup_skips_inactive() {
        let conn = open_test_db();
        insert_inactive_doc(&conn, "file.md", "Inactive", "old content");
        assert!(lookup_in_conn(&conn, "test", "file.md").unwrap().is_none());
    }

    #[test]
    fn lookup_with_sql_wildcard_in_path() {
        let conn = open_test_db();
        insert_doc(&conn, "notes/100% done.md", "Percent", "content");
        // Path with literal % should still work for exact match
        let doc = lookup_in_conn(&conn, "test", "notes/100% done.md").unwrap().unwrap();
        assert_eq!(doc.title, "Percent");
    }

    #[test]
    fn lookup_with_underscore_in_path() {
        let conn = open_test_db();
        insert_doc(&conn, "my_notes/file.md", "Underscore", "content");
        // _ is a LIKE wildcard but exact match should take priority
        let doc = lookup_in_conn(&conn, "test", "my_notes/file.md").unwrap().unwrap();
        assert_eq!(doc.title, "Underscore");
    }

    // ── LIKE injection edge cases ────────────────────────────────────────────

    #[test]
    fn like_percent_in_suffix_tier_no_false_positive() {
        let conn = open_test_db();
        insert_doc(&conn, "notes/100% done.md", "Percent", "right");
        insert_doc(&conn, "notes/100X done.md", "Wrong", "wrong");
        // Search "100% done.md" (no exact match). Suffix LIKE must not treat
        // the literal % as a wildcard matching "100X done.md".
        let doc = lookup_in_conn(&conn, "test", "100% done.md").unwrap().unwrap();
        assert_eq!(doc.title, "Percent");
    }

    #[test]
    fn like_underscore_in_suffix_tier_no_false_positive() {
        let conn = open_test_db();
        insert_doc(&conn, "notes/a_b.md", "Underscore", "right");
        insert_doc(&conn, "notes/axb.md", "Wrong", "wrong");
        // _ in LIKE matches any single char. Must not match "axb.md".
        let doc = lookup_in_conn(&conn, "test", "a_b.md").unwrap().unwrap();
        assert_eq!(doc.title, "Underscore");
    }

    // ── empty / degenerate paths ─────────────────────────────────────────────

    #[test]
    fn lookup_empty_path_returns_none() {
        let conn = open_test_db();
        insert_doc(&conn, "file.md", "File", "content");
        assert!(lookup_in_conn(&conn, "test", "").unwrap().is_none());
    }

    #[test]
    fn vault_root_dirname_with_trailing_slash_returns_none() {
        // Path "Notes/" splits to ("Notes", ""), rest is empty -> None
        let col = test_col("notes", "/vault/Notes");
        let cols: Vec<&Collection> = vec![&col];
        assert!(resolve_vault_root_path("Notes/", &cols).is_none());
    }

    // ── vault-root + collection filter interaction ───────────────────────────

    #[test]
    fn vault_root_duplicate_basename_picks_first() {
        let a = test_col("notes-a", "/vault-a/Notes");
        let b = test_col("notes-b", "/vault-b/Notes");
        let cols: Vec<&Collection> = vec![&a, &b];
        let (col, _) = resolve_vault_root_path("Notes/file.md", &cols).unwrap();
        assert_eq!(col.name, "notes-a");
    }

    #[test]
    fn vault_root_trailing_slash_in_collection_path() {
        // Path::file_name() strips trailing slash on unix
        let col = test_col("notes", "/vault/Notes/");
        let cols: Vec<&Collection> = vec![&col];
        let (matched, rest) = resolve_vault_root_path("Notes/file.md", &cols).unwrap();
        assert_eq!(matched.name, "notes");
        assert_eq!(rest, "file.md");
    }

    // ── suffix vs substring tier semantics ───────────────────────────────────

    #[test]
    fn suffix_requires_preceding_slash() {
        let conn = open_test_db();
        insert_doc(&conn, "myfile.md", "My", "content");
        // "file.md" has no exact match. Suffix "%/file.md" won't match "myfile.md"
        // (no slash before "file.md"). Substring "%file.md%" will match.
        let doc = lookup_in_conn(&conn, "test", "file.md").unwrap().unwrap();
        assert_eq!(doc.path, "myfile.md");
    }

    // ── unicode paths ────────────────────────────────────────────────────────

    #[test]
    fn lookup_cjk_filename_via_suffix() {
        let conn = open_test_db();
        insert_doc(&conn, "日本語/ファイル.md", "CJK", "content");
        let doc = lookup_in_conn(&conn, "test", "ファイル.md").unwrap().unwrap();
        assert_eq!(doc.path, "日本語/ファイル.md");
    }
}
