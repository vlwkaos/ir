// Shared YAML frontmatter parsing used by both the indexer and schema migration.
// Lives at the crate root so both `db` and `index` can import it without a circular dep.

/// Find the YAML frontmatter block in `doc`.
/// Returns `(yaml_content, end_byte_offset)` where `end_byte_offset` is the byte position
/// in `doc` immediately after the closing delimiter line (including its newline).
pub(crate) fn find_block(doc: &str) -> Option<(&str, usize)> {
    if !doc.starts_with("---") {
        return None;
    }
    let after_open = doc.get(3..)?;
    // Opening --- must be followed immediately by a newline (no trailing content on the line)
    if !after_open.starts_with('\n') && !after_open.starts_with("\r\n") {
        return None;
    }
    let nl_len = if after_open.starts_with("\r\n") { 2 } else { 1 };
    let content_start = 3 + nl_len;
    let content = &doc[content_start..];

    let mut pos = 0;
    while pos <= content.len() {
        let line_end = content[pos..]
            .find('\n')
            .map(|i| pos + i)
            .unwrap_or(content.len());
        let line = content[pos..line_end].trim_end_matches('\r');
        if line == "---" || line == "..." {
            let yaml = &content[..pos];
            let close_nl = if line_end < content.len() { 1 } else { 0 };
            let end_in_doc = content_start + line_end + close_nl;
            return Some((yaml, end_in_doc));
        }
        if line_end == content.len() {
            break;
        }
        pos = line_end + 1;
    }
    None
}

/// Parse YAML frontmatter from `doc`.
/// Returns the mapping if frontmatter is present and valid; None otherwise.
/// Malformed YAML is logged to stderr — the document is still indexed without metadata.
pub(crate) fn extract(doc: &str) -> Option<yaml_serde::Mapping> {
    let (yaml_block, _) = find_block(doc)?;
    match yaml_serde::from_str::<yaml_serde::Value>(yaml_block) {
        Ok(yaml_serde::Value::Mapping(m)) => Some(m),
        Ok(_) => None,
        Err(e) => {
            eprintln!("warn: malformed frontmatter YAML: {e}");
            None
        }
    }
}

/// Flatten a YAML mapping into `(key, value)` string pairs for `document_metadata` storage.
///
/// - Scalars: stringified (null → skip, bool → "true"/"false", number → decimal, string as-is)
/// - YAML date strings (YYYY-MM-DD or RFC3339+offset) → UTC RFC3339
/// - Sequences: one pair per element (same key, each element stringified)
/// - Nested mappings: skipped with a warning
pub(crate) fn flatten(mapping: &yaml_serde::Mapping) -> Vec<(String, String)> {
    let mut result = Vec::new();
    for (key, value) in mapping {
        let key_str = match key {
            yaml_serde::Value::String(s) => s.clone(),
            _ => continue,
        };
        flatten_value(&key_str, value, &mut result);
    }
    result
}

fn flatten_value(key: &str, value: &yaml_serde::Value, out: &mut Vec<(String, String)>) {
    match value {
        yaml_serde::Value::Null => {}
        yaml_serde::Value::Bool(b) => out.push((key.to_string(), b.to_string())),
        yaml_serde::Value::Number(n) => out.push((key.to_string(), n.to_string())),
        yaml_serde::Value::String(s) => out.push((key.to_string(), normalize_date(s))),
        yaml_serde::Value::Sequence(seq) => {
            for item in seq {
                match item {
                    yaml_serde::Value::String(s) => {
                        out.push((key.to_string(), normalize_date(s)));
                    }
                    yaml_serde::Value::Number(n) => out.push((key.to_string(), n.to_string())),
                    yaml_serde::Value::Bool(b) => out.push((key.to_string(), b.to_string())),
                    _ => {}
                }
            }
        }
        yaml_serde::Value::Mapping(_) => {
            eprintln!("warn: skipping nested map at frontmatter key '{key}'");
        }
        _ => {}
    }
}

/// Normalize `s` to UTC RFC3339 if it looks like a date; returns `s` unchanged otherwise.
pub(crate) fn normalize_date(s: &str) -> String {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return dt.with_timezone(&chrono::Utc).to_rfc3339();
    }
    if let Ok(naive) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
        && let Some(dt) = naive.and_hms_opt(0, 0, 0)
    {
        return chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(dt, chrono::Utc)
            .to_rfc3339();
    }
    s.to_string()
}
