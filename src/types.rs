use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoutingConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused_strong_floor: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused_strong_product: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bm25_strong_floor: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bm25_strong_gap: Option<f64>,
}

/// Per-collection (and, at the top level, global) overrides for the resolved
/// retrieval pipeline. Every field is `Option`: `None` defers to the built-in
/// default. Resolution precedence is `config > env > default` — see
/// `search::profile`. Query-time knobs apply only when all searched collections
/// agree on the same value (same rule as `RoutingConfig`); `expander` is resolved
/// globally from the top-level block because the model loads once per process.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RetrievalConfig {
    /// Use the HNSW ANN sidecar for vector kNN (exact fallback when absent/stale).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ann: Option<bool>,
    /// Build `doc_graph` from the ANN index (per-doc self-query) instead of matmul.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_build_from_ann: Option<bool>,
    /// Tier-0 graph expansion: BM25 seeds pull graph neighbors into the candidate list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t0_graph_expand: Option<bool>,
    /// Tier-2 rerank window size (docs sent to the reranker).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank_window: Option<usize>,
    /// Keep the reranked window above the un-judged tail (score-scale fix).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank_keep_window: Option<bool>,
    /// Load and use the in-process LLM query expander (global; top-level block).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expander: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub globs: Vec<String>,
    #[serde(default)]
    pub excludes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Preprocessor aliases to apply (in order) before FTS indexing and query time.
    /// Each alias must be registered in config.preprocessors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preprocessor: Option<Vec<String>>,
    /// Optional per-collection routing overrides for BM25/fused strong-signal shortcuts.
    /// Applies only when all searched collections agree on the same override value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<RoutingConfig>,
    /// Optional per-collection retrieval-pipeline overrides (ANN, graph, rerank window).
    /// Applies only when all searched collections agree; `config > env > default`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval: Option<RetrievalConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub collection: String,
    pub path: String,
    pub title: String,
    pub score: f64,
    pub snippet: Option<String>,
    pub hash: String,
    pub doc_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Best-matching chunk index within the document (set by vector search; None for BM25-only results).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_seq: Option<usize>,
}

impl SearchResult {
    pub fn sort_desc(results: &mut [Self]) {
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

/// Comparison operator for a filter clause.
/// Derives JsonSchema so MCP clients receive typed enum choices in the tool schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FilterOp {
    /// Exact match (case-sensitive): field = value
    #[serde(rename = "=")]
    Eq,
    /// Not equal (case-sensitive): field != value
    #[serde(rename = "!=")]
    Ne,
    /// Greater than (lexicographic; UTC RFC3339 for dates): field > value
    #[serde(rename = ">")]
    Gt,
    /// Greater than or equal: field >= value
    #[serde(rename = ">=")]
    Gte,
    /// Less than: field < value
    #[serde(rename = "<")]
    Lt,
    /// Less than or equal: field <= value
    #[serde(rename = "<=")]
    Lte,
    /// Case-insensitive substring match: field ~ value
    #[serde(rename = "~")]
    Contains,
    /// Case-insensitive not-contains: field !~ value
    #[serde(rename = "!~")]
    NotContains,
}

/// A single filter predicate: `field op value`.
///
/// Built-in fields: `path`, `modified_at`, `created_at` (always present).
/// Frontmatter fields: `meta.<ident>` where ident = `[A-Za-z_][A-Za-z0-9_.\-]*`.
/// Unknown fields → parse error (fail closed). `meta.*` clauses evaluate to false for
/// docs that have no metadata rows (no panic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterClause {
    pub field: String,
    pub op: FilterOp,
    pub value: String,
}

/// AND-only filter: all clauses must match for a document to pass.
/// Empty filter passes all documents (no-op).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Filter {
    #[serde(default)]
    pub clauses: Vec<FilterClause>,
}

impl Filter {
    /// Parse a list of CLI `-f "FIELD OP VALUE"` strings into a Filter.
    ///
    /// Op longest-match: `>=`, `<=`, `!=`, `!~` (2-char) before `>`, `<`, `=`, `~` (1-char).
    /// Date values for `modified_at`, `created_at`, `meta.date` are normalized to UTC RFC3339.
    pub fn parse(inputs: &[String]) -> Result<Self, String> {
        let mut clauses = Vec::new();
        for input in inputs {
            clauses.push(parse_clause(input.trim())?);
        }
        Ok(Self { clauses })
    }

    /// Build a Filter directly from pre-parsed clauses (MCP / daemon path — no re-parse).
    pub fn from_clauses(clauses: Vec<FilterClause>) -> Self {
        Self { clauses }
    }

    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }
}

fn parse_clause(s: &str) -> Result<FilterClause, String> {
    let (op_start, op, op_end) = find_op(s)?;
    let field = s[..op_start].trim().to_string();
    let value = s[op_end..].trim().to_string();

    if field.is_empty() {
        return Err(format!("empty field in filter clause: '{s}'"));
    }
    if value.is_empty() {
        return Err(format!("empty value in filter clause: '{s}'"));
    }

    validate_field(&field)?;
    let value = normalize_filter_value(&field, value)?;
    // Pre-lowercase for case-insensitive ops so match_op allocates only once (actual, not expected).
    let value = if matches!(op, FilterOp::Contains | FilterOp::NotContains) {
        value.to_ascii_lowercase()
    } else {
        value
    };

    Ok(FilterClause { field, op, value })
}

/// Scan left-to-right for the first operator, longest-match (2-char before 1-char).
fn find_op(s: &str) -> Result<(usize, FilterOp, usize), String> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'>' | b'<' | b'=' | b'~' | b'!' => {
                // Try 2-char op first
                if i + 1 < bytes.len() {
                    let op2 = match (bytes[i], bytes[i + 1]) {
                        (b'>', b'=') => Some(FilterOp::Gte),
                        (b'<', b'=') => Some(FilterOp::Lte),
                        (b'!', b'=') => Some(FilterOp::Ne),
                        (b'!', b'~') => Some(FilterOp::NotContains),
                        _ => None,
                    };
                    if let Some(op) = op2 {
                        return Ok((i, op, i + 2));
                    }
                }
                // Single-char op
                let op1 = match bytes[i] {
                    b'>' => Some(FilterOp::Gt),
                    b'<' => Some(FilterOp::Lt),
                    b'=' => Some(FilterOp::Eq),
                    b'~' => Some(FilterOp::Contains),
                    b'!' => {
                        return Err(format!(
                            "invalid operator at position {i}: '!' must be followed by '=' or '~'"
                        ));
                    }
                    _ => None,
                };
                if let Some(op) = op1 {
                    return Ok((i, op, i + 1));
                }
            }
            _ => {}
        }
        i += 1;
    }
    Err(format!(
        "no operator found in '{s}'; expected one of: =, !=, >, >=, <, <=, ~, !~"
    ))
}

fn validate_field(field: &str) -> Result<(), String> {
    match field {
        "path" | "modified_at" | "created_at" => Ok(()),
        f if f.starts_with("meta.") => {
            let ident = &f["meta.".len()..];
            if ident.is_empty() {
                return Err(format!(
                    "invalid field '{f}': 'meta.' requires an identifier suffix"
                ));
            }
            let mut chars = ident.chars();
            let first = chars.next().unwrap();
            if !matches!(first, 'A'..='Z' | 'a'..='z' | '_') {
                return Err(format!(
                    "invalid field '{f}': meta key must start with a letter or underscore"
                ));
            }
            for c in chars {
                if !matches!(c, 'A'..='Z' | 'a'..='z' | '0'..='9' | '_' | '.' | '-') {
                    return Err(format!(
                        "invalid field '{f}': meta key contains invalid character '{c}'"
                    ));
                }
            }
            Ok(())
        }
        _ => Err(format!(
            "unknown field '{field}'; allowed: path, modified_at, created_at, meta.<name>"
        )),
    }
}

/// Normalize date values for date fields to UTC RFC3339.
/// Applies to `modified_at`, `created_at`, and `meta.date`.
/// Other fields are returned as-is.
fn normalize_filter_value(field: &str, value: String) -> Result<String, String> {
    if matches!(field, "modified_at" | "created_at") || field == "meta.date" {
        let normalized = crate::frontmatter::normalize_date(&value);
        if normalized == value {
            return Err(format!(
                "unrecognized date format for '{field}': '{value}'\n  \
                 accepted: YYYY-MM-DD or RFC3339 (e.g. 2024-04-15T10:30:00Z)"
            ));
        }
        return Ok(normalized);
    }
    Ok(value)
}

/// Controls stderr output from `search_core`.
/// - `Quiet`: no stderr at all (MCP stdio transport)
/// - `Normal`: progress indicators + daemon decision logs
/// - `Verbose`: Normal + daemon timing lines
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Verbosity {
    Quiet,
    #[default]
    Normal,
    Verbose,
}

impl Verbosity {
    /// True when progress indicators ("searching...", "enhancing...") should print.
    pub fn show_progress(self) -> bool {
        self != Self::Quiet
    }
    /// True when daemon log lines (decisions, errors) should print.
    pub fn show_logs(self) -> bool {
        self != Self::Quiet
    }
    /// True when the daemon should include timing lines in its log.
    pub fn daemon_verbose(self) -> bool {
        self == Self::Verbose
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    Bm25,
    Vector,
    #[default]
    Hybrid,
}

impl std::str::FromStr for SearchMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bm25" => Ok(Self::Bm25),
            "vector" | "vec" => Ok(Self::Vector),
            "hybrid" => Ok(Self::Hybrid),
            _ => Err(format!("unknown mode '{s}'. Use: bm25, vector, hybrid")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_one(s: &str) -> FilterClause {
        Filter::parse(&[s.to_string()]).unwrap().clauses.remove(0)
    }

    fn parse_err(s: &str) -> String {
        Filter::parse(&[s.to_string()]).unwrap_err()
    }

    #[test]
    fn parse_all_ops() {
        assert_eq!(parse_one("path=foo").op, FilterOp::Eq);
        assert_eq!(parse_one("path!=foo").op, FilterOp::Ne);
        assert_eq!(
            parse_one("modified_at>2024-01-01T00:00:00Z").op,
            FilterOp::Gt
        );
        assert_eq!(
            parse_one("modified_at>=2024-01-01T00:00:00Z").op,
            FilterOp::Gte
        );
        assert_eq!(
            parse_one("modified_at<2024-01-01T00:00:00Z").op,
            FilterOp::Lt
        );
        assert_eq!(
            parse_one("modified_at<=2024-01-01T00:00:00Z").op,
            FilterOp::Lte
        );
        assert_eq!(parse_one("path~notes/").op, FilterOp::Contains);
        assert_eq!(parse_one("path!~archive/").op, FilterOp::NotContains);
    }

    #[test]
    fn date_normalization() {
        let c = parse_one("modified_at>=2024-01-01");
        assert_eq!(c.value, "2024-01-01T00:00:00+00:00");

        let c = parse_one("created_at>=2024-01-15T10:30:00+09:00");
        // 10:30 KST = 01:30 UTC
        assert_eq!(c.value, "2024-01-15T01:30:00+00:00");
    }

    #[test]
    fn meta_field_valid() {
        let c = parse_one("meta.tags=rust");
        assert_eq!(c.field, "meta.tags");
        assert_eq!(c.value, "rust");

        let c = parse_one("meta.author=vlwkaos");
        assert_eq!(c.field, "meta.author");

        parse_one("meta.some-key_1.x=v"); // hyphen, underscore, dot in ident
    }

    #[test]
    fn parse_errors() {
        // Unknown field
        assert!(parse_err("foo=bar").contains("unknown field"));
        // Bad op
        assert!(parse_err("path!bar").contains("must be followed"));
        // Bad date
        assert!(parse_err("modified_at>=notadate").contains("unrecognized date"));
        // Empty value
        assert!(parse_err("path=").contains("empty value"));
        // Empty field
        assert!(parse_err("=foo").contains("empty field"));
        // No op
        assert!(parse_err("pathfoo").contains("no operator"));
        // meta. with no ident
        assert!(parse_err("meta.=foo").contains("requires an identifier suffix"));
    }

    #[test]
    fn injection_values_survive_raw() {
        // SQL injection attempts in value must be stored verbatim (filter uses param binding)
        let c = parse_one("path=o'brien");
        assert_eq!(c.value, "o'brien");

        let c = parse_one("meta.title=a;DROP TABLE");
        assert_eq!(c.value, "a;DROP TABLE");
    }

    #[test]
    fn multiple_clauses() {
        let f = Filter::parse(&[
            "modified_at>=2024-01-01".to_string(),
            "meta.tags=rust".to_string(),
        ])
        .unwrap();
        assert_eq!(f.clauses.len(), 2);
        assert!(!f.is_empty());
    }

    #[test]
    fn empty_filter() {
        assert!(Filter::parse(&[]).unwrap().is_empty());
        assert!(Filter::default().is_empty());
    }
}
