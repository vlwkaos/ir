use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

use crate::error::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedUnit {
    pub seq: usize,
    pub unit_kind: String,
    pub language: Option<String>,
    pub symbol: Option<String>,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub title: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    Marker,
    Wikilink,
    Markdown,
    FrontmatterRelated,
    FrontmatterAlias,
}

impl LinkKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Marker => "marker",
            Self::Wikilink => "wikilink",
            Self::Markdown => "markdown",
            Self::FrontmatterRelated => "frontmatter_related",
            Self::FrontmatterAlias => "frontmatter_alias",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitLink {
    pub kind: LinkKind,
    pub target: String,
    pub raw: String,
}

pub fn is_code_path(path: &str) -> bool {
    language_for_path(path).is_some()
}

pub fn language_for_path(path: &str) -> Option<&'static str> {
    match std::path::Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
    {
        Some("rs") => Some("rust"),
        Some("py") => Some("python"),
        Some("js") | Some("jsx") => Some("javascript"),
        Some("ts") | Some("tsx") => Some("typescript"),
        Some("go") => Some("go"),
        Some("java") => Some("java"),
        Some("c") | Some("h") => Some("c"),
        Some("cc") | Some("cpp") | Some("cxx") | Some("hpp") => Some("cpp"),
        Some("cs") => Some("csharp"),
        Some("rb") => Some("ruby"),
        Some("php") => Some("php"),
        Some("swift") => Some("swift"),
        Some("kt") | Some("kts") => Some("kotlin"),
        Some("scala") => Some("scala"),
        Some("sh") | Some("bash") | Some("zsh") | Some("fish") => Some("shell"),
        Some("lua") => Some("lua"),
        Some("dart") => Some("dart"),
        Some("ex") | Some("exs") => Some("elixir"),
        Some("erl") | Some("hrl") => Some("erlang"),
        Some("fs") | Some("fsx") => Some("fsharp"),
        Some("clj") | Some("cljs") => Some("clojure"),
        _ => None,
    }
}

pub fn extract_units(path: &str, text: &str) -> Vec<IndexedUnit> {
    if !is_code_path(path) {
        return crate::index::chunker::chunk_document(text)
            .into_iter()
            .map(|chunk| {
                unit_from_range(RangeInput {
                    path,
                    doc: text,
                    seq: chunk.seq,
                    kind: "chunk",
                    language: None,
                    symbol: None,
                    start: chunk.pos,
                    end: chunk.pos + chunk.text.len(),
                })
            })
            .collect();
    }

    let language = language_for_path(path).map(str::to_string);
    let mut starts = find_code_unit_starts(text, language.as_deref());
    if starts.is_empty() {
        return vec![unit_from_range(RangeInput {
            path,
            doc: text,
            seq: 0,
            kind: "file",
            language,
            symbol: None,
            start: 0,
            end: text.len(),
        })];
    }

    starts.sort_by_key(|s| s.byte);
    starts.dedup_by_key(|s| s.byte);
    let mut out = Vec::new();
    for i in 0..starts.len() {
        let start = starts[i].byte;
        let end = starts
            .get(i + 1)
            .map(|s| trim_end_to_line_start(text, s.byte))
            .unwrap_or(text.len());
        if start >= end {
            continue;
        }
        let symbol = starts[i].symbol.clone();
        out.push(unit_from_range(RangeInput {
            path,
            doc: text,
            seq: out.len(),
            kind: &starts[i].kind,
            language: language.clone(),
            symbol: Some(symbol),
            start,
            end,
        }));
    }
    out
}

pub fn store_units(
    conn: &rusqlite::Connection,
    doc_id: i64,
    path: &str,
    hash: &str,
    text: &str,
) -> Result<()> {
    conn.execute("DELETE FROM content_units WHERE hash = ?1", [hash])?;
    conn.execute("DELETE FROM unit_links WHERE source_hash = ?1", [hash])?;

    let indexed_at = chrono::Utc::now().to_rfc3339();
    for unit in &extract_units(path, text) {
        conn.execute(
            "INSERT OR REPLACE INTO content_units
             (hash, seq, document_id, unit_kind, language, symbol, start_byte, end_byte,
              start_line, end_line, title, text, text_hash, indexed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            rusqlite::params![
                hash,
                unit.seq as i64,
                doc_id,
                unit.unit_kind,
                unit.language,
                unit.symbol,
                unit.start_byte as i64,
                unit.end_byte as i64,
                unit.start_line as i64,
                unit.end_line as i64,
                unit.title,
                unit.text,
                crate::index::hasher::hash_bytes(unit.text.as_bytes()),
                indexed_at,
            ],
        )?;

        for link in extract_links(path, unit) {
            conn.execute(
                "INSERT OR IGNORE INTO unit_links
                 (source_hash, source_seq, document_id, kind, target, raw, resolved_document_id, resolved_hash, resolved_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, NULL, NULL)",
                rusqlite::params![
                    hash,
                    unit.seq as i64,
                    doc_id,
                    link.kind.as_str(),
                    link.target,
                    link.raw,
                ],
            )?;
        }
    }

    // Frontmatter links describe the whole document; attach them to the first unit.
    for link in extract_frontmatter_links(text) {
        conn.execute(
            "INSERT OR IGNORE INTO unit_links
             (source_hash, source_seq, document_id, kind, target, raw, resolved_document_id, resolved_hash, resolved_seq)
             VALUES (?1, 0, ?2, ?3, ?4, ?5, NULL, NULL, NULL)",
            rusqlite::params![hash, doc_id, link.kind.as_str(), link.target, link.raw],
        )?;
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct CodeStart {
    byte: usize,
    kind: String,
    symbol: String,
}

fn find_code_unit_starts(text: &str, language: Option<&str>) -> Vec<CodeStart> {
    let mut out = Vec::new();
    for (line_start, line) in line_offsets(text) {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//")
            || trimmed.starts_with("/*")
            || trimmed.starts_with('*')
            || trimmed.starts_with('#')
        {
            continue;
        }
        if let Some((kind, symbol)) = classify_definition(trimmed, language) {
            let def_byte = line_start + (line.len() - trimmed.len());
            out.push(CodeStart {
                byte: leading_comment_start(text, def_byte, language),
                kind,
                symbol,
            });
        }
    }
    out
}

fn leading_comment_start(text: &str, def_byte: usize, language: Option<&str>) -> usize {
    let hash_comments = matches!(
        language,
        Some("python" | "ruby" | "shell") | Some("elixir") | Some("erlang")
    );
    let mut start = def_byte;
    while start > 0 {
        let prev_end = start.saturating_sub(1);
        let prev_start = text[..prev_end].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let line = text[prev_start..prev_end].trim_start();
        let is_comment = line.starts_with("//")
            || line.starts_with("///")
            || line.starts_with("/*")
            || line.starts_with('*')
            || (hash_comments && line.starts_with('#'));
        if is_comment {
            start = prev_start;
            continue;
        }
        if line.trim().is_empty() {
            break;
        }
        break;
    }
    start
}

fn classify_definition(line: &str, language: Option<&str>) -> Option<(String, String)> {
    for (re, kind) in definition_patterns(language) {
        if let Some(caps) = re.captures(line) {
            let symbol = caps
                .get(1)
                .map(|m| m.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("impl")
                .to_string();
            return Some(((*kind).to_string(), symbol));
        }
    }
    None
}

fn definition_patterns(language: Option<&str>) -> &'static [(Regex, &'static str)] {
    fn compile(patterns: &[(&str, &'static str)]) -> Vec<(Regex, &'static str)> {
        patterns
            .iter()
            .map(|(pat, kind)| (Regex::new(pat).expect("valid code symbol regex"), *kind))
            .collect()
    }

    static RUST: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    static PYTHON: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    static GO: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    static JVM: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    static RUBY: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    static PHP: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    static C_LIKE: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    static JS_TS: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    static EMPTY: [(Regex, &str); 0] = [];

    match language {
        Some("rust") => RUST.get_or_init(|| {
            compile(&[
                (r"^(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)", "function"),
                (r"^(?:pub\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)", "struct"),
                (r"^(?:pub\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)", "enum"),
                (r"^(?:pub\s+)?trait\s+([A-Za-z_][A-Za-z0-9_]*)", "trait"),
                (r"^impl(?:<[^>]+>)?\s+([A-Za-z_][A-Za-z0-9_:<>]*)?", "impl"),
                (r"^(?:pub\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)", "module"),
            ])
        }),
        Some("python") => PYTHON.get_or_init(|| {
            compile(&[
                (r"^(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", "function"),
                (r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", "class"),
            ])
        }),
        Some("go") => GO.get_or_init(|| {
            compile(&[
                (r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)", "function"),
                (r"^type\s+([A-Za-z_][A-Za-z0-9_]*)\s+struct", "struct"),
                (r"^type\s+([A-Za-z_][A-Za-z0-9_]*)\s+interface", "interface"),
            ])
        }),
        Some("java") | Some("csharp") => JVM.get_or_init(|| {
            compile(&[
                (r"^(?:public|private|protected|internal|static|final|abstract|\s)+\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", "class"),
                (r"^(?:public|private|protected|internal|static|final|abstract|\s)+\s*interface\s+([A-Za-z_][A-Za-z0-9_]*)", "interface"),
                (r"^(?:public|private|protected|internal|static|final|async|override|virtual|\s)+[\w<>\[\], ?]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", "method"),
            ])
        }),
        Some("ruby") => RUBY.get_or_init(|| {
            compile(&[
                (r"^def\s+([A-Za-z_][A-Za-z0-9_!?=]*)", "function"),
                (r"^class\s+([A-Za-z_][A-Za-z0-9_:]*)", "class"),
                (r"^module\s+([A-Za-z_][A-Za-z0-9_:]*)", "module"),
            ])
        }),
        Some("php") => PHP.get_or_init(|| {
            compile(&[
                (r"^(?:public|private|protected|static|\s)*function\s+([A-Za-z_][A-Za-z0-9_]*)", "function"),
                (r"^(?:abstract\s+|final\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)", "class"),
                (r"^interface\s+([A-Za-z_][A-Za-z0-9_]*)", "interface"),
            ])
        }),
        Some("c") | Some("cpp") => C_LIKE.get_or_init(|| {
            compile(&[
                (r"^(?:template\s*<[^>]+>\s*)?(?:class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)", "type"),
                (r"^[A-Za-z_][\w:<>\s\*&]+?\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*(?:const\s*)?\{?", "function"),
            ])
        }),
        Some("javascript") | Some("typescript") => JS_TS.get_or_init(|| {
            compile(&[
                (r"^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)", "function"),
                (r"^(?:export\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)", "class"),
                (r"^(?:export\s+)?interface\s+([A-Za-z_$][A-Za-z0-9_$]*)", "interface"),
                (r"^(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s*)?\(?", "function"),
                (r"^(?:async\s+)?([A-Za-z_$][A-Za-z0-9_$]*)\s*\([^)]*\)\s*\{", "method"),
            ])
        }),
        _ => &EMPTY,
    }
}

struct RangeInput<'a> {
    path: &'a str,
    doc: &'a str,
    seq: usize,
    kind: &'a str,
    language: Option<String>,
    symbol: Option<String>,
    start: usize,
    end: usize,
}

fn unit_from_range(input: RangeInput<'_>) -> IndexedUnit {
    let start = previous_char_boundary(input.doc, input.start.min(input.doc.len()));
    let end = previous_char_boundary(input.doc, input.end.min(input.doc.len())).max(start);
    let text = input.doc[start..end].to_string();
    let (start_line, end_line) = line_range_for_bytes(input.doc, start, end);
    let title = input
        .symbol
        .clone()
        .unwrap_or_else(|| crate::index::chunker::extract_title(input.doc, input.path));
    IndexedUnit {
        seq: input.seq,
        unit_kind: input.kind.to_string(),
        language: input.language,
        symbol: input.symbol,
        start_byte: start,
        end_byte: end,
        start_line,
        end_line,
        title,
        text,
    }
}

fn previous_char_boundary(s: &str, mut idx: usize) -> usize {
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn line_range_for_bytes(doc: &str, start: usize, end: usize) -> (usize, usize) {
    let start_line = doc[..start.min(doc.len())]
        .bytes()
        .filter(|b| *b == b'\n')
        .count()
        + 1;
    let end_line = doc[..end.min(doc.len())]
        .bytes()
        .filter(|b| *b == b'\n')
        .count()
        + 1;
    (start_line, end_line.max(start_line))
}

fn line_offsets(text: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    for line in text.split_inclusive('\n') {
        out.push((pos, line.trim_end_matches('\n').trim_end_matches('\r')));
        pos += line.len();
    }
    if text.is_empty() {
        out.push((0, ""));
    }
    out
}

fn trim_end_to_line_start(text: &str, byte: usize) -> usize {
    let mut pos = byte.min(text.len());
    while pos > 0 && text.as_bytes().get(pos - 1) != Some(&b'\n') {
        pos -= 1;
    }
    pos
}

pub fn extract_links(path: &str, unit: &IndexedUnit) -> Vec<UnitLink> {
    let text = if is_code_path(path) {
        comments_only(&unit.text, unit.language.as_deref())
    } else {
        unit.text.clone()
    };
    let mut links = Vec::new();
    links.extend(extract_markers(&text));
    links.extend(extract_wikilinks(&text));
    links.extend(extract_markdown_links(&text));
    dedupe_links(links)
}

pub fn extract_frontmatter_links(doc: &str) -> Vec<UnitLink> {
    let mut links = Vec::new();
    let Some(mapping) = crate::frontmatter::extract(doc) else {
        return links;
    };
    for (key, value) in crate::frontmatter::flatten(&mapping) {
        let kind = match key.as_str() {
            "related" => LinkKind::FrontmatterRelated,
            "aliases" | "alias" => LinkKind::FrontmatterAlias,
            _ => continue,
        };
        links.push(UnitLink {
            kind,
            target: normalize_target(&value),
            raw: value,
        });
    }
    dedupe_links(links)
}

fn extract_markers(text: &str) -> Vec<UnitLink> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"\[([a-z][a-z0-9]*(?:-[a-z0-9]+)+)\]").unwrap());
    re.captures_iter(text)
        .filter_map(|caps| {
            let m = caps.get(0)?;
            let after = text[m.end()..].chars().next();
            if matches!(after, Some('(' | '[')) {
                return None;
            }
            let target = caps.get(1)?.as_str().to_string();
            Some(UnitLink {
                kind: LinkKind::Marker,
                target: target.clone(),
                raw: format!("[{target}]"),
            })
        })
        .collect()
}

fn extract_wikilinks(text: &str) -> Vec<UnitLink> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"\[\[([^\]\n]+)\]\]").unwrap());
    re.captures_iter(text)
        .filter_map(|caps| {
            let raw_target = caps.get(1)?.as_str();
            let target = raw_target.split('|').next().unwrap_or(raw_target).trim();
            if target.is_empty() {
                return None;
            }
            Some(UnitLink {
                kind: LinkKind::Wikilink,
                target: normalize_target(target),
                raw: format!("[[{raw_target}]]"),
            })
        })
        .collect()
}

fn extract_markdown_links(text: &str) -> Vec<UnitLink> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"!?\[[^\]\n]*\]\(([^)\n]+)\)").unwrap());
    re.captures_iter(text)
        .filter_map(|caps| {
            let raw = caps.get(0)?.as_str();
            if raw.starts_with('!') {
                return None;
            }
            let target = caps.get(1)?.as_str().trim();
            if target.starts_with("http://")
                || target.starts_with("https://")
                || target.starts_with("mailto:")
                || target.is_empty()
            {
                return None;
            }
            Some(UnitLink {
                kind: LinkKind::Markdown,
                target: normalize_target(target),
                raw: raw.to_string(),
            })
        })
        .collect()
}

fn normalize_target(target: &str) -> String {
    target
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_string()
}

fn dedupe_links(links: Vec<UnitLink>) -> Vec<UnitLink> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for link in links {
        let key = (
            link.kind.as_str().to_string(),
            link.target.clone(),
            link.raw.clone(),
        );
        if seen.insert(key) {
            out.push(link);
        }
    }
    out
}

fn comments_only(text: &str, language: Option<&str>) -> String {
    let mut out = String::new();
    let hash_comments = matches!(
        language,
        Some("python" | "ruby" | "shell") | Some("elixir") | Some("erlang")
    );
    for line in text.lines() {
        let trimmed = line.trim_start();
        let is_comment = trimmed.starts_with("//")
            || trimmed.starts_with("///")
            || trimmed.starts_with("/*")
            || trimmed.starts_with('*')
            || (hash_comments && trimmed.starts_with('#'));
        if is_comment {
            out.push_str(trimmed);
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_parser_ignores_markdown_links() {
        let links = extract_markers("[cache-key] [not-marker](file.md)");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "cache-key");
    }

    #[test]
    fn wikilink_alias_targets_left_side() {
        let links = extract_wikilinks("[[Search pipeline#Shortcut|shortcut]]");
        assert_eq!(links[0].target, "Search pipeline#Shortcut");
    }

    #[test]
    fn markdown_links_keep_local_targets_and_filter_remote_or_images() {
        let links = extract_markdown_links(
            "[local](docs/cache.md#Policy) ![img](image.png) [web](https://example.com) [mail](mailto:a@b.c)",
        );
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].kind, LinkKind::Markdown);
        assert_eq!(links[0].target, "docs/cache.md#Policy");
    }

    #[test]
    fn frontmatter_links_extract_related_and_aliases() {
        let links = extract_frontmatter_links(
            r#"---
related:
  - src/cache.rs#load
aliases:
  - Cache Policy
---
# Note
"#,
        );
        assert_eq!(links.len(), 2);
        assert!(links.iter().any(|link| {
            link.kind == LinkKind::FrontmatterRelated && link.target == "src/cache.rs#load"
        }));
        assert!(links.iter().any(|link| {
            link.kind == LinkKind::FrontmatterAlias && link.target == "Cache Policy"
        }));
    }

    #[test]
    fn link_extraction_dedupes_same_kind_target_and_raw() {
        let unit = IndexedUnit {
            seq: 0,
            unit_kind: "chunk".into(),
            language: None,
            symbol: None,
            start_byte: 0,
            end_byte: 0,
            start_line: 1,
            end_line: 1,
            title: "note".into(),
            text: "[cache-key] [cache-key] [[Cache Note]] [[Cache Note]]".into(),
        };
        let links = extract_links("note.md", &unit);
        assert_eq!(links.len(), 2);
    }

    #[test]
    fn code_links_only_read_comments() {
        let unit = IndexedUnit {
            seq: 0,
            unit_kind: "function".into(),
            language: Some("rust".into()),
            symbol: Some("f".into()),
            start_byte: 0,
            end_byte: 0,
            start_line: 1,
            end_line: 2,
            title: "f".into(),
            text: "fn f() { let s = \"[not-anchor]\"; }\n// [real-anchor]\n".into(),
        };
        let links = extract_links("src/lib.rs", &unit);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "real-anchor");
    }

    #[test]
    fn extracts_rust_and_python_symbols() {
        let rust = extract_units("src/lib.rs", "pub struct Store {}\nfn search_core() {}\n");
        assert_eq!(rust[0].symbol.as_deref(), Some("Store"));
        assert_eq!(rust[1].symbol.as_deref(), Some("search_core"));
        let py = extract_units(
            "tool.py",
            "class Runner:\n    pass\ndef main():\n    pass\n",
        );
        assert_eq!(py[0].symbol.as_deref(), Some("Runner"));
        assert_eq!(py[1].symbol.as_deref(), Some("main"));
    }

    #[test]
    fn leading_comment_marker_attaches_to_code_unit() {
        let units = extract_units(
            "src/search.rs",
            "// [routing-shortcut]\n// policy note\npub fn route_query() {}\n",
        );
        assert_eq!(units.len(), 1);
        assert!(units[0].text.contains("[routing-shortcut]"));
        let links = extract_links("src/search.rs", &units[0]);
        assert_eq!(links[0].target, "routing-shortcut");
    }

    #[test]
    fn mainstream_language_detection_is_broad() {
        let cases = [
            ("main.rs", "rust"),
            ("tool.py", "python"),
            ("app.js", "javascript"),
            ("view.tsx", "typescript"),
            ("main.go", "go"),
            ("App.java", "java"),
            ("lib.c", "c"),
            ("lib.cpp", "cpp"),
            ("Program.cs", "csharp"),
            ("task.rb", "ruby"),
            ("index.php", "php"),
            ("App.swift", "swift"),
            ("Main.kt", "kotlin"),
            ("Job.scala", "scala"),
            ("run.sh", "shell"),
            ("init.lua", "lua"),
            ("main.dart", "dart"),
            ("worker.ex", "elixir"),
            ("server.erl", "erlang"),
            ("core.fs", "fsharp"),
            ("main.clj", "clojure"),
        ];
        for (path, language) in cases {
            assert_eq!(language_for_path(path), Some(language), "{path}");
            assert!(is_code_path(path), "{path}");
        }
    }

    #[test]
    fn unsupported_code_shape_falls_back_to_file_unit() {
        let units = extract_units("App.swift", "import Foundation\nlet value = 1\n");
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].unit_kind, "file");
        assert_eq!(units[0].language.as_deref(), Some("swift"));
    }
}
