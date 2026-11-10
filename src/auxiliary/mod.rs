// Auxiliary domains: non-document files (source code) indexed beside a
// collection's primary documents. Artifacts never become results; they add
// bounded, one-hop support to the documents that mention them.
// knowledge: bigpicture/Multiscript and Auxiliary Retrieval Roadmap.md (Branch 2)
//
//   update:  code files ─ extract (no lock) ─► artifacts + artifact_fts
//                                  (update txn) ─► artifact_edges (artifact → document)
//   search:  document BM25 hits ─┐
//            artifact BM25 hits ─┴─ score + β·support·(1 − score) ─► documents (+ evidence)

pub mod code;

use crate::db::{CollectionDb, fts};
use crate::error::Result;
use crate::index::{hasher, scanner};
use crate::preprocess::PreprocessChain;
use crate::types::{Collection, SearchResult};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::{HashMap, HashSet};

pub const CODE_DOMAIN: &str = "code";

/// Every auxiliary domain ir can index. `--aux` names them; `all` means every one.
pub const DOMAINS: &[&str] = &[CODE_DOMAIN];

/// Domains named by `--aux` / MCP `aux`: none named, or `all`, means every
/// domain; an unknown name is an error that lists the known ones.
pub fn resolve_domains(requested: &[String]) -> std::result::Result<Vec<&'static str>, String> {
    if requested.is_empty() || requested.iter().any(|d| d.trim() == "all") {
        return Ok(DOMAINS.to_vec());
    }
    requested
        .iter()
        .map(|d| {
            DOMAINS
                .iter()
                .find(|known| **known == d.trim())
                .copied()
                .ok_or_else(|| {
                    format!(
                        "unknown aux domain '{d}' (known: {}, all)",
                        DOMAINS.join(", ")
                    )
                })
        })
        .collect()
}

/// Meta key holding the number of stored artifacts. Search skips all
/// auxiliary work when it is absent or 0.
const META_COUNT: &str = "aux_artifacts";

/// Artifact hits considered per collection and query.
const ARTIFACT_HITS: usize = 20;
/// Share of the remaining score headroom an artifact can add: a document at
/// score s with support u ends at s + β·u·(1 − s), always below 1.
const SUPPORT_BETA: f64 = 0.4;
/// Documents an artifact may add that the document BM25 did not return.
const MAX_INJECTED: usize = 10;
/// An identifier mentioned by more documents is a hub and links nothing.
const HUB_DOCUMENTS: usize = 50;
/// A name defined by more artifacts is ambiguous (`new`, `init`) and links nothing.
const MAX_DEFINITIONS: usize = 3;
const EVIDENCE_PER_DOC: usize = 3;

// ── sync ────────────────────────────────────────────────────────────────────

/// Extracted source file waiting to be stored.
pub struct Extracted {
    path: String,
    hash: String,
    artifacts: Vec<code::Artifact>,
    /// artifact_fts body per artifact; preprocessed like documents when the
    /// collection has a preprocessor.
    bodies: Vec<String>,
    /// Documents named in the file's comments (see `code::anchors`).
    anchors: Vec<String>,
}

/// Auxiliary changes of one update, computed before the write lock.
pub struct Prepared {
    enabled: bool,
    /// {path → hash} of every source the scan matched.
    scanned: HashMap<String, String>,
    extracted: Vec<Extracted>,
    pub notes: Vec<String>,
}

/// Scan and extract the collection's code domain. Reads stored sources to
/// skip unchanged files; parsing happens here, outside the write lock.
pub fn prepare(
    conn: &Connection,
    collection: &Collection,
    preprocessor_commands: &[String],
    force: bool,
) -> Result<Prepared> {
    let Some(cfg) = collection
        .auxiliary
        .as_ref()
        .and_then(|a| a.code.as_ref())
        .filter(|c| !c.globs.is_empty())
    else {
        return Ok(Prepared {
            enabled: false,
            scanned: HashMap::new(),
            extracted: vec![],
            notes: vec![],
        });
    };

    let stored = if force {
        HashMap::new()
    } else {
        stored_sources(conn)?
    };
    let mut notes = Vec::new();
    let mut scanned = HashMap::new();
    let mut to_extract = Vec::new();
    let mut unsupported = 0usize;
    for f in scanner::scan_globs(&collection.path, &cfg.globs, &cfg.excludes)? {
        let Some(lang) = code::Lang::from_path(&f.rel_path) else {
            unsupported += 1;
            continue;
        };
        let bytes = std::fs::read(&f.abs_path)?;
        let hash = hasher::hash_bytes(&bytes);
        let current = stored
            .get(&f.rel_path)
            .is_some_and(|(h, v)| *h == hash && v == code::EXTRACTOR_VERSION);
        if !current {
            to_extract.push((f.rel_path.clone(), hash.clone(), lang, bytes));
        }
        scanned.insert(f.rel_path, hash);
    }
    if unsupported > 0 {
        notes.push(format!(
            "{unsupported} files matched the code globs but are not .rs/.py/.js/.ts/.tsx/.go; skipped"
        ));
    }

    let mut chain = None;
    let mut chain_failed = false;
    let mut extracted = Vec::with_capacity(to_extract.len());
    let mut skipped = Vec::new();
    for (path, hash, lang, bytes) in to_extract {
        let src = String::from_utf8_lossy(&bytes);
        let (artifacts, skip) = code::extract(&path, lang, &src);
        if let Some(skip) = skip {
            skipped.push(format!("{path} ({skip:?})"));
        }
        let mut bodies = Vec::with_capacity(artifacts.len());
        for a in &artifacts {
            bodies.push(preprocess(
                preprocessor_commands,
                &mut chain,
                &mut chain_failed,
                &a.text,
            ));
        }
        extracted.push(Extracted {
            path,
            hash,
            artifacts,
            bodies,
            anchors: code::anchors(&src),
        });
    }
    if !skipped.is_empty() {
        notes.push(format!(
            "{} code files indexed without symbols: {}",
            skipped.len(),
            skipped.join(", ")
        ));
    }
    if chain_failed {
        notes.push("code artifacts indexed without the preprocessor (it failed)".into());
    }
    Ok(Prepared {
        enabled: true,
        scanned,
        extracted,
        notes,
    })
}

/// Raw text when the collection has no preprocessor or the chain fails.
fn preprocess(
    commands: &[String],
    chain: &mut Option<PreprocessChain>,
    failed: &mut bool,
    text: &str,
) -> String {
    if commands.is_empty() || *failed {
        return text.to_string();
    }
    if chain.is_none() {
        match PreprocessChain::spawn_all(commands) {
            Ok(c) => *chain = Some(c),
            Err(_) => {
                *failed = true;
                return text.to_string();
            }
        }
    }
    match chain.as_mut().map(|c| c.process_text(text)) {
        Some(Ok(body)) => body,
        _ => {
            *failed = true;
            *chain = None;
            text.to_string()
        }
    }
}

/// {path → (hash, extractor)} of stored code sources.
fn stored_sources(conn: &Connection) -> Result<HashMap<String, (String, String)>> {
    let mut stmt =
        conn.prepare("SELECT path, hash, extractor FROM artifact_sources WHERE domain = ?1")?;
    let rows = stmt.query_map([CODE_DOMAIN], |r| {
        Ok((r.get::<_, String>(0)?, (r.get(1)?, r.get(2)?)))
    })?;
    Ok(rows.collect::<std::result::Result<_, _>>()?)
}

/// Apply prepared changes inside the update's write transaction, then
/// rebuild document links when documents or artifacts changed.
/// Returns notes for the user about links that point to nothing.
pub fn apply(
    conn: &Connection,
    prepared: &Prepared,
    documents_changed: bool,
) -> Result<Vec<String>> {
    let mut changed = false;
    if !prepared.enabled {
        // ^ The domain was removed from config: drop what an earlier update stored.
        let had: bool =
            conn.query_row("SELECT EXISTS (SELECT 1 FROM artifact_sources)", [], |r| {
                r.get(0)
            })?;
        if had {
            conn.execute_batch(
                "DELETE FROM artifact_edges;
                 DELETE FROM artifact_fts;
                 DELETE FROM artifacts;
                 DELETE FROM artifact_sources;",
            )?;
            changed = true;
        }
    } else {
        let stored = stored_sources(conn)?;
        for path in stored.keys() {
            if !prepared.scanned.contains_key(path) {
                delete_source(conn, path)?;
                changed = true;
            }
        }
        for e in &prepared.extracted {
            delete_source(conn, &e.path)?;
            store_source(conn, e)?;
            changed = true;
        }
    }
    let mut notes = Vec::new();
    if changed || documents_changed {
        notes = rebuild_edges(conn)?;
    }
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM artifacts", [], |r| r.get(0))?;
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
        params![META_COUNT, count.to_string()],
    )?;
    Ok(notes)
}

fn delete_source(conn: &Connection, path: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM artifact_fts WHERE rowid IN
           (SELECT id FROM artifacts WHERE domain = ?1 AND path = ?2)",
        params![CODE_DOMAIN, path],
    )?;
    // ^ ON DELETE CASCADE removes the artifact's edges.
    conn.execute(
        "DELETE FROM artifacts WHERE domain = ?1 AND path = ?2",
        params![CODE_DOMAIN, path],
    )?;
    conn.execute(
        "DELETE FROM artifact_sources WHERE domain = ?1 AND path = ?2",
        params![CODE_DOMAIN, path],
    )?;
    Ok(())
}

fn store_source(conn: &Connection, e: &Extracted) -> Result<()> {
    let mut file_id = None;
    for (a, body) in e.artifacts.iter().zip(&e.bodies) {
        conn.execute(
            "INSERT INTO artifacts (domain, kind, name, path, start_line, end_line, text)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                CODE_DOMAIN,
                a.kind,
                a.name,
                e.path,
                a.start_line as i64,
                a.end_line as i64,
                a.text
            ],
        )?;
        let id = conn.last_insert_rowid();
        file_id.get_or_insert(id);
        conn.execute(
            "INSERT INTO artifact_fts (rowid, name, body) VALUES (?1, ?2, ?3)",
            params![id, code::split_words(&a.name), body],
        )?;
    }
    // ^ The first artifact is the file; its anchors cover the whole file.
    if let Some(file_id) = file_id {
        for target in &e.anchors {
            conn.execute(
                "INSERT OR IGNORE INTO artifact_anchors (artifact_id, target) VALUES (?1, ?2)",
                params![file_id, target],
            )?;
        }
    }
    conn.execute(
        "INSERT INTO artifact_sources (domain, path, hash, extractor) VALUES (?1, ?2, ?3, ?4)",
        params![CODE_DOMAIN, e.path, e.hash, code::EXTRACTOR_VERSION],
    )?;
    Ok(())
}

// ── document links ──────────────────────────────────────────────────────────

/// Recompute every artifact → document edge from the active documents' text.
///
/// - `mentions_symbol`: the document names a symbol. Outside backticks only
///   distinctive identifiers count (`snake_case`, `camelCase`, `Path::item`);
///   inside backticks any name of 3+ characters does.
/// - `mentions_path`: the document names a source file by its path or a
///   unique path suffix (`db/schema.rs`).
///
/// Weight 1/(1 + ln df), df = documents mentioning the artifact. Names with
/// more than MAX_DEFINITIONS definitions and artifacts mentioned by more than
/// HUB_DOCUMENTS documents get no edges.
///
/// Explicit links get weight 1 and no hub or ambiguity limits:
/// - `declared`: the document's frontmatter `code:` lists a file path (or
///   unique suffix), `path:symbol`, or a symbol name.
/// - `anchored`: a code comment names the document (see `code::anchors`) by
///   its path (with or without `.md`) or its title.
///
/// Returns one note per explicit link that resolves to nothing: a stale
/// pointer after a rename or deletion.
fn rebuild_edges(conn: &Connection) -> Result<Vec<String>> {
    conn.execute("DELETE FROM artifact_edges", [])?;
    let mut insert = conn.prepare(
        "INSERT OR IGNORE INTO artifact_edges (artifact_id, document_id, relation, weight)
         VALUES (?1, ?2, ?3, ?4)",
    )?;

    let mut by_name: HashMap<String, Vec<i64>> = HashMap::new();
    let mut by_path_suffix: HashMap<String, Vec<i64>> = HashMap::new();
    {
        let mut stmt = conn.prepare("SELECT id, kind, name FROM artifacts")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        for row in rows {
            let (id, kind, name) = row?;
            if kind == "file" {
                for suffix in path_suffixes(&name) {
                    by_path_suffix.entry(suffix).or_default().push(id);
                }
            } else {
                by_name.entry(name).or_default().push(id);
            }
        }
    }
    let stale = link_explicit(conn, &mut insert, &by_name, &by_path_suffix)?;
    by_name.retain(|_, ids| ids.len() <= MAX_DEFINITIONS);
    // ^ `mod.rs` names many files; only a suffix that names one file links.
    by_path_suffix.retain(|_, ids| ids.len() == 1);
    if by_name.is_empty() && by_path_suffix.is_empty() {
        return Ok(stale);
    }

    // {(artifact, relation) → documents}
    let mut mentions: HashMap<(i64, &'static str), Vec<i64>> = HashMap::new();
    {
        let mut stmt = conn.prepare(
            "SELECT d.id, c.doc FROM documents d JOIN content c ON c.hash = d.hash
             WHERE d.active = 1",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
        for row in rows {
            let (doc_id, text) = row?;
            let mut seen: HashSet<(i64, &'static str)> = HashSet::new();
            for m in mentioned_identifiers(&text) {
                // ^ Short names (`lock`, `env`) are too generic even in backticks.
                if let Some(ids) = by_name.get(m.ident)
                    && ((m.in_backticks && m.ident.len() >= 5) || is_distinctive(m.ident))
                {
                    seen.extend(ids.iter().map(|&id| (id, "mentions_symbol")));
                }
            }
            for token in path_tokens(&text) {
                if let Some(ids) = by_path_suffix.get(token) {
                    seen.extend(ids.iter().map(|&id| (id, "mentions_path")));
                }
            }
            for key in seen {
                mentions.entry(key).or_default().push(doc_id);
            }
        }
    }

    for ((artifact_id, relation), docs) in mentions {
        if docs.len() > HUB_DOCUMENTS {
            continue;
        }
        let weight = 1.0 / (1.0 + (docs.len() as f64).ln());
        for doc_id in docs {
            insert.execute(params![artifact_id, doc_id, relation, weight])?;
        }
    }
    Ok(stale)
}

/// Insert `declared` and `anchored` edges; return notes for targets that
/// resolve to nothing (or to too many definitions to choose from).
fn link_explicit(
    conn: &Connection,
    insert: &mut rusqlite::Statement<'_>,
    by_name: &HashMap<String, Vec<i64>>,
    by_path_suffix: &HashMap<String, Vec<i64>>,
) -> Result<Vec<String>> {
    let mut stale = Vec::new();

    // Frontmatter `code:` → artifacts.
    let declared: Vec<(i64, String, String)> = {
        let mut stmt = conn.prepare(
            "SELECT m.document_id, d.path, m.value FROM document_metadata m
             JOIN documents d ON d.id = m.document_id
             WHERE m.key = 'code' AND d.active = 1",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        rows.collect::<std::result::Result<_, _>>()?
    };
    let mut symbol_in_file = conn.prepare(
        "SELECT id FROM artifacts WHERE kind != 'file' AND name = ?1
           AND (path = ?2 OR path LIKE '%/' || ?2)",
    )?;
    for (doc_id, doc_path, target) in declared {
        let target = target.trim();
        let ids: Vec<i64> = match target.rsplit_once(':') {
            Some((file, symbol)) if !file.is_empty() && !symbol.is_empty() => symbol_in_file
                .query_map(params![symbol, file], |r| r.get(0))?
                .collect::<std::result::Result<_, _>>()?,
            _ => by_path_suffix
                .get(target)
                .filter(|ids| ids.len() == 1)
                .or_else(|| {
                    by_name
                        .get(target)
                        .filter(|ids| ids.len() <= MAX_DEFINITIONS)
                })
                .cloned()
                .unwrap_or_default(),
        };
        if ids.is_empty() {
            stale.push(format!(
                "{doc_path}: code link `{target}` matches no unique file or symbol"
            ));
        }
        for id in ids {
            insert.execute(params![id, doc_id, "declared", 1.0])?;
        }
    }

    // Code comments → documents.
    let anchors: Vec<(i64, String, String)> = {
        let mut stmt = conn.prepare(
            // ^ An anchor anywhere in the file links every artifact of the file.
            "SELECT b.id, a.path, x.target FROM artifact_anchors x
             JOIN artifacts a ON a.id = x.artifact_id
             JOIN artifacts b ON b.domain = a.domain AND b.path = a.path",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        rows.collect::<std::result::Result<_, _>>()?
    };
    if !anchors.is_empty() {
        let docs: Vec<(i64, String, String)> = {
            let mut stmt =
                conn.prepare("SELECT id, path, title FROM documents WHERE active = 1")?;
            let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
            rows.collect::<std::result::Result<_, _>>()?
        };
        for (artifact_id, code_path, target) in anchors {
            let with_md = format!("{}.md", target.trim_end_matches(".md"));
            let matched: Vec<i64> = docs
                .iter()
                .filter(|(_, path, title)| {
                    *title == target || *path == with_md || path.ends_with(&format!("/{with_md}"))
                })
                .map(|(id, _, _)| *id)
                .collect();
            if matched.is_empty() {
                stale.push(format!(
                    "{code_path}: knowledge link `{target}` matches no document"
                ));
            }
            for doc_id in matched {
                insert.execute(params![artifact_id, doc_id, "anchored", 1.0])?;
            }
        }
    }
    stale.sort();
    stale.dedup();
    Ok(stale)
}

struct Mention<'a> {
    ident: &'a str,
    in_backticks: bool,
}

/// Identifier tokens of a document, marking those inside `backticks`.
/// `crate::db::fts::search` yields `crate`, `db`, `fts`, `search`.
fn mentioned_identifiers(text: &str) -> Vec<Mention<'_>> {
    let mut out = Vec::new();
    let mut in_backticks = false;
    let mut start: Option<usize> = None;
    for (i, c) in text
        .char_indices()
        .chain(std::iter::once((text.len(), ' ')))
    {
        if c.is_ascii_alphanumeric() || c == '_' {
            start.get_or_insert(i);
            continue;
        }
        if let Some(s) = start.take() {
            let ident = &text[s..i];
            if !ident.as_bytes()[0].is_ascii_digit() {
                out.push(Mention {
                    ident,
                    in_backticks,
                });
            }
        }
        if c == '`' {
            in_backticks = !in_backticks;
        } else if c == '\n' && text[..i].ends_with("\n") {
            // ^ A blank line ends an unclosed inline span.
            in_backticks = false;
        }
    }
    out
}

/// Names that rarely occur as plain words: `snake_case`, `camelCase`,
/// `PascalCase` with two humps, `SCREAMING_CASE`. Plain `search` or
/// `Config` link only from inside backticks.
fn is_distinctive(ident: &str) -> bool {
    if ident.len() < 4 {
        return false;
    }
    let inner_underscore = ident.trim_matches('_').contains('_');
    let bytes = ident.as_bytes();
    let hump = bytes
        .windows(2)
        .any(|w| w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase());
    inner_underscore || hump
}

/// Path-like tokens: contain `/` or a dotted file name, trimmed of the
/// surrounding punctuation of prose and Markdown.
fn path_tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                '`' | '(' | ')' | '[' | ']' | '<' | '>' | '"' | '\'' | ','
            )
    })
    .map(|t| {
        t.trim_start_matches("./")
            .trim_end_matches(['.', ':', ';', '!', '?'])
    })
    .map(|t| t.split(':').next().unwrap_or(t))
    .filter(|t| {
        t.contains('/')
            || t.rsplit_once('.')
                .is_some_and(|(stem, ext)| !stem.is_empty() && !ext.is_empty() && ext.len() <= 4)
    })
}

/// `src/db/schema.rs` → [`src/db/schema.rs`, `db/schema.rs`, `schema.rs`].
fn path_suffixes(path: &str) -> Vec<String> {
    let mut out = vec![path.to_string()];
    let mut rest = path;
    while let Some((_, tail)) = rest.split_once('/') {
        out.push(tail.to_string());
        rest = tail;
    }
    out
}

// ── search ──────────────────────────────────────────────────────────────────

/// True when the collection DB holds artifacts.
pub fn has_artifacts(conn: &Connection) -> bool {
    artifact_count(conn) > 0
}

/// Stored artifacts per auxiliary domain, by domain name; empty without one.
pub fn artifact_counts(conn: &Connection) -> Vec<(String, usize)> {
    if !has_artifacts(conn) {
        return vec![];
    }
    let Ok(mut stmt) =
        conn.prepare("SELECT domain, COUNT(*) FROM artifacts GROUP BY domain ORDER BY domain")
    else {
        return vec![];
    };
    stmt.query_map([], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as usize))
    })
    .map(|rows| rows.flatten().collect())
    .unwrap_or_default()
}

/// Number of stored auxiliary artifacts (all domains); 0 without one.
pub fn artifact_count(conn: &Connection) -> usize {
    conn.query_row("SELECT value FROM meta WHERE key = ?1", [META_COUNT], |r| {
        r.get::<_, String>(0)
    })
    .optional()
    .ok()
    .flatten()
    .and_then(|v| v.parse::<usize>().ok())
    .unwrap_or(0)
}

/// BM25 over one collection, then auxiliary support. Same as `fts::search`
/// for collections without artifacts.
pub fn bm25_with_support(db: &CollectionDb, q: &fts::BM25Query) -> Result<Vec<SearchResult>> {
    let mut results = fts::search(db.conn(), q)?;
    support(db, &q.fts_query, &mut results)?;
    Ok(results)
}

/// Raise or add documents of one collection that artifacts matching
/// `fts_query` link to. `results` are that collection's BM25 hits; they stay
/// sorted by score. No-op without stored artifacts.
pub fn support(db: &CollectionDb, fts_query: &str, results: &mut Vec<SearchResult>) -> Result<()> {
    if fts_query.is_empty() || !db.has_artifacts() {
        return Ok(());
    }
    let conn = db.conn();
    let hits = artifact_hits(conn, fts_query)?;
    if hits.is_empty() {
        return Ok(());
    }

    // {document → (support, evidence by contribution)}
    let mut by_doc: HashMap<i64, (f64, Evidence)> = HashMap::new();
    let mut edges = conn
        .prepare_cached("SELECT document_id, weight FROM artifact_edges WHERE artifact_id = ?1")?;
    for hit in &hits {
        let rows = edges.query_map([hit.id], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?)))?;
        for row in rows {
            let (doc_id, weight) = row?;
            let contribution = hit.score * weight;
            let entry = by_doc.entry(doc_id).or_insert((0.0, Vec::new()));
            entry.0 = entry.0.max(contribution);
            entry.1.push((contribution, hit.label()));
        }
    }
    if by_doc.is_empty() {
        return Ok(());
    }

    let mut doc_info = conn
        .prepare_cached("SELECT path, title, hash FROM documents WHERE id = ?1 AND active = 1")?;
    let mut ranked: Vec<(i64, f64, Evidence)> =
        by_doc.into_iter().map(|(d, (s, e))| (d, s, e)).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

    let mut injected = 0;
    for (doc_id, sup, mut evidence) in ranked {
        let Some((path, title, hash)) = doc_info
            .query_row([doc_id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .optional()?
        else {
            continue;
        };
        evidence.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let mut labels: Vec<String> = Vec::new();
        for (_, label) in evidence {
            if !labels.contains(&label) {
                labels.push(label);
            }
            if labels.len() == EVIDENCE_PER_DOC {
                break;
            }
        }
        if let Some(r) = results.iter_mut().find(|r| r.path == path) {
            r.score += SUPPORT_BETA * sup * (1.0 - r.score);
            r.evidence = Some(labels);
        } else if injected < MAX_INJECTED {
            injected += 1;
            results.push(SearchResult {
                collection: db.name.clone(),
                doc_id: format!("#{}", &hash[..6.min(hash.len())]),
                path,
                title,
                score: SUPPORT_BETA * sup,
                snippet: None,
                hash,
                content: None,
                chunk_seq: None,
                evidence: Some(labels),
                kind: None,
            });
        }
    }
    SearchResult::sort_desc(results);
    Ok(())
}

/// (contribution, label) of each artifact that supports one document.
type Evidence = Vec<(f64, String)>;

/// Code artifacts matching `query` in one collection, best first, as
/// results of kind `code` (for `--aux`). Empty without artifacts.
/// `path` is the source file; `doc_id` is `code:<path>:<line>`.
pub fn code_results(db: &CollectionDb, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
    if limit == 0 || !db.has_artifacts() {
        return Ok(vec![]);
    }
    let fts_query = fts::build_query_natural(&db.preprocess_query(query));
    if fts_query.is_empty() {
        return Ok(vec![]);
    }
    let mut out: Vec<SearchResult> = artifact_hits(db.conn(), &fts_query)?
        .into_iter()
        .map(|h| {
            // ^ Stored text lines: kind, name, name words, then signature and doc.
            let snippet = h.text.lines().skip(3).collect::<Vec<_>>().join(" ");
            SearchResult {
                collection: db.name.clone(),
                title: if h.kind == "file" {
                    format!("file {}", h.path)
                } else {
                    format!(
                        "{} {} (lines {}-{})",
                        h.kind, h.name, h.start_line, h.end_line
                    )
                },
                doc_id: format!("{CODE_DOMAIN}:{}:{}", h.path, h.start_line),
                path: h.path,
                score: h.score,
                snippet: (!snippet.is_empty()).then_some(snippet),
                hash: String::new(),
                content: None,
                chunk_seq: None,
                evidence: None,
                kind: Some(CODE_DOMAIN.to_string()),
            }
        })
        .collect();
    out.truncate(limit);
    Ok(out)
}

struct ArtifactHit {
    id: i64,
    kind: String,
    name: String,
    path: String,
    score: f64,
    start_line: i64,
    end_line: i64,
    /// Stored search text: kind, name, name words, signature, doc comment.
    text: String,
}

impl ArtifactHit {
    fn label(&self) -> String {
        if self.kind == "file" {
            format!("{CODE_DOMAIN}:{}", self.path)
        } else {
            format!("{CODE_DOMAIN}:{}:{}", self.path, self.name)
        }
    }
}

fn artifact_hits(conn: &Connection, fts_query: &str) -> Result<Vec<ArtifactHit>> {
    // ^ Name words weigh 5× the body (signature and doc comment).
    let mut stmt = conn.prepare_cached(
        "SELECT a.id, a.kind, a.name, a.path, bm25(artifact_fts, 5.0, 1.0) AS score,
                artifact_fts.name, artifact_fts.body, a.start_line, a.end_line, a.text
         FROM artifact_fts JOIN artifacts a ON a.id = artifact_fts.rowid
         WHERE artifact_fts MATCH ?1
         ORDER BY score ASC
         LIMIT ?2",
    )?;
    let terms = query_terms(fts_query);
    // ^ A natural-language query is an OR of its words. An artifact matching
    //   one of them ("per", "lock") is not evidence; require half.
    let needed = terms.len().div_ceil(2);
    let rows = stmt.query_map(params![fts_query, ARTIFACT_HITS as i64], |r| {
        let fts_name: String = r.get(5)?;
        let fts_body: String = r.get(6)?;
        Ok((
            ArtifactHit {
                id: r.get(0)?,
                kind: r.get(1)?,
                name: r.get(2)?,
                path: r.get(3)?,
                score: fts::normalize(r.get(4)?),
                start_line: r.get(7)?,
                end_line: r.get(8)?,
                text: r.get(9)?,
            },
            fts_name + " " + &fts_body,
        ))
    })?;
    let mut hits = Vec::new();
    for row in rows {
        let (hit, text) = row?;
        if covered_terms(&terms, &text) >= needed {
            hits.push(hit);
        }
    }
    Ok(hits)
}

/// Positive terms of an FTS query built by `fts::build_query*`:
/// `"cache"* OR "policy"* NOT "draft"` → [cache, policy].
fn query_terms(fts_query: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut rest = fts_query;
    while let Some(start) = rest.find('"') {
        let negated = rest[..start].trim_end().ends_with("NOT");
        let after = &rest[start + 1..];
        let Some(end) = after.find('"') else { break };
        let term = after[..end].to_lowercase();
        if !negated && !term.is_empty() && !terms.contains(&term) {
            terms.push(term);
        }
        rest = &after[end + 1..];
    }
    terms
}

/// How many `terms` occur in `text`, counting a shared stem (`entries` /
/// `entry`, `invalidation` / `invalidate`) as a match like porter does.
fn covered_terms(terms: &[String], text: &str) -> usize {
    let tokens: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect();
    terms
        .iter()
        .filter(|term| {
            term.split_whitespace().all(|word| {
                tokens.iter().any(|tok| {
                    if word.chars().count() <= 4 {
                        tok.starts_with(word)
                    } else {
                        let common = word
                            .chars()
                            .zip(tok.chars())
                            .take_while(|(a, b)| a == b)
                            .count();
                        common >= 4.max(word.chars().count() - 3)
                    }
                })
            })
        })
        .count()
}
