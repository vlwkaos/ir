// End-to-end auxiliary-domain scenarios through the compiled `ir` binary.
// A code domain never returns code files; its symbols raise or add the
// Markdown documents that mention them. Oracle is always a real BM25 search.
// knowledge: bigpicture/Multiscript and Auxiliary Retrieval Roadmap.md (Branch 2)

mod support;

use serde_json::Value;
use std::fs;
use std::path::Path;
use support::{Env, add_preprocessor, update_counts, write};

const CACHE_RS: &str = "//! In-memory cache for parsed records.

/// Drop outdated values after a schema migration.
pub fn invalidate_entry(key: &str) {}

/// Total number of cached records.
pub fn entry_count() -> usize {
    0
}
";

const REPORT_PY: &str = r#""""Invoice reports."""


class InvoiceReport:
    """Monthly statement for one customer."""

    def render_invoice(self, totals):
        """Render the PDF layout with line totals."""
        return totals
"#;

fn write_notes(dir: &Path) {
    write(
        dir,
        "cache.md",
        "# Cache layer\n\nWe call invalidate_entry whenever a record changes.\n",
    );
    write(
        dir,
        "billing.md",
        "# Billing\n\nThe monthly job calls `render_invoice` for every customer.\n",
    );
    write(
        dir,
        "design.md",
        "# Design\n\nDetails are in src/cache.rs.\n",
    );
    write(
        dir,
        "migration.md",
        "# Migration notes\n\nThe migration ran without problems.\n",
    );
}

fn write_code(dir: &Path) {
    write(dir, "src/cache.rs", CACHE_RS);
    write(dir, "src/report.py", REPORT_PY);
}

fn search(env: &Env, collection: &str, query: &str) -> Vec<Value> {
    let stdout = env.ok(&[
        "search", "--mode", "bm25", "--json", "--quiet", "-c", collection, query,
    ]);
    let results: Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("search output is not JSON ({e}): {stdout}"));
    results.as_array().expect("JSON array").clone()
}

fn paths(results: &[Value]) -> Vec<&str> {
    results
        .iter()
        .map(|r| r["path"].as_str().unwrap())
        .collect()
}

fn evidence_of<'a>(results: &'a [Value], path: &str) -> Vec<&'a str> {
    results
        .iter()
        .find(|r| r["path"] == path)
        .and_then(|r| r["evidence"].as_array())
        .map(|e| e.iter().map(|v| v.as_str().unwrap()).collect())
        .unwrap_or_default()
}

fn add_with_code(env: &Env, name: &str, extra: &[&str]) -> std::path::PathBuf {
    let mut args = vec!["--code-glob", "src/**/*.rs", "--code-glob", "src/**/*.py"];
    args.extend_from_slice(extra);
    let dir = env.add_collection(name, &args);
    write_notes(&dir);
    write_code(&dir);
    env.ok(&["update", name]);
    dir
}

#[test]
fn code_symbols_support_the_documents_that_mention_them() {
    let env = Env::new("e2e-aux-support");
    add_with_code(&env, "notes", &[]);

    // ^ No document contains these words; only the doc comment of
    //   `invalidate_entry` does, and cache.md names that function.
    let results = search(&env, "notes", "outdated values schema migration");
    assert!(
        paths(&results).contains(&"cache.md"),
        "cache.md not supported: {results:?}"
    );
    assert_eq!(
        evidence_of(&results, "cache.md"),
        ["code:src/cache.rs:invalidate_entry"]
    );
    // The document BM25 hit stays and carries no evidence.
    assert!(paths(&results).contains(&"migration.md"));
    assert!(evidence_of(&results, "migration.md").is_empty());

    // Python docstring → `render_invoice`, named in backticks by billing.md.
    let results = search(&env, "notes", "pdf layout line totals");
    assert_eq!(paths(&results).first(), Some(&"billing.md"), "{results:?}");
    assert_eq!(
        evidence_of(&results, "billing.md"),
        ["code:src/report.py:render_invoice"]
    );

    // Module doc → file artifact, linked by the path design.md names.
    let results = search(&env, "notes", "parsed records memory");
    assert!(
        evidence_of(&results, "design.md").contains(&"code:src/cache.rs"),
        "{results:?}"
    );

    // Code files are never results.
    for query in [
        "invalidate entry",
        "render invoice",
        "cache rs",
        "InvoiceReport",
    ] {
        let results = search(&env, "notes", query);
        assert!(
            paths(&results).iter().all(|p| p.ends_with(".md")),
            "{query}: {results:?}"
        );
    }
}

#[test]
fn collection_without_code_domain_matches_one_with_no_artifact_hits() {
    let env = Env::new("e2e-aux-plain");
    let plain = env.add_collection("plain", &[]);
    write_notes(&plain);
    write_code(&plain);
    env.ok(&["update", "plain"]);
    add_with_code(&env, "coded", &[]);

    // ^ No artifact matches these, so both collections must agree exactly,
    //   scores included.
    for query in [
        "migration problems",
        "monthly customer job",
        "cache layer record",
    ] {
        let a = search(&env, "plain", query);
        let b = search(&env, "coded", query);
        let strip = |rs: &[Value]| -> Vec<(String, String)> {
            rs.iter()
                .map(|r| (r["path"].to_string(), r["score"].to_string()))
                .collect()
        };
        assert_eq!(strip(&a), strip(&b), "{query}");
    }
    // The plain collection never reports evidence or adds documents.
    let results = search(&env, "plain", "outdated values schema migration");
    assert_eq!(paths(&results), ["migration.md"]);
    assert!(results.iter().all(|r| r.get("evidence").is_none()));
}

#[test]
fn code_and_document_changes_update_the_links() {
    let env = Env::new("e2e-aux-lifecycle");
    let dir = add_with_code(&env, "notes", &[]);
    let supported = |query: &str| {
        let results = search(&env, "notes", query);
        !evidence_of(&results, "cache.md").is_empty()
    };
    assert!(supported("outdated values schema migration"));

    // An update with no changes keeps the links.
    let stdout = env.ok(&["update", "notes"]);
    assert_eq!(update_counts(&stdout), (4, 0, 0, 0));
    assert!(supported("outdated values schema migration"));

    // The doc comment changes: the old words stop supporting, the new ones do.
    write(
        &dir,
        "src/cache.rs",
        &CACHE_RS.replace(
            "Drop outdated values after a schema migration.",
            "Remove expired rows after a reload.",
        ),
    );
    env.ok(&["update", "notes"]);
    assert!(!supported("outdated values schema migration"));
    assert!(supported("expired rows reload"));

    // The document stops naming the function: no link.
    write(&dir, "cache.md", "# Cache layer\n\nRecords change often.\n");
    env.ok(&["update", "notes"]);
    assert!(!supported("expired rows reload"));

    // It names it again, then the code file is deleted: no link.
    write_notes(&dir);
    env.ok(&["update", "notes"]);
    assert!(supported("expired rows reload"));
    fs::remove_file(dir.join("src/cache.rs")).unwrap();
    env.ok(&["update", "notes"]);
    assert!(!supported("expired rows reload"));
}

#[test]
fn removing_the_code_domain_drops_its_artifacts() {
    let env = Env::new("e2e-aux-remove");
    let dir = add_with_code(&env, "notes", &[]);
    assert!(
        !evidence_of(
            &search(&env, "notes", "outdated values schema migration"),
            "cache.md"
        )
        .is_empty()
    );

    // ^ `collection rm` keeps the DB; the same name re-added without
    //   --code-glob reuses it, and the next update must purge the artifacts.
    env.ok(&["collection", "rm", "notes"]);
    env.ok(&["collection", "add", "notes", dir.to_str().unwrap()]);
    env.ok(&["update", "notes"]);
    let results = search(&env, "notes", "outdated values schema migration");
    assert_eq!(paths(&results), ["migration.md"]);
    assert!(results.iter().all(|r| r.get("evidence").is_none()));
}

#[test]
fn preprocessed_collection_indexes_artifact_text_through_its_chain() {
    let env = Env::new("e2e-aux-preprocessed");
    // ^ `cat` obeys the line protocol; other filters buffer and deadlock.
    add_preprocessor(&env, "pass", "cat");
    add_with_code(&env, "notes", &["-p", "pass"]);
    let results = search(&env, "notes", "outdated values schema migration");
    assert_eq!(
        evidence_of(&results, "cache.md"),
        ["code:src/cache.rs:invalidate_entry"],
        "{results:?}"
    );
}

#[test]
fn typescript_javascript_and_go_symbols_are_extracted() {
    let env = Env::new("e2e-aux-langs");
    let dir = env.add_collection(
        "notes",
        &[
            "--code-glob",
            "web/**/*.ts",
            "--code-glob",
            "web/**/*.js",
            "--code-glob",
            "svc/**/*.go",
        ],
    );
    write(
        &dir,
        "web/session.ts",
        "/** Refresh the access token before it expires. */\n\
         export async function renewSessionToken(id: string): Promise<void> {}\n\n\
         export interface TokenStore {\n  /** Persist credentials across restarts. */\n  saveCredentials(id: string): void;\n}\n",
    );
    write(
        &dir,
        "web/format.js",
        "/** Pretty-print currency amounts for receipts. */\nexport const formatAmount = (n) => n.toFixed(2);\n",
    );
    write(
        &dir,
        "svc/queue.go",
        "package svc\n\n// RetryQueue holds deliveries waiting for backoff.\ntype RetryQueue struct{}\n\n\
         // DrainBacklog flushes pending webhooks to subscribers.\nfunc (q *RetryQueue) DrainBacklog() {}\n",
    );
    write(
        &dir,
        "auth.md",
        "# Auth\n\nThe client calls renewSessionToken on startup.\n",
    );
    write(
        &dir,
        "store.md",
        "# Store\n\nImplement `saveCredentials` in each backend.\n",
    );
    write(
        &dir,
        "receipts.md",
        "# Receipts\n\nTotals go through formatAmount.\n",
    );
    write(
        &dir,
        "hooks.md",
        "# Hooks\n\nOperators can run DrainBacklog by hand.\n",
    );
    env.ok(&["update", "notes"]);

    for (query, path, evidence) in [
        (
            "access token expires",
            "auth.md",
            "code:web/session.ts:renewSessionToken",
        ),
        (
            "persist credentials restarts",
            "store.md",
            "code:web/session.ts:saveCredentials",
        ),
        (
            "currency amounts receipts",
            "receipts.md",
            "code:web/format.js:formatAmount",
        ),
        (
            "pending webhooks subscribers",
            "hooks.md",
            "code:svc/queue.go:DrainBacklog",
        ),
    ] {
        let results = search(&env, "notes", query);
        assert!(
            evidence_of(&results, path).contains(&evidence),
            "{query}: {results:?}"
        );
    }
}

#[test]
fn unusable_code_files_warn_and_still_index_the_rest() {
    let env = Env::new("e2e-aux-unusable");
    let dir = add_with_code(&env, "notes", &["--code-glob", "src/**/*.txt"]);
    write(&dir, "src/readme.txt", "plain text, no grammar\n");
    // ^ Over the 1 MiB limit: stored as a file artifact without symbols.
    write(&dir, "src/huge.rs", &"// filler line\n".repeat(80_000));
    // A syntax error later in the file: the valid function before it is kept.
    write(
        &dir,
        "src/broken.rs",
        "/// Rotate signing keys nightly.\npub fn rotate_keys() {}\n\nfn half( {{{\n",
    );
    write(
        &dir,
        "keys.md",
        "# Keys\n\nOps run rotate_keys from cron. See src/huge.rs.\n",
    );
    let (ok, stdout, stderr) = env.run(&["update", "notes"]);
    assert!(ok, "{stdout}\n{stderr}");
    assert!(stderr.contains("not .rs/.py/.js/.ts/.tsx/.go"), "{stderr}");
    assert!(stderr.contains("src/huge.rs (TooLarge)"), "{stderr}");

    let results = search(&env, "notes", "signing keys nightly");
    assert!(
        evidence_of(&results, "keys.md").contains(&"code:src/broken.rs:rotate_keys"),
        "{results:?}"
    );
    // The oversized file has no symbols but still links through its path.
    assert!(evidence_of(&search(&env, "notes", "huge"), "keys.md").contains(&"code:src/huge.rs"),);
    // The other code files are unaffected.
    assert!(
        !evidence_of(
            &search(&env, "notes", "outdated values schema migration"),
            "cache.md"
        )
        .is_empty()
    );
}

#[test]
fn forced_and_concurrent_updates_keep_one_copy_of_each_artifact() {
    let env = Env::new("e2e-aux-concurrent");
    add_with_code(&env, "notes", &[]);
    env.ok(&["update", "notes", "--force"]);
    let children: Vec<_> = (0..4)
        .map(|_| {
            env.ir()
                .args(["update", "notes", "--force"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .expect("spawn update")
        })
        .collect();
    for child in children {
        let out = child.wait_with_output().expect("update finished");
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    // ^ A duplicated artifact would list the same evidence twice or add a
    //   second edge; the label set must be exactly one entry.
    let results = search(&env, "notes", "outdated values schema migration");
    assert_eq!(
        evidence_of(&results, "cache.md"),
        ["code:src/cache.rs:invalidate_entry"]
    );
    let conn = rusqlite::Connection::open(env.db_path("notes")).unwrap();
    let dupes: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM (SELECT path, name, kind FROM artifacts
             GROUP BY path, name, kind HAVING COUNT(*) > 1)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(dupes, 0);
}

#[test]
fn declared_and_anchored_links_connect_code_that_documents_never_name() {
    let env = Env::new("e2e-aux-explicit");
    let dir = env.add_collection("notes", &["--code-glob", "src/**/*.rs"]);
    // ^ Neither document names the function or the file in its text.
    write(
        &dir,
        "ops/rotation.md",
        "---\ncode: [src/keys.rs:rotate_all, src/gone.rs, missing_symbol]\n---\n# Rotation\n\nHow we rotate credentials.\n",
    );
    write(
        &dir,
        "design/ledger.md",
        "# Ledger design\n\nWhy balances are append-only.\n",
    );
    write(
        &dir,
        "src/keys.rs",
        "/// Replace every signing secret before expiry.\npub fn rotate_all() {}\n",
    );
    write(
        &dir,
        "src/ledger.rs",
        "// knowledge: design/ledger.md (append-only rule)\n// knowledge: design/removed.md\n\
         /// Apply a double-entry posting.\npub fn post_entry() {}\n",
    );
    let (ok, stdout, stderr) = env.run(&["update", "notes"]);
    assert!(ok, "{stdout}\n{stderr}");
    // Stale pointers are reported, one line each.
    assert!(
        stderr
            .contains("ops/rotation.md: code link `src/gone.rs` matches no unique file or symbol"),
        "{stderr}"
    );
    assert!(
        stderr.contains("ops/rotation.md: code link `missing_symbol`"),
        "{stderr}"
    );
    assert!(
        stderr.contains("src/ledger.rs: knowledge link `design/removed.md` matches no document"),
        "{stderr}"
    );

    let results = search(&env, "notes", "replace signing secret before expiry");
    assert_eq!(
        evidence_of(&results, "ops/rotation.md"),
        ["code:src/keys.rs:rotate_all"],
        "{results:?}"
    );
    let results = search(&env, "notes", "double entry posting");
    assert!(
        evidence_of(&results, "design/ledger.md")
            .iter()
            .any(|e| e.starts_with("code:src/ledger.rs")),
        "{results:?}"
    );
}

#[test]
fn one_shared_word_is_not_evidence() {
    let env = Env::new("e2e-aux-coverage");
    let dir = env.add_collection("notes", &["--code-glob", "src/**/*.rs"]);
    write(
        &dir,
        "chunks.md",
        "# Chunks\n\nSee `token_budget` for sizing.\n",
    );
    write(
        &dir,
        "src/chunker.rs",
        "/// Characters per token for the size estimate.\npub const token_budget: usize = 4;\n",
    );
    env.ok(&["update", "notes"]);
    // ^ Only "per" overlaps: an OR query must not let it support chunks.md.
    let results = search(&env, "notes", "register sqlite extension once per process");
    assert!(evidence_of(&results, "chunks.md").is_empty(), "{results:?}");
    // Most of the words overlap: it does.
    let results = search(&env, "notes", "characters per token size estimate");
    assert_eq!(
        evidence_of(&results, "chunks.md"),
        ["code:src/chunker.rs:token_budget"],
        "{results:?}"
    );
}

#[test]
fn backticked_anchor_syntax_is_an_example_not_a_link() {
    let env = Env::new("e2e-aux-anchor-example");
    let dir = env.add_collection("notes", &["--code-glob", "src/**/*.rs"]);
    write(&dir, "a.md", "# A\n\ntext\n");
    write(
        &dir,
        "src/doc.rs",
        "/// Links use `// knowledge: <path>` or `[[Wiki Title]]`.\npub fn explain() {}\n",
    );
    let (ok, stdout, stderr) = env.run(&["update", "notes"]);
    assert!(ok, "{stdout}\n{stderr}");
    assert!(!stderr.contains("knowledge link"), "{stderr}");
}

fn search_with_code(env: &Env, collection: &str, query: &str) -> Vec<Value> {
    let stdout = env.ok(&[
        "search",
        "--mode",
        "bm25",
        "--json",
        "--quiet",
        "--include-code",
        "-c",
        collection,
        query,
    ]);
    serde_json::from_str::<Value>(&stdout)
        .unwrap_or_else(|e| panic!("not JSON ({e}): {stdout}"))
        .as_array()
        .unwrap()
        .clone()
}

#[test]
fn include_code_returns_code_after_documents_only_on_request() {
    let env = Env::new("e2e-aux-include-code");
    add_with_code(&env, "notes", &[]);
    let plain = env.add_collection("plain", &[]);
    write_notes(&plain);
    write_code(&plain);
    env.ok(&["update", "plain"]);

    let query = "outdated values schema migration";
    // Without the flag: documents only.
    assert!(
        search(&env, "notes", query)
            .iter()
            .all(|r| r.get("kind").is_none())
    );

    let results = search_with_code(&env, "notes", query);
    let kinds: Vec<Option<&str>> = results.iter().map(|r| r["kind"].as_str()).collect();
    let first_code = kinds
        .iter()
        .position(|k| *k == Some("code"))
        .expect("a code result");
    // ^ Code is auxiliary: every document comes first.
    assert!(
        kinds[first_code..].iter().all(|k| *k == Some("code")),
        "{results:?}"
    );
    assert!(first_code > 0, "documents first: {results:?}");
    let code = &results[first_code];
    assert_eq!(code["path"], "src/cache.rs");
    assert_eq!(code["doc_id"], "code:src/cache.rs:4");
    assert!(
        code["title"]
            .as_str()
            .unwrap()
            .starts_with("function invalidate_entry")
    );
    assert!(
        code["snippet"]
            .as_str()
            .unwrap()
            .contains("Drop outdated values")
    );

    // A collection without a code domain returns no code even when asked.
    assert!(
        search_with_code(&env, "plain", query)
            .iter()
            .all(|r| r.get("kind").is_none())
    );

    // MCP: the same opt-in field.
    let mut mcp = support::Mcp::start(&env);
    let mcp_results = mcp
        .tool(
            "search",
            serde_json::json!({"query": query, "mode": "bm25", "collections": ["notes"], "include_code": true}),
        )
        .expect("search tool");
    assert!(
        mcp_results
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["kind"] == "code"),
        "{mcp_results}"
    );
}

/// Agents learn about the code domain only where it applies: `ir collection
/// add` hints once when a folder holds source files and no code domain was
/// asked for; `ir status`, MCP `status` and the MCP connect instructions name
/// the code only for collections that have it, so setups without code spend
/// no extra output on it.
#[test]
fn code_domain_is_announced_only_where_it_applies() {
    let env = Env::new("aux-announce");

    let repo = env.path("repo");
    write(
        &repo,
        "docs/cache.md",
        "# Cache\n\nNotes about the cache.\n",
    );
    write(
        &repo,
        "src/cache.rs",
        "/// Evict stale entries.\npub fn evict_stale() {}\n",
    );
    let plain = env.path("plain");
    write(&plain, "a.md", "# Plain\n\nOnly notes here.\n");

    let hint_for = |name: &str, dir: &std::path::Path, extra: &[&str]| {
        let mut args = vec!["collection", "add", name, dir.to_str().unwrap()];
        args.extend_from_slice(extra);
        let (ok, _, stderr) = env.run(&args);
        assert!(ok, "collection add failed: {stderr}");
        // ^ the hint spans two lines: "note: …" then the config snippet
        stderr.starts_with("note:") && stderr.contains("**/*.rs")
    };
    let hinted_repo = hint_for("repo", &repo, &[]);
    let hinted_plain = hint_for("plain", &plain, &[]);
    let hinted_with_code = hint_for(
        "code",
        &repo,
        &["--glob", "docs/**/*.md", "--code-glob", "src/**/*.rs"],
    );
    env.ok(&["update"]);

    let status = env.ok(&["status"]);
    let status_line = |name: &str| {
        status
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("{name} ")))
            .unwrap_or_default()
            .to_string()
    };

    let mut server = support::Mcp::start(&env);
    let mcp_status = server.tool("status", serde_json::json!({})).unwrap();
    let code_symbols = |name: &str| {
        mcp_status["collections"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .map(|c| c.get("code_symbols").cloned())
    };

    assert_eq!(
        (
            hinted_repo,
            hinted_plain,
            hinted_with_code,
            status_line("code").contains("code: "),
            status_line("plain").contains("code: "),
            code_symbols("code").flatten().is_some(),
            code_symbols("plain"),
            server.instructions.contains("include_code"),
        ),
        (true, false, false, true, false, true, Some(None), true),
        "status:\n{status}\nmcp status: {mcp_status}\ninstructions: {}",
        server.instructions
    );
}
