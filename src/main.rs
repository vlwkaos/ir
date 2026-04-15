mod cli;
mod config;
mod daemon;
mod db;
mod error;
mod get;
mod index;
mod llm;
mod mcp;
mod preprocess;
mod search;
mod types;

use clap::Parser;
use cli::{Cli, CollectionCmd, Command, DaemonCmd, PreprocessorCmd, output};
use get::{DocContent, MultiGetResult};
use config::{Config, collection_db_path};
use error::Result;
use types::{Collection, SearchMode};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Collection { cmd } => handle_collection(cmd),
        Command::Status => handle_status(),
        Command::Update { collection, force } => handle_update(collection, force),
        Command::Embed { collection, force } => handle_embed(collection, force),
        Command::Search {
            query,
            mode,
            limit,
            min_score,
            collections,
            all,
            full,
            chunk,
            json,
            csv,
            md,
            files,
            verbose,
        } => handle_search(
            query.join(" "),
            mode,
            if all { crate::db::vectors::KNN_MAX } else { limit },
            min_score,
            collections,
            full,
            chunk,
            json,
            csv,
            md,
            files,
            verbose,
        ),
        Command::Get { target, collections, section, offset, max_chars, json } => {
            handle_get(target, collections, section, offset, max_chars, json)
        }
        Command::MultiGet { targets, collections, max_chars, json, files } => {
            handle_multi_get(targets, collections, max_chars, json, files)
        }
        Command::Daemon { cmd } => match cmd {
            DaemonCmd::Start { timeout } => daemon::start_server(timeout),
            DaemonCmd::Stop => daemon::stop(),
            DaemonCmd::Status => daemon::status(),
        },
        Command::Preprocessor { cmd } => handle_preprocessor(cmd),
        Command::Mcp { http } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| error::Error::Other(e.to_string()))?;
            rt.block_on(mcp::run(http))
        }
    }
}

fn handle_get(
    target: String,
    collections: Vec<String>,
    section: Option<String>,
    offset: Option<usize>,
    max_chars: Option<usize>,
    json: bool,
) -> Result<()> {
    let config = Config::load()?;
    let filter = resolve_collections(&config, &collections)?;
    match get::fetch_document_with_config(&target, &filter, &config)? {
        Some(mut doc) => {
            if let Some(ref heading) = section {
                let extracted = get::extract_section(&doc.content, heading)
                    .unwrap_or("")
                    .to_string();
                doc.content = extracted;
            }
            if offset.is_some() || max_chars.is_some() {
                doc.content = get::trim_content(&doc.content, offset, max_chars).to_string();
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&doc)?);
            } else {
                print!("{}", doc.content);
            }
        }
        None => {
            eprintln!("not found: {target}");
            std::process::exit(1);
        }
    }
    Ok(())
}

fn handle_multi_get(
    targets: Vec<String>,
    collections: Vec<String>,
    max_chars: Option<usize>,
    json: bool,
    files: bool,
) -> Result<()> {
    let config = Config::load()?;
    let filter = resolve_collections(&config, &collections)?;
    let mut found: Vec<DocContent> = Vec::new();
    let mut not_found: Vec<String> = Vec::new();
    for target in &targets {
        match get::fetch_document_with_config(target, &filter, &config)? {
            Some(mut doc) => {
                if max_chars.is_some() {
                    doc.content = get::trim_content(&doc.content, None, max_chars).to_string();
                }
                found.push(doc);
            }
            None => not_found.push(target.clone()),
        }
    }
    let has_missing = !not_found.is_empty();
    if json {
        println!("{}", serde_json::to_string_pretty(&MultiGetResult { found, not_found })?);
    } else {
        if files {
            for doc in &found {
                println!("{}", doc.path);
            }
        } else {
            for (i, doc) in found.iter().enumerate() {
                if i > 0 { println!("---"); }
                eprintln!("[{}] {}", doc.collection, doc.path);
                print!("{}", doc.content);
            }
        }
        for path in &not_found {
            eprintln!("not found: {path}");
        }
        if has_missing {
            std::process::exit(1);
        }
    }
    Ok(())
}

fn handle_collection(cmd: CollectionCmd) -> Result<()> {
    let mut config = Config::load()?;
    match cmd {
        CollectionCmd::Add {
            name,
            path,
            glob,
            exclude,
            description,
            preprocessor,
        } => {
            // Validate aliases before mutating config.
            for alias in &preprocessor {
                if !config.preprocessors.contains_key(alias.as_str()) {
                    return Err(error::Error::Other(format!(
                        "preprocessor alias '{alias}' not registered. Run: ir preprocessor add {alias} <command>"
                    )));
                }
            }
            let resolved = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone().into());
            config.add_collection(Collection {
                name: name.clone(),
                path: resolved.to_string_lossy().into_owned(),
                globs: glob,
                excludes: exclude,
                description,
                preprocessor: if preprocessor.is_empty() { None } else { Some(preprocessor) },
            })?;
            config.save()?;
            println!("added collection '{name}'");
        }
        CollectionCmd::Rm { name, purge } => {
            config.remove_collection(&name)?;
            config.save()?;
            if purge {
                let db_path = collection_db_path(&name);
                if db_path.exists() {
                    std::fs::remove_file(&db_path)?;
                    println!("removed collection '{name}' and deleted database");
                } else {
                    println!("removed collection '{name}'");
                }
            } else {
                println!("removed collection '{name}' (database kept)");
            }
        }
        CollectionCmd::Rename { old, new } => {
            config.rename_collection(&old, &new)?;
            config.save()?;
            println!("renamed '{old}' → '{new}'");
        }
        CollectionCmd::SetPath { name, path } => {
            config.set_collection_path(&name, &path)?;
            config.save()?;
            println!("updated path for '{name}' → {path}");
            println!("run `ir daemon stop` then `ir update {name}` to sync");
        }
        CollectionCmd::Ls => {
            if config.collections.is_empty() {
                println!("no collections configured");
            } else {
                for c in &config.collections {
                    let pp = c.preprocessor.as_ref()
                        .filter(|v| !v.is_empty())
                        .map(|v| format!("  [{}]", v.join(", ")))
                        .unwrap_or_default();
                    if let Some(desc) = &c.description {
                        println!("{:<20} {}{}  # {}", c.name, c.path, pp, desc);
                    } else {
                        println!("{:<20} {}{}", c.name, c.path, pp);
                    }
                }
            }
        }
    }
    Ok(())
}

fn handle_status() -> Result<()> {
    let config = Config::load()?;
    println!("collections: {}", config.collections.len());
    for col in &config.collections {
        let db_path = collection_db_path(&col.name);
        let db_exists = db_path.exists();
        let status = if db_exists { "indexed" } else { "not indexed" };
        let size = if db_exists {
            let bytes = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
            format!("{:.1} MB", bytes as f64 / 1_048_576.0)
        } else {
            String::new()
        };
        println!("  {:<20} {:<12} {}  {}", col.name, status, col.path, size);
    }
    Ok(())
}

fn handle_update(collection: Option<String>, force: bool) -> Result<()> {
    let config = Config::load()?;
    let cols: Vec<_> = match &collection {
        Some(name) => {
            let c = config
                .get_collection(name)
                .ok_or_else(|| error::Error::CollectionNotFound(name.clone()))?;
            vec![c]
        }
        None => config.collections.iter().collect(),
    };

    for col in cols {
        let db_path = collection_db_path(&col.name);
        let pp_aliases = col.preprocessor.as_deref().unwrap_or(&[]);
        let has_preprocessor = !config.resolve_preprocessor_commands(pp_aliases).is_empty();
        let db = db::CollectionDb::open(&col.name, &db_path, has_preprocessor)?;
        println!("updating '{}'…", col.name);
        let opts = index::UpdateOptions { force };
        let (added, updated, deactivated) = index::update(&db, col, &opts, &config)?;
        println!(
            "  {} added, {} updated, {} deactivated",
            added, updated, deactivated
        );
    }
    Ok(())
}

/// Search core: runs the tier-0/1/2 pipeline and returns ranked results.
/// Used by both `ir search` and `ir mcp`. Does not print to stdout.
/// `verbosity` controls stderr output -- see `types::Verbosity`.
pub(crate) fn search_core(
    query: &str,
    mode: &str,
    limit: usize,
    min_score: Option<f64>,
    collection_filter: &[String],
    verbosity: types::Verbosity,
) -> Result<Vec<types::SearchResult>> {
    llm::download::prepare_model_envs()?;

    let config = Config::load()?;
    let collection_names = resolve_collections(&config, collection_filter)?;
    let search_mode: SearchMode = mode.parse().map_err(error::Error::Other)?;

    let cols: Vec<_> = collection_names.iter()
        .filter_map(|name| config.get_collection(name))
        .collect();
    let dbs: Vec<db::CollectionDb> = cols.iter()
        .map(|c| {
            let pp_aliases = c.preprocessor.as_deref().unwrap_or(&[]);
            let pp_commands = config.resolve_preprocessor_commands(pp_aliases);
            db::CollectionDb::open_rw(&c.name, &collection_db_path(&c.name), pp_commands)
        })
        .collect::<Result<Vec<_>>>()?;

    let bm25_req = search::fan_out::SearchRequest { query, limit, min_score };
    let bm25_results = search::fan_out::bm25(&dbs, &bm25_req)?;

    match search_mode {
        SearchMode::Bm25 => return Ok(bm25_results),
        SearchMode::Vector => {}
        SearchMode::Hybrid => {
            if search::hybrid::is_bm25_strong_signal(&bm25_results) {
                if !daemon::is_running() {
                    llm::download::prepare_model_envs()?;
                    let _ = daemon::start_in_background();
                }
                return Ok(bm25_results);
            }
        }
    }

    if !daemon::is_running() {
        llm::download::prepare_model_envs()?;
        if let Err(e) = daemon::start_in_background() {
            if verbosity.show_progress() { eprintln!("note: could not start daemon ({e})"); }
            return Ok(bm25_results);
        }
    }

    let req = daemon::DaemonRequest {
        query: query.to_string(),
        collections: collection_names,
        limit,
        min_score,
        mode: mode.to_string(),
        verbose: verbosity.daemon_verbose(),
    };

    let log_lines = |lines: &[String]| {
        if verbosity.show_logs() {
            for line in lines { eprintln!("{line}"); }
        }
    };

    if verbosity.show_progress() { eprint!("searching..."); }
    if !daemon::wait_ready(3_000) {
        if verbosity.show_progress() { eprintln!(); }
        return Ok(bm25_results);
    }

    let tier2_before = daemon::is_tier2_ready();

    let tier1 = match daemon::query(&req) {
        Ok(r) => r,
        Err(e) => {
            if verbosity.show_progress() { eprintln!("\nnote: daemon query error: {e}"); }
            return Ok(bm25_results);
        }
    };

    if tier2_before || search_mode != SearchMode::Hybrid {
        if verbosity.show_progress() { eprintln!(); }
        log_lines(&tier1.log);
        return Ok(to_search_results(tier1.results));
    }

    let tier1_log = tier1.log;
    let tier1_results = to_search_results(tier1.results);
    if search::hybrid::is_strong_signal(&tier1_results) {
        if verbosity.show_progress() { eprintln!(); }
        log_lines(&tier1_log);
        return Ok(tier1_results);
    }

    if verbosity.show_progress() { eprint!(" enhancing..."); }
    if !daemon::wait_tier2(7_000) {
        if verbosity.show_progress() { eprintln!(); }
        log_lines(&tier1_log);
        return Ok(tier1_results);
    }

    match daemon::query(&req) {
        Ok(tier2) => {
            if verbosity.show_progress() { eprintln!(); }
            log_lines(&tier2.log);
            Ok(to_search_results(tier2.results))
        }
        Err(_) => {
            if verbosity.show_progress() { eprintln!(); }
            log_lines(&tier1_log);
            Ok(tier1_results)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_search(
    query: String,
    mode: String,
    limit: usize,
    min_score: Option<f64>,
    collection_filter: Vec<String>,
    full: bool,
    chunk: bool,
    json: bool,
    csv: bool,
    md: bool,
    files: bool,
    verbose: bool,
) -> Result<()> {
    let fmt = if json {
        output::Format::Json
    } else if csv {
        output::Format::Csv
    } else if md {
        output::Format::Markdown
    } else if files {
        output::Format::Files
    } else {
        output::Format::Pretty
    };

    let verbosity = if verbose { types::Verbosity::Verbose } else { types::Verbosity::Normal };
    let mut results = search_core(&query, &mode, limit, min_score, &collection_filter, verbosity)?;

    if full {
        let config = Config::load()?;
        let cols: Vec<_> = results.iter()
            .map(|r| r.collection.as_str())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .filter_map(|name| config.get_collection(name))
            .collect();
        let dbs: Vec<db::CollectionDb> = cols.iter()
            .map(|c| {
                let pp_aliases = c.preprocessor.as_deref().unwrap_or(&[]);
                let pp_commands = config.resolve_preprocessor_commands(pp_aliases);
                db::CollectionDb::open_rw(&c.name, &collection_db_path(&c.name), pp_commands)
            })
            .collect::<Result<Vec<_>>>()?;
        fill_content(&mut results, &dbs);
    } else if chunk {
        get::populate_chunk_content(&mut results)?;
    }

    output::print_results(&mut results, fmt);
    Ok(())
}

fn to_search_results(daemon_results: Vec<daemon::DaemonResult>) -> Vec<types::SearchResult> {
    daemon_results.into_iter()
        .map(|r| types::SearchResult {
            collection: r.collection,
            path: r.path,
            title: r.title,
            score: r.score,
            snippet: if r.snippet.is_empty() { None } else { Some(r.snippet) },
            hash: r.hash,
            doc_id: r.doc_id,
            content: None,
            chunk_seq: r.chunk_seq,
        })
        .collect()
}

pub(crate) fn fill_content(results: &mut [types::SearchResult], dbs: &[db::CollectionDb]) {
    let db_map: std::collections::HashMap<&str, &db::CollectionDb> =
        dbs.iter().map(|d| (d.name.as_str(), d)).collect();

    // Group unique hashes by collection for batch queries.
    let mut per_col: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for r in results.iter() {
        if db_map.contains_key(r.collection.as_str()) {
            per_col.entry(r.collection.clone()).or_default().push(r.hash.clone());
        }
    }

    // One SELECT ... IN (...) per collection.
    let mut content_cache: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for (col_name, hashes) in &per_col {
        let mut unique: Vec<&str> = hashes.iter().map(String::as_str).collect();
        unique.sort_unstable();
        unique.dedup();
        if let Some(db) = db_map.get(col_name.as_str()) {
            content_cache.extend(db::fetch_content_batch(db.conn(), &unique));
        }
    }

    for r in results.iter_mut() {
        r.content = content_cache.get(&r.hash).cloned();
    }
}

fn resolve_collections(config: &Config, filter: &[String]) -> Result<Vec<String>> {
    if filter.is_empty() {
        let cwd = std::env::current_dir().unwrap_or_default();
        if let Some(col) = config::detect_collection(&config.collections, &cwd) {
            Ok(vec![col.name.clone()])
        } else {
            Ok(config.collections.iter().map(|c| c.name.clone()).collect())
        }
    } else {
        let unknown: Vec<&str> = filter
            .iter()
            .filter(|name| config.get_collection(name).is_none())
            .map(|s| s.as_str())
            .collect();
        if !unknown.is_empty() {
            return Err(error::Error::Other(format!(
                "unknown collection(s): {}",
                unknown.join(", ")
            )));
        }
        Ok(filter.to_vec())
    }
}

fn handle_preprocessor(cmd: PreprocessorCmd) -> Result<()> {
    let mut config = Config::load()?;
    match cmd {
        PreprocessorCmd::Add { alias, command } => {
            if command.is_empty() {
                return Err(error::Error::Other("command must not be empty".into()));
            }
            let cmd_str = command.join(" ");
            config.add_preprocessor(&alias, &cmd_str)?;
            config.save()?;
            println!("registered preprocessor '{alias}': {cmd_str}");
        }
        PreprocessorCmd::Install { lang } => {
            install_preprocessor(&mut config, &lang)?;
        }
        PreprocessorCmd::List => {
            let known = known_preprocessors();
            let mut entries: Vec<_> = config.preprocessors.iter().collect();
            entries.sort_by_key(|(k, _)| k.as_str());
            if !entries.is_empty() {
                println!("registered:");
                for (alias, cmd) in &entries {
                    println!("  {:<10} {}", alias, cmd);
                }
            }
            let uninstalled: Vec<_> = known
                .iter()
                .filter(|k| !config.preprocessors.contains_key(k.alias))
                .collect();
            if !uninstalled.is_empty() {
                if !entries.is_empty() { println!(); }
                println!("available (ir preprocessor install <lang>):");
                for k in uninstalled {
                    println!("  {:<10} {}", k.alias, k.description);
                }
            }
            println!();
            println!("  custom: ir preprocessor add <alias> <command>");
        }
        PreprocessorCmd::Bind { alias, collection } => {
            if !config.preprocessors.contains_key(&alias) {
                let known_aliases: Vec<&str> = known_preprocessors().iter().map(|k| k.alias).collect();
                let hint = if known_aliases.contains(&alias.as_str()) {
                    format!("run: ir preprocessor install {alias}")
                } else {
                    format!("run: ir preprocessor add {alias} <command>  (or `ir preprocessor list` to see options)")
                };
                return Err(error::Error::Other(format!(
                    "preprocessor alias '{alias}' not registered — {hint}"
                )));
            }
            let targets = match collection {
                Some(name) => vec![name],
                None => pick_collections_for_bind(&config, &alias)?,
            };
            for name in targets {
                let col = config.collections.iter_mut()
                    .find(|c| c.name == name)
                    .ok_or_else(|| error::Error::Other(format!("collection '{name}' not found")))?;
                let pp = col.preprocessor.get_or_insert_with(Vec::new);
                if !pp.contains(&alias) { pp.push(alias.clone()); }
                config.save()?;
                println!("bound '{alias}' to '{name}', re-indexing…");
                if let Err(e) = handle_update(Some(name.clone()), false) {
                    eprintln!("warning: re-index failed for '{name}': {e}");
                }
            }
        }
        PreprocessorCmd::Unbind { alias, collection } => {
            let col = config.collections.iter_mut()
                .find(|c| c.name == collection)
                .ok_or_else(|| error::Error::Other(format!("collection '{collection}' not found")))?;
            let pp = col.preprocessor.get_or_insert_with(Vec::new);
            if !pp.contains(&alias) {
                println!("'{alias}' not bound to '{collection}'");
            } else {
                pp.retain(|a| a != &alias);
                if pp.is_empty() { col.preprocessor = None; }
                config.save()?;
                println!("unbound '{alias}' from '{collection}', re-indexing…");
                handle_update(Some(collection), false)?;
            }
        }
        PreprocessorCmd::Remove { alias, delete } => {
            let cmd = config.preprocessors.get(&alias).cloned();
            config.remove_preprocessor(&alias)?;
            config.save()?;
            if delete {
                if let Some(cmd_str) = cmd {
                    let path = std::path::Path::new(&cmd_str);
                    let preprocess_dir = config::ir_dir().join("preprocessors");
                    if path.starts_with(&preprocess_dir) && path.is_file() {
                        std::fs::remove_file(path).map_err(error::Error::Io)?;
                        println!("deleted {}", path.display());
                    } else {
                        println!("note: '{cmd_str}' is outside the ir preprocessors dir, not deleted");
                    }
                }
            }
            println!("removed preprocessor '{alias}'");
        }
    }
    Ok(())
}

enum PreprocessorKind {
    Binary { binary_name: &'static str },
    Script { repo_subdir: &'static str, script_name: &'static str },
}
struct KnownPreprocessor {
    alias: &'static str,
    description: &'static str,
    kind: PreprocessorKind,
}

fn known_preprocessors() -> &'static [KnownPreprocessor] {
    &[
        KnownPreprocessor {
            alias: "ko",
            description: "Korean morphological analysis (Lindera + ko-dic)",
            kind: PreprocessorKind::Binary { binary_name: "lindera-tokenize" },
        },
        KnownPreprocessor {
            alias: "ja",
            description: "Japanese morphological analysis (Lindera + ipadic)",
            kind: PreprocessorKind::Binary { binary_name: "lindera-tokenize-ja" },
        },
        KnownPreprocessor {
            alias: "zh",
            description: "Chinese bigram tokenization",
            kind: PreprocessorKind::Binary { binary_name: "bigram-tokenize-zh" },
        },
    ]
}

/// Interactively pick collections to bind an alias to.
/// Shows all collections with current preprocessors; pre-checks ones already bound.
/// Returns selected collection names.
fn pick_collections_for_bind(config: &Config, alias: &str) -> Result<Vec<String>> {
    if config.collections.is_empty() {
        println!("no collections configured");
        return Ok(vec![]);
    }
    let items: Vec<String> = config.collections.iter().map(|c| {
        let pp = match c.preprocessor.as_deref() {
            Some(pp) if !pp.is_empty() => format!(" [{}]", pp.join(", ")),
            _ => String::new(),
        };
        format!("{}{}", c.name, pp)
    }).collect();
    let defaults: Vec<bool> = config.collections.iter()
        .map(|c| c.preprocessor.as_deref().unwrap_or(&[]).contains(&alias.to_string()))
        .collect();
    let selections = dialoguer::MultiSelect::new()
        .with_prompt(format!("bind '{alias}' to collections (space to toggle, enter to confirm)"))
        .items(&items)
        .defaults(&defaults)
        .interact()
        .map_err(|e| error::Error::Other(format!("prompt: {e}")))?;
    Ok(selections.into_iter().map(|i| config.collections[i].name.clone()).collect())
}

/// Download/install a bundled preprocessor and register it.
fn install_preprocessor(config: &mut Config, lang: &str) -> Result<()> {
    let known = known_preprocessors();

    let available: Vec<&str> = known.iter().map(|e| e.alias).collect();
    let entry = known
        .iter()
        .find(|e| e.alias == lang)
        .ok_or_else(|| error::Error::Other(
            format!("unknown lang '{lang}'. Available: {}", available.join(", "))
        ))?;

    let install_dir = config::ir_dir().join("preprocessors").join(entry.alias);
    std::fs::create_dir_all(&install_dir)?;

    let cmd_str = match &entry.kind {
        PreprocessorKind::Binary { binary_name } => {
            let bin_path = install_dir.join(binary_name);
            let arch = if cfg!(target_os = "macos") {
                "darwin-universal"
            } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
                "linux-x86_64"
            } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
                "linux-aarch64"
            } else {
                return Err(error::Error::Other(
                    "preprocessor install is only supported on macOS and Linux (x86_64/aarch64)".into()
                ));
            };
            let tarball = format!("{binary_name}-{arch}.tar.gz");
            let url = format!(
                "https://github.com/vlwkaos/ir/releases/latest/download/{tarball}"
            );
            let tar_path = install_dir.join(&tarball);
            let status = std::process::Command::new("curl")
                .args(["-fsSL", &url, "-o", &tar_path.to_string_lossy()])
                .status()
                .map_err(|e| error::Error::Other(format!("curl: {e}")))?;
            if !status.success() {
                return Err(error::Error::Other(format!(
                    "download failed: {url}"
                )));
            }
            let tar_status = std::process::Command::new("tar")
                .args(["-xzf", &tar_path.to_string_lossy(), "-C", &install_dir.to_string_lossy()])
                .status()
                .map_err(|e| error::Error::Other(format!("tar: {e}")))?;
            if !tar_status.success() {
                std::fs::remove_file(&tar_path).ok();
                return Err(error::Error::Other(format!(
                    "failed to extract {tarball}: tar exited with {}", tar_status
                )));
            }
            std::fs::remove_file(&tar_path).ok();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&bin_path, std::fs::Permissions::from_mode(0o755))
                    .map_err(error::Error::Io)?;
            }
            bin_path.to_string_lossy().into_owned()
        }
        PreprocessorKind::Script { repo_subdir, script_name } => {
            let script_path = install_dir.join(script_name);
            let url = format!(
                "https://raw.githubusercontent.com/vlwkaos/ir/main/preprocessors/{repo_subdir}/{script_name}"
            );
            let status = std::process::Command::new("curl")
                .args(["-fsSL", &url, "-o", &script_path.to_string_lossy()])
                .status()
                .map_err(|e| error::Error::Other(format!("curl: {e}")))?;
            if !status.success() {
                return Err(error::Error::Other(format!(
                    "download failed. Install manually to {}", script_path.display()
                )));
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755))
                    .map_err(error::Error::Io)?;
            }
            script_path.to_string_lossy().into_owned()
        }
    };

    let alias = entry.alias;
    config.add_preprocessor(alias, &cmd_str)?;
    config.save()?;
    println!("installed '{alias}' preprocessor → {cmd_str}");

    if !config.collections.is_empty() {
        println!();
        let targets = pick_collections_for_bind(config, alias)?;
        for name in targets {
            let col = config.collections.iter_mut()
                .find(|c| c.name == name).unwrap();
            let pp = col.preprocessor.get_or_insert_with(Vec::new);
            if !pp.contains(&alias.to_string()) { pp.push(alias.to_string()); }
            config.save()?;
            println!("bound '{alias}' to '{name}', re-indexing…");
            handle_update(Some(name), false)?;
        }
    }

    Ok(())
}

fn handle_embed(collection: Option<String>, force: bool) -> Result<()> {
    let config = Config::load()?;
    let cols: Vec<_> = match &collection {
        Some(name) => {
            let c = config
                .get_collection(name)
                .ok_or_else(|| error::Error::CollectionNotFound(name.clone()))?;
            vec![c]
        }
        None => config.collections.iter().collect(),
    };

    llm::download::prepare_model_envs()?;
    println!("loading embedding model…");
    let embedder = llm::embedding::Embedder::load_default()?;

    for col in cols {
        let db_path = collection_db_path(&col.name);
        let pp_aliases = col.preprocessor.as_deref().unwrap_or(&[]);
        let has_preprocessor = !config.resolve_preprocessor_commands(pp_aliases).is_empty();
        let db = db::CollectionDb::open(&col.name, &db_path, has_preprocessor)?;
        println!("embedding '{}'…", col.name);
        let opts = index::embed::EmbedOptions { force };
        let (docs, chunks) = index::embed::embed(&db, &embedder, &opts, llm::models::EMBEDDING)?;
        println!("  {} documents, {} chunks embedded", docs, chunks);
    }
    Ok(())
}
