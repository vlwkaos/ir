mod cli;
mod config;
mod daemon;
mod db;
mod error;
mod frontmatter;
mod get;
mod index;
mod llm;
mod mcp;
mod preprocess;
mod search;
mod types;

use clap::Parser;
use cli::{Cli, CollectionCmd, Command, DaemonCmd, PreprocessorCmd, output};
use config::{Config, collection_db_path};
use error::Result;
use get::{DocContent, MultiGetResult};
use std::path::{Path, PathBuf};
use types::{Collection, SearchMode};

fn ko_default_routing() -> types::RoutingConfig {
    types::RoutingConfig {
        fused_strong_floor: None,
        fused_strong_product: Some(0.05),
        bm25_strong_floor: None,
        bm25_strong_gap: None,
    }
}

fn apply_bind_defaults(col: &mut Collection, alias: &str) {
    if alias == "ko" && col.routing.is_none() {
        col.routing = Some(ko_default_routing());
    }
}

fn clear_bind_defaults_if_auto(col: &mut Collection, alias: &str) {
    if alias == "ko" && col.routing.as_ref() == Some(&ko_default_routing()) {
        col.routing = None;
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    // ^ Set IR_DIR so preprocessor commands stored as "$IR_DIR/..." expand correctly at runtime.
    // Must happen before any config load.
    unsafe { std::env::set_var("IR_DIR", config::ir_dir()) };
    let cli = Cli::parse();
    // `--config-path` overrides only the config file (not the data dir). Set the
    // internal transport var before any Config::load(); the spawned daemon
    // inherits it, so client and daemon resolve the same config file.
    if let Some(path) = cli.config_path.as_ref() {
        unsafe { std::env::set_var("IR_CONFIG_FILE", path) };
    }
    match cli.command {
        Command::Collection { cmd } => handle_collection(cmd),
        Command::Status => handle_status(),
        Command::Sync { collection, force } => handle_sync(collection, force),
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
            quiet,
            filter,
        } => handle_search(
            query.join(" "),
            mode,
            if all {
                crate::db::vectors::KNN_MAX
            } else {
                limit
            },
            min_score,
            collections,
            full,
            chunk,
            json,
            csv,
            md,
            files,
            if verbose {
                types::Verbosity::Verbose
            } else if quiet {
                types::Verbosity::Quiet
            } else {
                types::Verbosity::Normal
            },
            filter,
        ),
        Command::Get {
            target,
            collections,
            section,
            offset,
            max_chars,
            json,
        } => handle_get(target, collections, section, offset, max_chars, json),
        Command::MultiGet {
            targets,
            collections,
            max_chars,
            json,
            files,
        } => handle_multi_get(targets, collections, max_chars, json, files),
        Command::Daemon { cmd } => match cmd {
            DaemonCmd::Start { timeout } => daemon::start_server(timeout),
            DaemonCmd::Stop => daemon::stop(),
            DaemonCmd::Status => daemon::status(),
        },
        Command::Preprocessor { cmd } => handle_preprocessor(cmd),
        Command::Mcp { http, cors } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| error::Error::Other(e.to_string()))?;
            rt.block_on(mcp::run(http, cors))
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
        println!(
            "{}",
            serde_json::to_string_pretty(&MultiGetResult { found, not_found })?
        );
    } else {
        if files {
            for doc in &found {
                println!("{}", doc.path);
            }
        } else {
            for (i, doc) in found.iter().enumerate() {
                if i > 0 {
                    println!("---");
                }
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
            let store_path = config::portable_path(&path)?;
            config.add_collection(Collection {
                name: name.clone(),
                path: store_path,
                globs: glob,
                excludes: exclude,
                description,
                preprocessor: if preprocessor.is_empty() {
                    None
                } else {
                    Some(preprocessor)
                },
                routing: None,
                retrieval: None,
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
                    let pp = c
                        .preprocessor
                        .as_ref()
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
    let cols = index_collections(&config, collection.as_deref())?;

    for col in cols {
        update_collection(&config, col, force)?;
    }
    Ok(())
}

fn update_collection(
    config: &Config,
    collection: &Collection,
    force: bool,
) -> Result<db::CollectionDb> {
    let db_path = collection_db_path(&collection.name);
    let pp_aliases = collection.preprocessor.as_deref().unwrap_or(&[]);
    let has_preprocessor = !config.resolve_preprocessor_commands(pp_aliases).is_empty();
    let db = db::CollectionDb::open(&collection.name, &db_path, has_preprocessor)?;
    println!("updating '{}'…", collection.name);
    let opts = index::UpdateOptions { force };
    let (added, updated, deactivated) = index::update(&db, collection, &opts, config)?;
    println!(
        "  {} added, {} updated, {} deactivated",
        added, updated, deactivated
    );
    Ok(db)
}

fn index_collections<'a>(
    config: &'a Config,
    collection: Option<&str>,
) -> Result<Vec<&'a Collection>> {
    match collection {
        Some(name) => config
            .get_collection(name)
            .map(|c| vec![c])
            .ok_or_else(|| error::Error::CollectionNotFound(name.to_string())),
        None => Ok(config.collections.iter().collect()),
    }
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
    filter: types::Filter,
) -> Result<Vec<types::SearchResult>> {
    let config = Config::load()?;
    let collection_names = resolve_collections(&config, collection_filter)?;
    let search_mode: SearchMode = mode.parse().map_err(error::Error::Other)?;
    let daemon_was_running = daemon::is_running();
    let daemon_warmup_error = if daemon_was_running {
        None
    } else {
        daemon::start_in_background().err().map(|e| e.to_string())
    };
    let daemon_warmup_started = !daemon_was_running && daemon_warmup_error.is_none();

    let cols: Vec<_> = collection_names
        .iter()
        .filter_map(|name| config.get_collection(name))
        .collect();
    let dbs: Vec<db::CollectionDb> = cols
        .iter()
        .map(|c| {
            let pp_aliases = c.preprocessor.as_deref().unwrap_or(&[]);
            let pp_commands = config.resolve_preprocessor_commands(pp_aliases);
            db::CollectionDb::open_rw(
                &c.name,
                &collection_db_path(&c.name),
                pp_commands,
                c.routing.clone(),
                c.retrieval.clone(),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    // Tier-0: BM25 with over-fetch when filter is active
    let fetch_limit = if filter.is_empty() {
        limit
    } else {
        (limit * search::filter::over_fetch_multiplier(&filter)).clamp(50, 500)
    };

    let bm25_req = search::fan_out::SearchRequest {
        query,
        limit: fetch_limit,
        min_score: None, // ^ applied after tier-0 filter below
    };
    let mut bm25_results = search::fan_out::bm25(&dbs, &bm25_req)?;

    // Tier-0 kNN-graph expansion of BM25 seeds (profile.t0_graph_expand;
    // config > env > default). Runs before the tier-0 filter so injected docs
    // are filtered like any other.
    let t0_profile = search::profile::resolve_for_query(dbs.iter().map(|d| d.retrieval.as_ref()));
    search::graph::maybe_expand_t0(
        &dbs,
        &mut bm25_results,
        fetch_limit,
        t0_profile.t0_graph_expand,
    );

    // Tier-0 filter: apply before BM25 strong-signal check
    search::filter::apply(&mut bm25_results, &filter, &dbs)?;
    if let Some(min) = min_score {
        bm25_results.retain(|r| r.score >= min);
    }
    bm25_results.truncate(limit);

    // Research instrumentation: emit BM25 signal data for threshold calibration.
    // Activated by IR_BENCH_SIGNALS=1; no-op in normal use.
    if std::env::var("IR_BENCH_SIGNALS").is_ok() {
        let top = bm25_results.first().map(|r| r.score).unwrap_or(0.0);
        let gap = if bm25_results.len() >= 2 {
            top - bm25_results[1].score
        } else {
            top
        };
        // Emit top-10 scores for relative dominance and percentile analysis
        let scores: Vec<String> = bm25_results
            .iter()
            .take(10)
            .map(|r| format!("{:.6}", r.score))
            .collect();
        if scores.is_empty() {
            eprintln!("SIGNAL_BM25\t{top:.6}\t{gap:.6}");
        } else {
            eprintln!("SIGNAL_BM25\t{top:.6}\t{gap:.6}\t{}", scores.join(","));
        }
    }
    let disable_shortcuts = std::env::var("IR_DISABLE_SHORTCUTS").is_ok();

    let (bm25_strong_floor, bm25_strong_gap) =
        search::hybrid::bm25_strong_signal_thresholds_for_collections(&cols);

    match search_mode {
        SearchMode::Bm25 => return Ok(bm25_results),
        SearchMode::Vector => {}
        SearchMode::Hybrid => {
            // Only shortcut if post-filter count meets limit (else escalate for more candidates)
            if !disable_shortcuts
                && search::hybrid::is_bm25_strong_signal(
                    &bm25_results,
                    bm25_strong_floor,
                    bm25_strong_gap,
                )
                && (filter.is_empty() || bm25_results.len() >= limit)
            {
                return Ok(bm25_results);
            }
        }
    }

    if daemon_warmup_started && !bm25_results.is_empty() {
        if verbosity.show_progress() {
            eprintln!("note: daemon warming in background — returning BM25 for this query");
        }
        return Ok(bm25_results);
    }

    if !daemon::is_running() {
        if let Err(e) = daemon::start_in_background() {
            if verbosity.show_progress() {
                eprintln!("note: could not start daemon ({e})");
                if bm25_results.is_empty() {
                    eprintln!(
                        "note: BM25 also found nothing — run `ir embed <collection>` to enable vector search"
                    );
                }
            }
            return Ok(bm25_results);
        }
    } else if let Some(err) = daemon_warmup_error
        && verbosity.show_progress()
    {
        eprintln!("note: background daemon warmup failed ({err})");
    }

    let req = daemon::DaemonRequest {
        query: query.to_string(),
        collections: collection_names,
        limit,
        min_score,
        mode: mode.to_string(),
        verbose: verbosity.daemon_verbose(),
        filter: filter.clauses,
    };
    let (strong_signal_floor, strong_signal_product) =
        search::hybrid::strong_signal_thresholds_for_collections(&cols);

    // SIGNAL_ lines always re-emitted to stderr (picked up by beir-eval --signals).
    // Other log lines gated on verbosity as usual.
    let log_lines = |lines: &[String]| {
        for line in lines {
            if line.starts_with("SIGNAL_") || verbosity.show_logs() {
                eprintln!("{line}");
            }
        }
    };

    if verbosity.show_progress() {
        eprint!("searching...");
    }
    // When BM25 found nothing, the daemon is the only source of results — wait longer.
    let wait_ms = if bm25_results.is_empty() {
        10_000
    } else {
        3_000
    };
    if !daemon::wait_ready(wait_ms) {
        if verbosity.show_progress() {
            eprintln!();
            if bm25_results.is_empty() {
                eprintln!(
                    "note: BM25 found no results and daemon not ready — try `ir embed <collection>` or check model paths"
                );
            }
        }
        return Ok(bm25_results);
    }

    let tier2_before = daemon::is_tier2_ready();

    let tier1 = match daemon::query(&req) {
        Ok(r) => r,
        Err(e) => {
            if verbosity.show_progress() {
                eprintln!("\nnote: daemon query error: {e}");
                if bm25_results.is_empty() {
                    eprintln!("note: BM25 also found nothing — check collection embeddings");
                }
            }
            return Ok(bm25_results);
        }
    };

    if tier2_before || search_mode != SearchMode::Hybrid {
        if verbosity.show_progress() {
            eprintln!();
        }
        log_lines(&tier1.log);
        return Ok(to_search_results(tier1.results));
    }

    let tier1_log = tier1.log;
    let tier1_results = to_search_results(tier1.results);
    if !disable_shortcuts
        && search::hybrid::is_strong_signal(
            &tier1_results,
            strong_signal_floor,
            strong_signal_product,
        )
    {
        if verbosity.show_progress() {
            eprintln!();
        }
        log_lines(&tier1_log);
        return Ok(tier1_results);
    }

    if verbosity.show_progress() {
        eprint!(" enhancing...");
    }
    if !daemon::wait_tier2(7_000) {
        if verbosity.show_progress() {
            eprintln!();
        }
        log_lines(&tier1_log);
        return Ok(tier1_results);
    }

    match daemon::query(&req) {
        Ok(tier2) => {
            if verbosity.show_progress() {
                eprintln!();
            }
            log_lines(&tier2.log);
            Ok(to_search_results(tier2.results))
        }
        Err(_) => {
            if verbosity.show_progress() {
                eprintln!();
            }
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
    verbosity: types::Verbosity,
    filter_strs: Vec<String>,
) -> Result<()> {
    let filter = types::Filter::parse(&filter_strs).map_err(error::Error::Other)?;

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

    let mut results = search_core(
        &query,
        &mode,
        limit,
        min_score,
        &collection_filter,
        verbosity,
        filter,
    )?;

    if full {
        let config = Config::load()?;
        let cols: Vec<_> = results
            .iter()
            .map(|r| r.collection.as_str())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .filter_map(|name| config.get_collection(name))
            .collect();
        let dbs: Vec<db::CollectionDb> = cols
            .iter()
            .map(|c| {
                let pp_aliases = c.preprocessor.as_deref().unwrap_or(&[]);
                let pp_commands = config.resolve_preprocessor_commands(pp_aliases);
                db::CollectionDb::open_rw(
                    &c.name,
                    &collection_db_path(&c.name),
                    pp_commands,
                    c.routing.clone(),
                    c.retrieval.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        fill_content(&mut results, &dbs);
    } else if chunk {
        get::populate_chunk_content(&mut results)?;
    }

    output::print_results(&results, fmt);
    Ok(())
}

fn to_search_results(daemon_results: Vec<daemon::DaemonResult>) -> Vec<types::SearchResult> {
    daemon_results
        .into_iter()
        .map(|r| types::SearchResult {
            collection: r.collection,
            path: r.path,
            title: r.title,
            score: r.score,
            snippet: if r.snippet.is_empty() {
                None
            } else {
                Some(r.snippet)
            },
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
            per_col
                .entry(r.collection.clone())
                .or_default()
                .push(r.hash.clone());
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
                if !entries.is_empty() {
                    println!();
                }
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
                let known_aliases: Vec<&str> =
                    known_preprocessors().iter().map(|k| k.alias).collect();
                let hint = if known_aliases.contains(&alias.as_str()) {
                    format!("run: ir preprocessor install {alias}")
                } else {
                    format!(
                        "run: ir preprocessor add {alias} <command>  (or `ir preprocessor list` to see options)"
                    )
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
                let col = config
                    .collections
                    .iter_mut()
                    .find(|c| c.name == name)
                    .ok_or_else(|| error::Error::Other(format!("collection '{name}' not found")))?;
                let pp = col.preprocessor.get_or_insert_with(Vec::new);
                if !pp.contains(&alias) {
                    pp.push(alias.clone());
                }
                apply_bind_defaults(col, &alias);
                config.save()?;
                println!("bound '{alias}' to '{name}', re-indexing…");
                if let Err(e) = handle_update(Some(name.clone()), false) {
                    eprintln!("warning: re-index failed for '{name}': {e}");
                }
            }
        }
        PreprocessorCmd::Unbind { alias, collection } => {
            let col = config
                .collections
                .iter_mut()
                .find(|c| c.name == collection)
                .ok_or_else(|| {
                    error::Error::Other(format!("collection '{collection}' not found"))
                })?;
            let pp = col.preprocessor.get_or_insert_with(Vec::new);
            if !pp.contains(&alias) {
                println!("'{alias}' not bound to '{collection}'");
            } else {
                pp.retain(|a| a != &alias);
                if pp.is_empty() {
                    col.preprocessor = None;
                }
                clear_bind_defaults_if_auto(col, &alias);
                config.save()?;
                println!("unbound '{alias}' from '{collection}', re-indexing…");
                handle_update(Some(collection), false)?;
            }
        }
        PreprocessorCmd::Remove { alias, delete } => {
            let cmd = config.preprocessors.get(&alias).cloned();
            config.remove_preprocessor(&alias)?;
            config.save()?;
            if delete && let Some(cmd_str) = cmd {
                let path = std::path::Path::new(&cmd_str);
                let preprocess_dir = config::ir_dir().join("preprocessors");
                if path.starts_with(&preprocess_dir) && path.is_file() {
                    std::fs::remove_file(path).map_err(error::Error::Io)?;
                    println!("deleted {}", path.display());
                } else {
                    println!("note: '{cmd_str}' is outside the ir preprocessors dir, not deleted");
                }
            }
            println!("removed preprocessor '{alias}'");
        }
    }
    Ok(())
}

struct KnownPreprocessor {
    alias: &'static str,
    description: &'static str,
    // ^ lindera release asset prefix (e.g. "ko-dic" → lindera-ko-dic-{ver}.zip)
    dict_name: &'static str,
    // ^ compact JSON passed as --token-filter arg (no spaces); None = raw wakati
    token_filter: Option<&'static str>,
}

fn known_preprocessors() -> &'static [KnownPreprocessor] {
    &[
        KnownPreprocessor {
            alias: "ko",
            description: "Korean morphological analysis (Lindera + ko-dic)",
            dict_name: "ko-dic",
            token_filter: Some(
                r#"korean_stop_tags:{"tags":["JKS","JKC","JKG","JKO","JKB","JKV","JKQ","JX","JC","EP","EF","EC","ETN","ETM","XPN","XSN","XSV","XSA","SF","SP","SS","SE","SO","SW","SWK"]}"#,
            ),
        },
        KnownPreprocessor {
            alias: "ja",
            description: "Japanese morphological analysis (Lindera + ipadic)",
            dict_name: "ipadic",
            token_filter: Some(
                r#"japanese_stop_tags:{"tags":["接続詞","助詞","助動詞","記号","フィラー","非言語音","その他,間投"]}"#,
            ),
        },
        KnownPreprocessor {
            alias: "zh",
            description: "Chinese word segmentation (Lindera + jieba)",
            dict_name: "jieba",
            token_filter: None,
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
    let items: Vec<String> = config
        .collections
        .iter()
        .map(|c| {
            let pp = match c.preprocessor.as_deref() {
                Some(pp) if !pp.is_empty() => format!(" [{}]", pp.join(", ")),
                _ => String::new(),
            };
            format!("{}{}", c.name, pp)
        })
        .collect();
    let defaults: Vec<bool> = config
        .collections
        .iter()
        .map(|c| {
            c.preprocessor
                .as_deref()
                .unwrap_or(&[])
                .contains(&alias.to_string())
        })
        .collect();
    let selections = dialoguer::MultiSelect::new()
        .with_prompt(format!(
            "bind '{alias}' to collections (space to toggle, enter to confirm)"
        ))
        .items(&items)
        .defaults(&defaults)
        .interact()
        .map_err(|e| error::Error::Other(format!("prompt: {e}")))?;
    Ok(selections
        .into_iter()
        .map(|i| config.collections[i].name.clone())
        .collect())
}

/// Download official lindera CLI binary + language dictionary, register command.
fn install_preprocessor(config: &mut Config, lang: &str) -> Result<()> {
    let known = known_preprocessors();
    let available: Vec<&str> = known.iter().map(|e| e.alias).collect();
    let entry = known.iter().find(|e| e.alias == lang).ok_or_else(|| {
        error::Error::Other(format!(
            "unknown lang '{lang}'. Available: {}",
            available.join(", ")
        ))
    })?;

    let triple = lindera_platform_triple()?;
    let tag = LINDERA_VERSION;
    let version = tag.trim_start_matches('v');

    let preprocessors_dir = config::ir_dir().join("preprocessors");
    std::fs::create_dir_all(&preprocessors_dir)?;

    let bin_path = install_lindera_binary(&preprocessors_dir, tag, triple)?;
    let dict_path = install_lindera_dict(&preprocessors_dir, entry.dict_name, tag, version)?;

    // Store paths as $IR_DIR/preprocessors/... so config.yml is portable across machines.
    let bin_rel = bin_path
        .strip_prefix(&preprocessors_dir)
        .expect("install_lindera_binary returns path under preprocessors_dir");
    let dict_rel = dict_path
        .strip_prefix(&preprocessors_dir)
        .expect("install_lindera_dict returns path under preprocessors_dir");
    let mut cmd_str = format!(
        "$IR_DIR/preprocessors/{} tokenize --dict $IR_DIR/preprocessors/{} -o wakati -m decompose",
        bin_rel.display(),
        dict_rel.display(),
    );
    if let Some(filter) = entry.token_filter {
        cmd_str.push_str(" --token-filter ");
        cmd_str.push_str(filter);
    }

    let alias = entry.alias;
    config.add_preprocessor(alias, &cmd_str)?;
    config.save()?;
    println!("installed '{alias}' preprocessor (lindera {tag})");
    println!("  → {cmd_str}");

    if !config.collections.is_empty() {
        println!();
        let targets = pick_collections_for_bind(config, alias)?;
        for name in targets {
            let col = config
                .collections
                .iter_mut()
                .find(|c| c.name == name)
                .unwrap();
            let pp = col.preprocessor.get_or_insert_with(Vec::new);
            if !pp.contains(&alias.to_string()) {
                pp.push(alias.to_string());
            }
            apply_bind_defaults(col, alias);
            println!("bound '{alias}' to '{name}', re-indexing…");
            if let Err(e) = handle_update(Some(name.clone()), false) {
                eprintln!("warning: re-index failed for '{name}': {e}");
            }
        }
        config.save()?;
    }

    Ok(())
}

fn lindera_platform_triple() -> Result<&'static str> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Ok("aarch64-apple-darwin")
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        Ok("x86_64-apple-darwin")
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Ok("x86_64-unknown-linux-gnu")
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        Ok("aarch64-unknown-linux-gnu")
    } else {
        Err(error::Error::Other(
            "preprocessor install is only supported on macOS (arm64/x86_64) and Linux (x86_64/aarch64)".into()
        ))
    }
}

/// Pinned lindera release. Update intentionally after vetting CLI flag compatibility.
/// Do NOT use /releases/latest — a major version bump could silently break the
/// tokenizer output format or --token-filter argument syntax.
const LINDERA_VERSION: &str = "v3.0.5";
const _: () = assert!(
    matches!(LINDERA_VERSION.as_bytes(), [b'v', b'0'..=b'9', ..]),
    "LINDERA_VERSION must start with 'v' followed by a digit"
);

/// Install the shared lindera CLI binary into preprocessors_dir/lindera-{tag}/. Skips if present.
/// ^ versioned dir ensures a LINDERA_VERSION bump triggers a fresh download
fn install_lindera_binary(preprocessors_dir: &Path, tag: &str, triple: &str) -> Result<PathBuf> {
    let bin_dir = preprocessors_dir.join(format!("lindera-{tag}"));
    let bin_path = bin_dir.join("lindera");
    if bin_path.exists() {
        return Ok(bin_path);
    }
    std::fs::create_dir_all(&bin_dir)?;
    let filename = format!("lindera-{triple}-{tag}.zip");
    let url = format!("https://github.com/lindera/lindera/releases/download/{tag}/{filename}");
    println!("downloading lindera binary…");
    let zip_path = bin_dir.join(&filename);
    download_file(&url, &zip_path)?;
    extract_zip_flat(&zip_path, &bin_dir)?;
    std::fs::remove_file(&zip_path).ok();
    if !bin_path.exists() {
        return Err(error::Error::Other(format!(
            "lindera binary not found after extraction (expected: {})",
            bin_path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&bin_path, std::fs::Permissions::from_mode(0o755))
            .map_err(error::Error::Io)?;
    }
    Ok(bin_path)
}

/// Download and extract a lindera language dictionary into preprocessors_dir/{dict_name}/.
/// Skips if dict_dir already contains files.
fn install_lindera_dict(
    preprocessors_dir: &Path,
    dict_name: &str,
    tag: &str,
    version: &str,
) -> Result<PathBuf> {
    let dict_dir = preprocessors_dir.join(dict_name);
    if dict_dir.is_dir() && std::fs::read_dir(&dict_dir)?.next().is_some() {
        return Ok(dict_dir);
    }
    std::fs::create_dir_all(&dict_dir)?;
    let filename = format!("lindera-{dict_name}-{version}.zip");
    let url = format!("https://github.com/lindera/lindera/releases/download/{tag}/{filename}");
    println!("downloading {dict_name} dictionary…");
    let zip_path = dict_dir.join(&filename);
    download_file(&url, &zip_path)?;
    extract_zip_flat(&zip_path, &dict_dir)?;
    std::fs::remove_file(&zip_path).ok();
    Ok(dict_dir)
}

fn download_file(url: &str, dest: &Path) -> Result<()> {
    let status = std::process::Command::new("curl")
        .args(["-fsSL", url, "-o", &dest.to_string_lossy()])
        .status()
        .map_err(|e| error::Error::Other(format!("curl: {e}")))?;
    if !status.success() {
        return Err(error::Error::Other(format!("download failed: {url}")));
    }
    Ok(())
}

fn extract_zip_flat(zip_path: &Path, dest_dir: &Path) -> Result<()> {
    let status = std::process::Command::new("unzip")
        .args([
            "-o",
            "-j",
            &zip_path.to_string_lossy(),
            "-d",
            &dest_dir.to_string_lossy(),
        ])
        .status()
        .map_err(|e| error::Error::Other(format!("unzip: {e}")))?;
    if !status.success() {
        return Err(error::Error::Other(format!(
            "failed to extract {} (is `unzip` installed?)",
            zip_path.display()
        )));
    }
    Ok(())
}

// ^ docs: README.md#quick-start
fn handle_sync(collection: Option<String>, force: bool) -> Result<()> {
    handle_sync_phases(collection.as_deref(), force, force)
}

fn handle_embed(collection: Option<String>, force: bool) -> Result<()> {
    handle_sync_phases(collection.as_deref(), false, force)
}

fn handle_sync_phases(
    collection: Option<&str>,
    force_update: bool,
    force_embed: bool,
) -> Result<()> {
    let config = Config::load()?;
    let cols = index_collections(&config, collection)?;
    let mut pending = Vec::with_capacity(cols.len());

    for col in &cols {
        let db = update_collection(&config, col, force_update)?;
        let count = index::embed::pending_count(db.conn(), force_embed)?;
        pending.push((db, count));
    }

    if pending.iter().all(|(_, count)| *count == 0) {
        for (db, _) in pending {
            println!("embedding '{}'…", db.name);
            println!("  0 documents, 0 chunks embedded");
            maybe_build_graph(&db)?;
            maybe_sync_ann(&db)?;
        }
        return Ok(());
    }

    llm::download::prepare_model_envs()?;
    println!("loading embedding model…");
    let embedder = llm::embedding::Embedder::load_default()?;

    for (db, count) in pending {
        println!("embedding '{}'…", db.name);
        if count == 0 {
            println!("  0 documents, 0 chunks embedded");
            maybe_build_graph(&db)?;
            maybe_sync_ann(&db)?;
            continue;
        }
        let opts = index::embed::EmbedOptions { force: force_embed };
        let (docs, chunks) = index::embed::embed(&db, &embedder, &opts, llm::models::EMBEDDING)?;
        println!("  {} documents, {} chunks embedded", docs, chunks);
        maybe_build_graph(&db)?;
        maybe_sync_ann(&db)?;
    }
    Ok(())
}

/// Build the kNN document graph after embedding when a graph consumer is on
/// (profile.t0_graph_expand; config > env > default). IR_GRAPH_BUILD forces a
/// build for the research-only graph features (T1/T2) that stay env-gated.
/// Reads stored chunk embeddings only — no model inference; safe on
/// already-embedded collections (embed no-op still triggers a rebuild).
fn maybe_build_graph(db: &db::CollectionDb) -> Result<()> {
    let profile = search::profile::resolve_for_build(db.retrieval.as_ref());
    if !profile.t0_graph_expand && !config::env_flag("IR_GRAPH_BUILD") {
        return Ok(());
    }
    // Nothing to build for an empty collection — stay silent (parity with ANN sync).
    if db.active_doc_count() == 0 {
        return Ok(());
    }
    // Cap k so fetch_m(k) = (k+2)*3+8 can't overflow on a fat-fingered sweep value.
    let k = std::env::var("IR_GRAPH_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.clamp(1, 1024))
        .unwrap_or(10);
    println!("building doc graph for '{}' (k={k})…", db.name);
    let (docs, edges) = db::graph::build(db.conn(), k)?;
    println!("  {} documents, {} edges", docs, edges);
    Ok(())
}

/// Sync the HNSW ANN sidecar after embedding when ANN is on (profile.ann;
/// config > env > default). Incremental — only newly embedded chunks are added;
/// model/dim changes rebuild from stored vectors without model inference.
fn maybe_sync_ann(db: &db::CollectionDb) -> Result<()> {
    let profile = search::profile::resolve_for_build(db.retrieval.as_ref());
    if !profile.ann {
        return Ok(());
    }
    let (total, added) = db::ann::sync(db.conn())?;
    if added > 0 || total > 0 {
        println!("  ann index: {total} vectors ({added} added)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_collection() -> Collection {
        Collection {
            name: "c".into(),
            path: "/tmp".into(),
            globs: vec![],
            excludes: vec![],
            description: None,
            preprocessor: None,
            routing: None,
            retrieval: None,
        }
    }

    #[test]
    fn ko_bind_applies_default_routing() {
        let mut col = test_collection();
        apply_bind_defaults(&mut col, "ko");
        assert_eq!(col.routing, Some(ko_default_routing()));
    }

    #[test]
    fn non_ko_bind_leaves_routing_untouched() {
        let mut col = test_collection();
        apply_bind_defaults(&mut col, "ja");
        assert_eq!(col.routing, None);
    }

    #[test]
    fn ko_unbind_clears_only_auto_default() {
        let mut col = test_collection();
        col.routing = Some(ko_default_routing());
        clear_bind_defaults_if_auto(&mut col, "ko");
        assert_eq!(col.routing, None);
    }
}
