mod cli;
mod config;
mod daemon;
mod db;
mod error;
mod index;
mod llm;
mod search;
mod types;

use clap::Parser;
use cli::{Cli, CollectionCmd, Command, DaemonCmd, output};
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
            json,
            csv,
            md,
            files,
            verbose,
        ),
        Command::Get { .. } => {
            eprintln!("not yet implemented");
            Ok(())
        }
        Command::Daemon { cmd } => match cmd {
            DaemonCmd::Start { timeout } => daemon::start_server(timeout),
            DaemonCmd::Stop => daemon::stop(),
            DaemonCmd::Status => daemon::status(),
        },
    }
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
        } => {
            let resolved = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone().into());
            config.add_collection(Collection {
                name: name.clone(),
                path: resolved.to_string_lossy().into_owned(),
                globs: glob,
                excludes: exclude,
                description,
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
                    if let Some(desc) = &c.description {
                        println!("{:<20} {}  # {}", c.name, c.path, desc);
                    } else {
                        println!("{:<20} {}", c.name, c.path);
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
        let db = db::CollectionDb::open(&col.name, &db_path)?;
        println!("updating '{}'…", col.name);
        let opts = index::UpdateOptions { force };
        let (added, updated, deactivated) = index::update(&db, col, &opts)?;
        println!(
            "  {} added, {} updated, {} deactivated",
            added, updated, deactivated
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_search(
    query: String,
    mode: String,
    limit: usize,
    min_score: Option<f64>,
    collection_filter: Vec<String>,
    full: bool,
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

    let config = Config::load()?;
    let collection_names = resolve_collections(&config, &collection_filter)?;
    let search_mode: SearchMode = mode.parse().map_err(error::Error::Other)?;

    // Open DBs for in-process BM25 (tier-0; also used as fallback).
    let cols: Vec<_> = collection_names.iter()
        .filter_map(|name| config.get_collection(name))
        .collect();
    let dbs: Vec<db::CollectionDb> = cols.iter()
        .map(|c| db::CollectionDb::open_rw(&c.name, &collection_db_path(&c.name)))
        .collect::<Result<Vec<_>>>()?;

    // Tier-0: BM25 in-process, no model needed.
    let bm25_req = search::fan_out::SearchRequest { query: &query, limit, min_score };
    let bm25_results = search::fan_out::bm25(&dbs, &bm25_req)?;

    // Mode dispatch before going to daemon.
    match search_mode {
        // bm25 mode: return BM25 results directly, no daemon needed.
        SearchMode::Bm25 => {
            output::print_results(&bm25_results, fmt, full);
            return Ok(());
        }
        // vector mode: skip BM25 shortcut — go straight to daemon.
        SearchMode::Vector => {}
        // hybrid mode: strong BM25 signal shortcuts LLM work.
        SearchMode::Hybrid => {
            if search::hybrid::is_bm25_strong_signal(&bm25_results) {
                if !daemon::is_running() { let _ = daemon::start_in_background(); }
                output::print_results(&bm25_results, fmt, full);
                return Ok(());
            }
        }
    }

    // Need better results — ensure daemon is running.
    if !daemon::is_running() {
        if let Err(e) = daemon::start_in_background() {
            eprintln!("note: could not start daemon ({e})");
            output::print_results(&bm25_results, fmt, full);
            return Ok(());
        }
    }

    let req = daemon::DaemonRequest {
        query: query.clone(),
        collections: collection_names.clone(),
        limit,
        min_score,
        mode: mode.clone(),
        verbose,
    };

    eprint!("searching...");
    if !daemon::wait_ready(3_000) {
        eprintln!();
        output::print_results(&bm25_results, fmt, full);
        return Ok(());
    }

    // If tier-2 was ready before connecting, this query gets full hybrid.
    let tier2_before = daemon::is_tier2_ready();

    let tier1 = match daemon::query(&req) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("\nnote: daemon query error: {e}");
            output::print_results(&bm25_results, fmt, full);
            return Ok(());
        }
    };

    // Daemon already had full hybrid, or non-hybrid mode — tier-1 result is final.
    if tier2_before || search_mode != SearchMode::Hybrid {
        eprintln!();
        for line in &tier1.log { eprintln!("{line}"); }
        output::print_results(&to_search_results(tier1.results), fmt, full);
        return Ok(());
    }

    // Check if tier-1 result is strong enough to skip tier-2.
    let top = tier1.results.first().map(|r| r.score).unwrap_or(0.0);
    let gap = tier1.results.get(1).map(|r| top - r.score).unwrap_or(top);
    if top >= search::hybrid::STRONG_SIGNAL_FLOOR
        && top * gap >= search::hybrid::STRONG_SIGNAL_PRODUCT
    {
        eprintln!();
        for line in &tier1.log { eprintln!("{line}"); }
        output::print_results(&to_search_results(tier1.results), fmt, full);
        return Ok(());
    }

    // Tier-2: wait for expander+reranker, then re-query for full hybrid.
    eprint!(" enhancing...");
    if !daemon::wait_tier2(7_000) {
        eprintln!();
        for line in &tier1.log { eprintln!("{line}"); }
        output::print_results(&to_search_results(tier1.results), fmt, full);
        return Ok(());
    }

    match daemon::query(&req) {
        Ok(tier2) => {
            eprintln!();
            for line in &tier2.log { eprintln!("{line}"); }
            output::print_results(&to_search_results(tier2.results), fmt, full);
        }
        Err(_) => {
            eprintln!();
            for line in &tier1.log { eprintln!("{line}"); }
            output::print_results(&to_search_results(tier1.results), fmt, full);
        }
    }
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
        })
        .collect()
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

    println!("loading embedding model…");
    let embedder = llm::embedding::Embedder::load_default()?;

    for col in cols {
        let db_path = collection_db_path(&col.name);
        let db = db::CollectionDb::open(&col.name, &db_path)?;
        println!("embedding '{}'…", col.name);
        let opts = index::embed::EmbedOptions { force };
        let (docs, chunks) = index::embed::embed(&db, &embedder, &opts, llm::models::EMBEDDING)?;
        println!("  {} documents, {} chunks embedded", docs, chunks);
    }
    Ok(())
}
