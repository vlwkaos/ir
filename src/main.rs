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

    // Load config once; used by both daemon and direct paths.
    let config = Config::load()?;
    let collection_names = resolve_collections(&config, &collection_filter)?;

    // Route through daemon — start it if not running, fall back to direct on failure.
    {
        let req = daemon::DaemonRequest {
            query: query.clone(),
            collections: collection_names.clone(),
            limit,
            min_score,
            mode: mode.clone(),
            verbose,
        };

        let ready = if daemon::is_running() {
            true
        } else {
            match daemon::start_in_background() {
                Ok(()) => {
                    eprint!("daemon is starting...");
                    let ready = daemon::wait_ready(10_000);
                    eprintln!();
                    ready
                }
                Err(e) => { eprintln!("note: could not start daemon ({e}), running direct"); false }
            }
        };

        if ready {
            match daemon::query(&req) {
                Ok(daemon::QueryResponse { results: daemon_results, log }) => {
                    for line in &log {
                        eprintln!("{line}");
                    }
                    let results: Vec<types::SearchResult> = daemon_results.into_iter()
                        .map(|r| types::SearchResult {
                            collection: r.collection,
                            path: r.path,
                            title: r.title,
                            score: r.score,
                            snippet: if r.snippet.is_empty() { None } else { Some(r.snippet) },
                            hash: r.hash,
                            doc_id: r.doc_id,
                        })
                        .collect();
                    output::print_results(&results, fmt, full);
                    return Ok(());
                }
                Err(e) => eprintln!("daemon error, falling back to direct: {e}"),
            }
        }
    }

    let search_mode: SearchMode = mode.parse().map_err(error::Error::Other)?;

    let cols: Vec<_> = collection_names.iter()
        .filter_map(|name| config.get_collection(name))
        .collect();

    let dbs: Vec<db::CollectionDb> = cols
        .iter()
        .map(|c| db::CollectionDb::open_rw(&c.name, &collection_db_path(&c.name)))
        .collect::<Result<Vec<_>>>()?;

    let results = match search_mode {
        SearchMode::Bm25 => {
            let req = search::fan_out::SearchRequest {
                query: &query,
                limit,
                min_score,
            };
            search::fan_out::bm25(&dbs, &req)?
        }
        SearchMode::Vector => {
            let embedder = llm::embedding::Embedder::load_default()?;
            let req = search::vector::VecSearchRequest {
                query: &query,
                limit,
                min_score,
            };
            search::vector::search(&embedder, &dbs, &req)?
        }
        SearchMode::Hybrid => {
            let embedder = llm::embedding::Embedder::load_default()?;
            // Trio is the default (nDCG@10=0.4032). Qwen3.5 only if IR_QWEN_MODEL is set.
            let qwen = std::env::var_os("IR_QWEN_MODEL")
                .map(|_| llm::qwen::Qwen35::load_default()
                    .map_err(|e| eprintln!("note: qwen3.5 unavailable: {e}"))
                    .ok())
                .flatten()
                .map(std::sync::Arc::new);
            let (expander, scorer) = if let Some(q) = qwen {
                (
                    Some(Box::new(q.clone()) as Box<dyn llm::expander::QueryExpander>),
                    Some(Box::new(q) as Box<dyn llm::scoring::Scorer>),
                )
            } else {
                let exp = llm::expander::Expander::load_default()
                    .map_err(|e| eprintln!("note: expander unavailable: {e}"))
                    .ok()
                    .map(|e| Box::new(e) as Box<dyn llm::expander::QueryExpander>);
                let rer = llm::reranker::Reranker::load_default()
                    .map_err(|e| eprintln!("note: reranker unavailable: {e}"))
                    .ok()
                    .map(|r| Box::new(r) as Box<dyn llm::scoring::Scorer>);
                (exp, rer)
            };
            let hs = search::hybrid::HybridSearch {
                embedder,
                expander,
                scorer,
                expander_cache: db::expander_cache::ExpanderCache::open().ok(),
            };
            let req = search::hybrid::HybridRequest {
                query: &query,
                limit,
                min_score,
                verbose,
            };
            let out = hs.search(&dbs, &req)?;
            for line in &out.log { eprintln!("{line}"); }
            out.results
        }
    };

    output::print_results(&results, fmt, full);
    Ok(())
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
