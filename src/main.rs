mod cli;
mod config;
mod db;
mod error;
mod index;
mod llm;
mod search;
mod types;

use clap::Parser;
use cli::{output, Cli, CollectionCmd, Command};
use config::{collection_db_path, Config};
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
        } => handle_search(
            query.join(" "),
            mode,
            if all { 100_000 } else { limit },
            min_score,
            collections,
            full,
            json,
            csv,
            md,
            files,
        ),
        Command::Get { .. } | Command::MultiGet { .. } | Command::Ls { .. } => {
            eprintln!("not yet implemented");
            Ok(())
        }
    }
}

fn handle_collection(cmd: CollectionCmd) -> Result<()> {
    let mut config = Config::load()?;
    match cmd {
        CollectionCmd::Add { name, path } => {
            let resolved = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone().into());
            config.add_collection(Collection {
                name: name.clone(),
                path: resolved.to_string_lossy().into_owned(),
                globs: vec![],
                excludes: vec![],
                description: None,
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
                    println!("{:<20} {}", c.name, c.path);
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
        println!(
            "  {:<20} {:<12} {}  {}",
            col.name, status, col.path, size
        );
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
) -> Result<()> {
    let search_mode: SearchMode = mode
        .parse()
        .map_err(|e| error::Error::Other(e))?;

    if !matches!(search_mode, SearchMode::Bm25) {
        eprintln!("search --mode {mode}: not yet implemented (Phase 4/5). Using bm25.");
    }

    let config = Config::load()?;
    let cols: Vec<_> = if collection_filter.is_empty() {
        // Auto-detect from CWD, fall back to all collections.
        let cwd = std::env::current_dir().unwrap_or_default();
        if let Some(col) = config::detect_collection(&config.collections, &cwd) {
            vec![col]
        } else {
            config.collections.iter().collect()
        }
    } else {
        collection_filter
            .iter()
            .filter_map(|name| config.get_collection(name))
            .collect()
    };

    // Open DBs for targeted collections.
    let dbs: Vec<db::CollectionDb> = cols
        .iter()
        .map(|c| db::CollectionDb::open(&c.name, &collection_db_path(&c.name)))
        .collect::<Result<Vec<_>>>()?;

    let req = search::fan_out::SearchRequest {
        query: &query,
        limit,
        min_score,
    };

    let results = search::fan_out::bm25(&dbs, &req)?;

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

    output::print_results(&results, fmt, full);
    Ok(())
}

fn handle_embed(collection: Option<String>, _force: bool) -> Result<()> {
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
        println!("embed: '{}' — not yet implemented (Phase 4)", col.name);
    }
    Ok(())
}
