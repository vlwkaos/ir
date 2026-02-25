mod cli;
mod config;
mod db;
mod error;
mod index;
mod llm;
mod search;
mod types;

use clap::Parser;
use cli::{Cli, CollectionCmd, Command};
use config::{collection_db_path, Config};
use error::Result;
use types::Collection;

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
        Command::Search { .. } => {
            eprintln!("search: not yet implemented (Phase 3)");
            Ok(())
        }
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

fn handle_update(collection: Option<String>, _force: bool) -> Result<()> {
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
        let _db = db::CollectionDb::open(&col.name, &db_path)?;
        println!("update: '{}' — indexing not yet implemented (Phase 2)", col.name);
    }
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
