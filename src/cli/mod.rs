// CLI command definitions (clap derive).
// docs: https://docs.rs/clap/latest/clap/_derive/index.html

pub mod output;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "ir", about = "Local markdown search engine")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Index or re-index collections
    Update {
        /// Only update this collection (default: all)
        collection: Option<String>,
        /// Force full re-index from scratch
        #[arg(long, short)]
        force: bool,
    },
    /// Generate vector embeddings
    Embed {
        /// Only embed this collection (default: all)
        collection: Option<String>,
        /// Re-embed even unchanged chunks
        #[arg(long, short)]
        force: bool,
    },
    /// Search indexed documents
    Search {
        /// Search query
        query: Vec<String>,
        /// Search mode: bm25 | vector | hybrid
        #[arg(long, default_value = "hybrid")]
        mode: String,
        /// Max results
        #[arg(short = 'n', default_value = "10")]
        limit: usize,
        /// Minimum relevance score (0-1)
        #[arg(long)]
        min_score: Option<f64>,
        /// Restrict to collection(s)
        #[arg(short = 'c', long = "collection")]
        collections: Vec<String>,
        /// Return all results (up to 100k)
        #[arg(long)]
        all: bool,
        /// Show full document content
        #[arg(long)]
        full: bool,
        /// JSON output
        #[arg(long)]
        json: bool,
        /// CSV output
        #[arg(long)]
        csv: bool,
        /// Markdown output
        #[arg(long)]
        md: bool,
        /// Files-only listing
        #[arg(long)]
        files: bool,
    },
    /// Retrieve a document by path or docid
    Get {
        target: String,
        /// JSON output
        #[arg(long)]
        json: bool,
    },
    /// Manage collections
    Collection {
        #[command(subcommand)]
        cmd: CollectionCmd,
    },
    /// Show index health
    Status,
}

#[derive(Subcommand, Debug)]
pub enum CollectionCmd {
    /// Register a collection
    Add {
        name: String,
        path: String,
        /// Glob patterns to include (default: **/*.md)
        #[arg(long)]
        glob: Vec<String>,
        /// Glob patterns to exclude
        #[arg(long)]
        exclude: Vec<String>,
        /// Short description
        #[arg(long)]
        description: Option<String>,
    },
    /// Remove a collection (keeps DB file by default)
    Rm {
        name: String,
        /// Also delete the database file
        #[arg(long)]
        purge: bool,
    },
    /// Rename a collection
    Rename { old: String, new: String },
    /// List all collections
    Ls,
}
