// CLI command definitions (clap derive).
// docs: https://docs.rs/clap/latest/clap/_derive/index.html

pub mod output;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "ir", about = "Local markdown search engine", version)]
pub struct Cli {
    /// Use an alternate config.yml. Overrides ONLY the config file, not the data
    /// dir (collections/caches stay put) — lets a benchmark vary pipeline config
    /// across candidates while reusing one embedded corpus.
    /// Precedence: this arg > IR_CONFIG_FILE > <config-dir>/config.yml.
    #[arg(long = "config-path", global = true, value_name = "FILE")]
    pub config_path: Option<PathBuf>,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Synchronize text index and vector embeddings
    Sync {
        /// Only synchronize this collection (default: all)
        collection: Option<String>,
        /// Force full text re-index and vector regeneration
        #[arg(long, short)]
        force: bool,
    },
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
        /// Return all results (up to sqlite-vec kNN limit of 4096)
        #[arg(long)]
        all: bool,
        /// Show full document content
        #[arg(long, conflicts_with = "chunk")]
        full: bool,
        /// Show best-matching chunk content (vector results only; falls back to snippet for BM25)
        #[arg(long, conflicts_with = "full")]
        chunk: bool,
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
        /// Show pipeline decisions and timing
        #[arg(long, short = 'v')]
        verbose: bool,
        /// Suppress all stderr output (progress, logs)
        #[arg(long, short = 'q', conflicts_with = "verbose")]
        quiet: bool,
        /// Filter clause: FIELD OP VALUE (repeatable; clauses are ANDed).
        /// Fields: path, modified_at, created_at, meta.<name>
        /// Ops: =, !=, >, >=, <, <=, ~ (contains), !~ (not-contains)
        /// Dates normalized to UTC RFC3339. Example: -f "modified_at>=2024-01-01"
        #[arg(short = 'f', long = "filter", action = clap::ArgAction::Append, value_name = "CLAUSE")]
        filter: Vec<String>,
    },
    /// Retrieve a document by path
    Get {
        /// Path to retrieve. Accepts collection-relative ("2026/file.md"), vault-root
        /// ("Notes/2026/file.md"), or partial paths (suffix/substring match).
        target: String,
        /// Restrict to collection(s)
        #[arg(short = 'c', long = "collection")]
        collections: Vec<String>,
        /// Return only the named section (heading text, case-insensitive)
        #[arg(long)]
        section: Option<String>,
        /// Start output at this character offset into the document
        #[arg(long)]
        offset: Option<usize>,
        /// Truncate output to this many characters
        #[arg(long)]
        max_chars: Option<usize>,
        /// JSON output
        #[arg(long)]
        json: bool,
    },
    /// Retrieve multiple documents by path
    #[command(name = "multi-get")]
    MultiGet {
        /// Paths to retrieve (same matching rules as get)
        #[arg(required = false)]
        targets: Vec<String>,
        /// Restrict to collection(s)
        #[arg(short = 'c', long = "collection")]
        collections: Vec<String>,
        /// Truncate each document to this many characters
        #[arg(long)]
        max_chars: Option<usize>,
        /// JSON output
        #[arg(long)]
        json: bool,
        /// Print matched paths only (one per line)
        #[arg(long)]
        files: bool,
    },
    /// Manage collections
    Collection {
        #[command(subcommand)]
        cmd: CollectionCmd,
    },
    /// Show index health
    Status,
    /// Manage the search daemon (keeps models warm between queries)
    Daemon {
        #[command(subcommand)]
        cmd: DaemonCmd,
    },
    /// Manage text preprocessors for CJK / morphological analysis
    Preprocessor {
        #[command(subcommand)]
        cmd: PreprocessorCmd,
    },
    /// Start an MCP server for Claude Desktop / Claude Code integration
    Mcp {
        /// Serve over HTTP on the given port instead of stdio (e.g. --http 3620)
        #[arg(long)]
        http: Option<u16>,
        /// Set Access-Control-Allow-Origin for HTTP mode ('*' or exact origin).
        /// For non-loopback deployments the server's Host header must also be trusted by rmcp;
        /// use '*' to disable rmcp's host check alongside wildcard CORS.
        #[arg(long, requires = "http")]
        cors: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
pub enum DaemonCmd {
    /// Start the daemon in the foreground
    Start {
        /// Seconds of inactivity before auto-shutdown (0 = never)
        #[arg(long, default_value = "3600")]
        timeout: u64,
    },
    /// Stop the running daemon
    Stop,
    /// Show daemon status
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
        /// Preprocessor alias(es) for BM25/FTS5 tokenization (registered via `ir preprocessor add`)
        #[arg(long = "preprocessor", short = 'p')]
        preprocessor: Vec<String>,
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
    /// Update the source path of a collection
    SetPath { name: String, path: String },
}

#[derive(Subcommand, Debug)]
pub enum PreprocessorCmd {
    /// Register a preprocessor alias
    Add {
        /// Alias name (e.g. "ko")
        alias: String,
        /// Command to run (e.g. "kiwi-tokenize" or "mecab -Owakati")
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        command: Vec<String>,
    },
    /// Download and register a bundled preprocessor wrapper
    Install {
        /// Language code to install (ko, ja, zh)
        lang: String,
    },
    /// List registered preprocessors and available bundled ones
    List,
    /// Unregister a preprocessor alias (soft by default)
    Remove {
        alias: String,
        /// Also delete the binary if installed under the ir preprocessors dir
        #[arg(long, short = 'd')]
        delete: bool,
    },
    /// Wire a registered preprocessor alias to a collection and re-index
    Bind {
        /// Registered preprocessor alias (e.g. "ko", "ja"). Run `ir preprocessor list` to see available aliases.
        alias: String,
        collection: Option<String>,
    },
    /// Remove a preprocessor from a collection and re-index
    Unbind {
        /// Registered preprocessor alias (e.g. "ko", "ja")
        alias: String,
        collection: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Parser, error::ErrorKind};

    #[test]
    fn given_sync_without_arguments_when_parsed_then_collection_is_optional() {
        let cli = Cli::try_parse_from(["ir", "sync"]).unwrap();

        let Command::Sync { collection, force } = cli.command else {
            panic!("expected sync command");
        };
        assert_eq!((collection, force), (None, false));
    }

    #[test]
    fn given_sync_with_collection_when_parsed_then_collection_is_selected() {
        let cli = Cli::try_parse_from(["ir", "sync", "notes"]).unwrap();

        let Command::Sync { collection, force } = cli.command else {
            panic!("expected sync command");
        };
        assert_eq!((collection.as_deref(), force), (Some("notes"), false));
    }

    #[test]
    fn given_sync_with_short_force_when_parsed_then_force_is_enabled() {
        let cli = Cli::try_parse_from(["ir", "sync", "-f"]).unwrap();

        let Command::Sync { collection, force } = cli.command else {
            panic!("expected sync command");
        };
        assert_eq!((collection, force), (None, true));
    }

    #[test]
    fn given_sync_with_long_force_after_collection_when_parsed_then_force_is_enabled() {
        let cli = Cli::try_parse_from(["ir", "sync", "notes", "--force"]).unwrap();

        let Command::Sync { collection, force } = cli.command else {
            panic!("expected sync command");
        };
        assert_eq!((collection.as_deref(), force), (Some("notes"), true));
    }

    #[test]
    fn given_sync_with_long_force_before_collection_when_parsed_then_force_is_enabled() {
        let cli = Cli::try_parse_from(["ir", "sync", "--force", "notes"]).unwrap();

        let Command::Sync { collection, force } = cli.command else {
            panic!("expected sync command");
        };
        assert_eq!((collection.as_deref(), force), (Some("notes"), true));
    }

    #[test]
    fn given_sync_with_two_collections_when_parsed_then_extra_argument_is_unknown() {
        let error = Cli::try_parse_from(["ir", "sync", "first", "second"]).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::UnknownArgument);
    }
}
