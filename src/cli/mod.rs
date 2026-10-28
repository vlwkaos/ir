// CLI command definitions (clap derive).
// docs: https://docs.rs/clap/latest/clap/_derive/index.html

pub mod output;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

// ^ Help text is the agent-facing guide: what exists, which choice to make,
//   and one runnable example each. Keep it short and true to the code.
const TOP_HELP: &str = "\
Quick start:
  ir collection add notes ~/notes      register a folder (default glob **/*.md)
  ir sync notes                        index text + embed vectors; re-run after edits
  ir search \"how do we rotate keys\"    hybrid search (collection auto-detected from cwd)
  ir get <path> --section \"Rotation\"   read a result, or one section of it

For agents:
  - Add --json: results carry path, title, score, snippet, doc_id.
  - --mode bm25 for exact words and identifiers (instant, no models);
    hybrid (default) for questions; vector when query and documents differ in language.
  - Narrow with -c <collection>, -n <count>, -f \"meta.tags=rust\", -f \"modified_at>=2026-01-01\".
  - Read with `ir get` (--section, --offset, --max-chars) or `ir multi-get`, not by opening files.
  - `ir status` lists collections and their index state; `ir mcp` serves the same tools over MCP.

Run `ir <command> --help` for options and examples.";

const SEARCH_HELP: &str = "\
Examples:
  ir search \"cache invalidation policy\" --json
  ir search max_connections --mode bm25           exact identifier, no model load
  ir search \"배포 롤백 절차\" -c notes -n 5
  ir search release -f \"meta.status=draft\" -f \"modified_at>=2026-01-01\"
  ir search \"token refresh\" --include-code        also code symbols, after the documents

Modes: bm25 = keywords, instant. vector = meaning, works across languages.
hybrid = both, plus a reranker on hard queries; the first call can wait for models to load.
Without -c: the collection containing the current directory, else all collections.
Output: pretty (default), --json, --csv, --md, --files. --full adds document text,
--chunk the best-matching chunk. Results that code supports list it under \"evidence\".";

const GET_HELP: &str = "\
Examples:
  ir get notes/2026/plan.md
  ir get plan.md --section \"Open questions\"       suffix match; section by heading text
  ir get plan.md --offset 4000 --max-chars 4000   page through a long document
  ir multi-get a.md b.md --max-chars 2000 --json";

const ADD_HELP: &str = "\
Examples:
  ir collection add notes ~/notes
  ir collection add docs ~/repo --glob 'docs/**/*.md' --exclude 'docs/archive/**'
  ir collection add kb ~/kb -p ko        Korean BM25 (install first: ir preprocessor install ko)
  ir collection add proj ~/repo --glob '**/*.md' --code-glob 'src/**/*.rs'

Then run `ir sync <name>`. Code files are never document results: their symbols raise the
documents that name them. Link code a document does not name with frontmatter
`code: [src/x.rs, src/x.rs:symbol]` or a code comment `knowledge: <document path>`.";

const SYNC_HELP: &str = "\
sync = update + embed, the usual command after files change.
update: text index only (fast, no models). embed: vectors only.
--force: update rebuilds the text index and drops stored vectors; sync rebuilds both.";

#[derive(Parser, Debug)]
#[command(
    name = "ir",
    about = "Local search over markdown collections (BM25, vectors, hybrid); optional code links",
    version,
    after_help = TOP_HELP
)]
pub struct Cli {
    /// Use another config.yml; collections and caches stay in the data dir
    /// (same as IR_CONFIG_FILE; IR_CONFIG_DIR moves everything)
    #[arg(long = "config-path", global = true, value_name = "FILE")]
    pub config_path: Option<PathBuf>,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Index text and embed vectors for changed files (run after edits)
    #[command(after_help = SYNC_HELP)]
    Sync {
        /// Only synchronize this collection (default: all)
        collection: Option<String>,
        /// Force full text re-index and vector regeneration
        #[arg(long, short)]
        force: bool,
    },
    /// Update the text index only (fast, no models)
    #[command(after_help = SYNC_HELP)]
    Update {
        /// Only update this collection (default: all)
        collection: Option<String>,
        /// Force full re-index from scratch
        #[arg(long, short)]
        force: bool,
    },
    /// Embed vectors for documents that have none (or all, with --force)
    #[command(after_help = SYNC_HELP)]
    Embed {
        /// Only embed this collection (default: all)
        collection: Option<String>,
        /// Re-embed even unchanged chunks
        #[arg(long, short)]
        force: bool,
    },
    /// Search documents: bm25, vector or hybrid (default)
    #[command(after_help = SEARCH_HELP)]
    Search {
        /// Search query
        query: Vec<String>,
        /// bm25 (keywords, instant) | vector (meaning, cross-language) | hybrid
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
        /// Also return matching code symbols and files (collections with a code
        /// domain), after the documents. Filters apply to documents only.
        #[arg(long)]
        include_code: bool,
    },
    /// Print a document (or one section) by path; partial paths match
    #[command(after_help = GET_HELP)]
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
    /// Print several documents by path
    #[command(name = "multi-get", after_help = GET_HELP)]
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
    /// Add, list, rename, move or remove collections (indexed folders)
    Collection {
        #[command(subcommand)]
        cmd: CollectionCmd,
    },
    /// List collections with their index state, path and size
    Status,
    /// Model daemon (search starts it on demand; stop it after upgrading ir)
    Daemon {
        #[command(subcommand)]
        cmd: DaemonCmd,
    },
    /// Korean/Japanese/Chinese tokenizers for BM25 (`ir preprocessor install ko|ja|zh`)
    Preprocessor {
        #[command(subcommand)]
        cmd: PreprocessorCmd,
    },
    /// Serve search, get, multi_get, status and update over MCP (stdio, or --http PORT)
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
    /// Register a folder as a collection (then run `ir sync <name>`)
    #[command(after_help = ADD_HELP)]
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
        /// Source files to index as an auxiliary code domain (e.g. 'src/**/*.rs').
        /// Their symbols and doc comments support matching documents; code files
        /// are never returned as results.
        #[arg(long = "code-glob")]
        code_glob: Vec<String>,
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
