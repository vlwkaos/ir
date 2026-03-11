// ir daemon — keeps models warm in a background process; accepts search queries
// over a Unix socket, eliminating per-invocation model load time (~3-7s → ~0ms).
//
// Default model stack: EmbeddingGemma-300M + qmd-expander-1.7B + Qwen3-Reranker-0.6B
// (the trio, nDCG@10=0.4032 on NFCorpus — best measured configuration).
// Metal is used by default on macOS (IR_GPU_LAYERS=99).
//
// Protocol: newline-delimited JSON over Unix socket.
//   request:  {"query":"...","collections":["name"],"limit":10,"min_score":0.2,"mode":"hybrid"}
//   response: {"ok":true,"results":[{"path":"...","score":0.9,"title":"...","snippet":"..."}]}
//   response: {"ok":false,"error":"..."}

use crate::config::{self, collection_db_path};
use crate::db::CollectionDb;
use crate::error::{Error, Result};
use crate::search;
use crate::types::SearchMode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ── Protocol types ────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct DaemonRequest {
    pub query: String,
    pub collections: Vec<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    pub min_score: Option<f64>,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default)]
    pub verbose: bool,
}

fn default_limit() -> usize { 10 }
fn default_mode() -> String { "hybrid".into() }

#[derive(Serialize)]
struct DaemonResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    results: Option<Vec<DaemonResult>>,
    #[serde(default)]
    log: Vec<String>,
}

pub struct QueryResponse {
    pub results: Vec<DaemonResult>,
    pub log: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct DaemonResult {
    pub collection: String,
    pub path: String,
    pub title: String,
    pub score: f64,
    pub snippet: String,
    pub hash: String,
    pub doc_id: String,
}

// ── Client ────────────────────────────────────────────────────────────────────

/// Returns true if the daemon socket is reachable.
pub fn is_running() -> bool {
    UnixStream::connect(config::daemon_socket_path()).is_ok()
}

/// Spawn `ir daemon start` as a detached background process.
/// Stderr is redirected to `~/.config/ir/daemon.log`.
pub fn start_in_background() -> Result<()> {
    let exe = std::env::current_exe().map_err(Error::Io)?;
    let log_path = config::daemon_socket_path()
        .parent()
        .unwrap_or(std::path::Path::new("/tmp"))
        .join("daemon.log");
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(Error::Io)?;
    std::process::Command::new(exe)
        .args(["daemon", "start"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(log_file)
        .spawn()
        .map_err(|e| Error::Other(format!("spawn daemon: {e}")))?;
    Ok(())
}

/// Poll the daemon socket until it accepts connections or `timeout_ms` elapses.
/// Returns true if ready.
pub fn wait_ready(timeout_ms: u64) -> bool {
    let deadline = std::time::Instant::now() + Duration::from_millis(timeout_ms);
    while std::time::Instant::now() < deadline {
        if is_running() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

/// Returns true if the daemon's tier-2 models (expander + reranker) are loaded.
pub fn is_tier2_ready() -> bool {
    config::daemon_tier2_path().exists()
}

/// Poll for tier-2 readiness or `timeout_ms` elapses. Returns true if ready.
pub fn wait_tier2(timeout_ms: u64) -> bool {
    let deadline = std::time::Instant::now() + Duration::from_millis(timeout_ms);
    while std::time::Instant::now() < deadline {
        if is_tier2_ready() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

/// Tier-2 models sent from background loader to main accept loop via sync_channel.
struct Tier2 {
    expander: Option<Box<dyn crate::llm::expander::QueryExpander>>,
    scorer: Option<Box<dyn crate::llm::scoring::Scorer>>,
}

/// Send a search request to the daemon and return parsed results + pipeline log.
pub fn query(req: &DaemonRequest) -> Result<QueryResponse> {
    let sock = config::daemon_socket_path();
    let stream = UnixStream::connect(&sock)
        .map_err(|e| Error::Other(format!("daemon connect: {e}")))?;

    let mut writer = stream.try_clone().map_err(Error::Io)?;
    let reader = BufReader::new(stream);

    let payload = serde_json::to_string(req)
        .map_err(|e| Error::Other(format!("serialize: {e}")))?;
    writer.write_all(payload.as_bytes()).map_err(Error::Io)?;
    writer.write_all(b"\n").map_err(Error::Io)?;

    let line = reader.lines()
        .next()
        .ok_or_else(|| Error::Other("daemon closed connection".into()))?
        .map_err(Error::Io)?;

    let resp: serde_json::Value = serde_json::from_str(&line)
        .map_err(|e| Error::Other(format!("parse response: {e}")))?;

    if resp["ok"].as_bool().unwrap_or(false) {
        let results: Vec<DaemonResult> = serde_json::from_value(
            resp["results"].clone()
        ).map_err(|e| Error::Other(format!("parse results: {e}")))?;
        let log: Vec<String> = serde_json::from_value(
            resp["log"].clone()
        ).unwrap_or_default();
        Ok(QueryResponse { results, log })
    } else {
        Err(Error::Other(
            resp["error"].as_str().unwrap_or("daemon error").to_string()
        ))
    }
}

// ── Server ────────────────────────────────────────────────────────────────────

/// Config snapshot loaded once at daemon startup; avoids YAML disk reads per query.
struct DaemonState {
    collection_paths: HashMap<String, PathBuf>,
}

impl DaemonState {
    fn load() -> Result<Self> {
        let cfg = config::Config::load()?;
        let collection_paths = cfg.collections.iter()
            .map(|c| (c.name.clone(), collection_db_path(&c.name)))
            .collect();
        Ok(Self { collection_paths })
    }
}

pub fn start_server(timeout_secs: u64) -> Result<()> {
    let sock_path = config::daemon_socket_path();
    let pid_path = config::daemon_pid_path();
    let tier2_path = config::daemon_tier2_path();

    if let Some(parent) = sock_path.parent() {
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
        // Restrict to owner-only so other local users cannot reach the socket.
        std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))
            .map_err(Error::Io)?;
    }

    // Remove stale files; next start will always replace them.
    if sock_path.exists() { std::fs::remove_file(&sock_path).map_err(Error::Io)?; }
    let _ = std::fs::remove_file(&tier2_path);

    let gpu_on = crate::llm::gpu_layers() > 0;
    eprintln!("loading models (Metal: {})...", if gpu_on { "on" } else { "off" });

    // Tier-1: load embedder only.
    let embedder = crate::llm::embedding::Embedder::load_default()
        .map_err(|e| Error::Other(format!("load embedder: {e}")))?;
    eprintln!("  embedder ready");

    let mut hybrid = search::hybrid::HybridSearch {
        embedder,
        expander: None,
        scorer: None,
        expander_cache: None,
    };

    let state = DaemonState::load()?;

    // Bind socket — tier-1 signal. Clients can connect for score-fusion queries now.
    let listener = UnixListener::bind(&sock_path)
        .map_err(|e| Error::Other(format!("bind {}: {e}", sock_path.display())))?;
    std::fs::set_permissions(&sock_path, std::fs::Permissions::from_mode(0o600))
        .map_err(Error::Io)?;
    // Write PID immediately so stop() works from tier-1 onward.
    std::fs::write(&pid_path, std::process::id().to_string()).map_err(Error::Io)?;

    let timeout_msg = if timeout_secs > 0 { format!("  timeout={}s", timeout_secs) } else { "  timeout=never".into() };
    eprintln!("daemon ready (tier 1)  socket={}{}", sock_path.display(), timeout_msg);

    // Tier-2: load expander+reranker in background; signal readiness via tier2_path.
    // Race-free: send to channel BEFORE writing tier2_path. Main thread's try_recv()
    // fires on the next connection after tier2_path appears, so models are always present.
    let (tx, rx) = std::sync::mpsc::sync_channel::<Tier2>(1);
    let tier2_path_bg = tier2_path.clone();
    std::thread::spawn(move || {
        let qwen = std::env::var_os("IR_QWEN_MODEL")
            .map(|_| crate::llm::qwen::Qwen35::load_default()
                .map_err(|e| { eprintln!("  note: qwen unavailable ({e})"); e })
                .ok())
            .flatten()
            .map(std::sync::Arc::new);

        let (expander, scorer) = if let Some(q) = qwen {
            eprintln!("  qwen ready");
            (
                Some(Box::new(q.clone()) as Box<dyn crate::llm::expander::QueryExpander>),
                Some(Box::new(q) as Box<dyn crate::llm::scoring::Scorer>),
            )
        } else {
            let exp = crate::llm::expander::Expander::load_default()
                .map_err(|e| eprintln!("  note: expander unavailable ({e})"))
                .ok()
                .map(|e| { eprintln!("  expander ready"); Box::new(e) as Box<dyn crate::llm::expander::QueryExpander> });
            let rer = crate::llm::reranker::Reranker::load_default()
                .map_err(|e| eprintln!("  note: reranker unavailable ({e})"))
                .ok()
                .map(|r| { eprintln!("  reranker ready"); Box::new(r) as Box<dyn crate::llm::scoring::Scorer> });
            (exp, rer)
        };

        // Send models to main thread before writing signal file.
        let _ = tx.send(Tier2 { expander, scorer });
        let _ = std::fs::write(&tier2_path_bg, "");
        eprintln!("  tier-2 ready");
    });

    // Inactivity watchdog: exit after `timeout_secs` of no queries.
    let last_activity = Arc::new(AtomicU64::new(unix_now()));
    if timeout_secs > 0 {
        let last = Arc::clone(&last_activity);
        let sock = sock_path.clone();
        let pid = pid_path.clone();
        let t2 = tier2_path.clone();
        std::thread::spawn(move || {
            let check_every = (timeout_secs / 10).clamp(1, 30);
            loop {
                std::thread::sleep(Duration::from_secs(check_every));
                let idle = unix_now().saturating_sub(last.load(Ordering::Relaxed));
                if idle >= timeout_secs {
                    eprintln!("daemon: idle for {idle}s, shutting down");
                    let _ = std::fs::remove_file(&sock);
                    let _ = std::fs::remove_file(&pid);
                    let _ = std::fs::remove_file(&t2);
                    std::process::exit(0);
                }
            }
        });
    }

    for stream in listener.incoming() {
        // Pick up tier-2 models if background thread finished loading.
        if let Ok(t2) = rx.try_recv() {
            hybrid.expander = t2.expander;
            hybrid.scorer = t2.scorer;
            hybrid.expander_cache = crate::db::expander_cache::ExpanderCache::open()
                .map_err(|e| eprintln!("  note: expander cache unavailable ({e})"))
                .ok();
        }
        match stream {
            Ok(s) => {
                last_activity.store(unix_now(), Ordering::Relaxed);
                if let Err(e) = handle_connection(s, &hybrid, &state) {
                    eprintln!("connection error: {e}");
                }
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }

    let _ = std::fs::remove_file(&sock_path);
    let _ = std::fs::remove_file(&pid_path);
    let _ = std::fs::remove_file(&tier2_path);
    Ok(())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn handle_connection(
    stream: UnixStream,
    hybrid: &search::hybrid::HybridSearch,
    state: &DaemonState,
) -> Result<()> {
    const MAX_REQUEST_BYTES: u64 = 64 * 1024;
    let mut writer = stream.try_clone().map_err(Error::Io)?;
    let mut reader = BufReader::new(stream.take(MAX_REQUEST_BYTES));

    // One request per connection — read one line, write one line, done.
    let mut line = String::new();
    if reader.read_line(&mut line).map_err(Error::Io)? == 0 || line.trim().is_empty() {
        return Ok(());
    }
    if line.len() as u64 >= MAX_REQUEST_BYTES {
        return Err(Error::Other("request exceeds 64KiB limit".into()));
    }

    let resp = match handle_request(line.trim_end(), hybrid, state) {
        Ok((results, log)) => DaemonResponse { ok: true, error: None, results: Some(results), log },
        Err(e) => DaemonResponse { ok: false, error: Some(e.to_string()), results: None, log: vec![] },
    };

    let json = serde_json::to_string(&resp)
        .map_err(|e| Error::Other(format!("serialize: {e}")))?;
    writer.write_all(json.as_bytes()).map_err(Error::Io)?;
    writer.write_all(b"\n").map_err(Error::Io)?;
    writer.flush().map_err(Error::Io)?;
    Ok(())
}

fn handle_request(
    line: &str,
    hybrid: &search::hybrid::HybridSearch,
    state: &DaemonState,
) -> Result<(Vec<DaemonResult>, Vec<String>)> {
    let req: DaemonRequest = serde_json::from_str(line)
        .map_err(|e| Error::Other(format!("parse request: {e}")))?;

    let mode: SearchMode = req.mode.parse().map_err(Error::Other)?;

    let selected: Vec<(&String, &PathBuf)> = req.collections.iter()
        .filter_map(|name| state.collection_paths.get(name).map(|p| (name, p)))
        .collect();

    if selected.is_empty() {
        return Err(Error::Other(format!(
            "no matching collections for {:?}", req.collections
        )));
    }

    // Fresh RW connections per query — sees live index updates, enables cache writes.
    let dbs: Vec<CollectionDb> = selected.iter()
        .map(|(name, path)| CollectionDb::open_rw(name.as_str(), path))
        .collect::<Result<Vec<_>>>()?;

    let (results, log) = match mode {
        SearchMode::Hybrid => {
            let r = search::hybrid::HybridRequest {
                query: &req.query,
                limit: req.limit,
                min_score: req.min_score,
                verbose: req.verbose,
            };
            let out = hybrid.search(&dbs, &r)?;
            (out.results, out.log)
        }
        SearchMode::Vector => {
            let r = search::vector::VecSearchRequest {
                query: &req.query,
                limit: req.limit,
                min_score: req.min_score,
            };
            (search::vector::search(&hybrid.embedder, &dbs, &r)?, vec![])
        }
        SearchMode::Bm25 => {
            let r = search::fan_out::SearchRequest {
                query: &req.query,
                limit: req.limit,
                min_score: req.min_score,
            };
            (search::fan_out::bm25(&dbs, &r)?, vec![])
        }
    };

    Ok((results.into_iter().map(|r| DaemonResult {
        collection: r.collection,
        path: r.path,
        title: r.title,
        score: r.score,
        snippet: r.snippet.unwrap_or_default(),
        hash: r.hash,
        doc_id: r.doc_id,
    }).collect(), log))
}

// ── Stop / status ─────────────────────────────────────────────────────────────

pub fn stop() -> Result<()> {
    let pid_path = config::daemon_pid_path();
    let sock_path = config::daemon_socket_path();

    if !pid_path.exists() {
        // Tier-1 may be up (socket bound) but tier-2 still loading (no PID yet).
        if sock_path.exists() {
            let _ = std::fs::remove_file(&sock_path);
            let _ = std::fs::remove_file(&config::daemon_tier2_path());
            eprintln!("daemon stopping (tier-2 still loading, socket removed)");
        } else {
            eprintln!("daemon not running (no pid file)");
        }
        return Ok(());
    }

    let pid_str = std::fs::read_to_string(&pid_path).map_err(Error::Io)?;
    let pid: i32 = pid_str.trim().parse()
        .map_err(|_| Error::Other(format!("invalid pid file: {pid_str:?}")))?;

    unsafe extern "C" { fn kill(pid: i32, sig: i32) -> i32; }
    // Probe first (sig=0): verifies the process exists without sending a signal.
    if unsafe { kill(pid, 0) } != 0 {
        eprintln!("warning: pid {pid} not found — removing stale pid file");
        let _ = std::fs::remove_file(&pid_path);
        return Ok(());
    }
    let rc = unsafe { kill(pid, 15) }; // SIGTERM
    if rc != 0 {
        eprintln!("warning: kill({pid}) failed — process may already be gone");
    }

    let _ = std::fs::remove_file(&pid_path);
    let _ = std::fs::remove_file(&sock_path);
    let _ = std::fs::remove_file(&config::daemon_tier2_path());
    eprintln!("daemon stopped (pid {pid})");
    Ok(())
}

pub fn status() -> Result<()> {
    if is_running() {
        let pid = std::fs::read_to_string(config::daemon_pid_path())
            .unwrap_or_else(|_| "?".into());
        println!("running  pid={}", pid.trim());
    } else {
        println!("not running");
    }
    Ok(())
}
