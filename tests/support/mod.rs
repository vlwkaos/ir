// Shared apparatus for end-to-end tests. Every scenario drives the compiled `ir`
// binary (CLI or `ir mcp` over stdio) against an isolated config dir and real
// files on disk; nothing is mocked and no crate internals are called.
#![allow(dead_code)]

use serde_json::{Value, json};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Output, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};
use tempfile::{Builder, TempDir};

const INHERITED_MODEL_ENV_VARS: &[&str] = &[
    "IR_COMBINED_MODEL",
    "IR_QWEN_MODEL",
    "IR_EXPANDER_MODEL",
    "IR_RERANKER_MODEL",
    "QMD_EMBEDDING_MODEL",
    "QMD_EMBED_MODEL",
    "QMD_EXPANDER_MODEL",
    "QMD_EXPAND_MODEL",
    "QMD_RERANKER_MODEL",
    "QMD_RERANK_MODEL",
    "QMD_MODEL_DIRS",
];

/// One isolated ir installation: its own config dir, no reachable models.
pub struct Env {
    root: TempDir,
    pub config_dir: PathBuf,
    empty_model_dir: PathBuf,
    invalid_model: PathBuf,
    // ^ A search starts the daemon in the background unless another process
    //   holds the startup lock; while this is held no scenario leaves a
    //   detached `ir daemon start` behind. None once a scenario allows it.
    daemon_startup_lock: Option<fs::File>,
    // The caller's own models are reachable (scenarios that need a real daemon).
    real_models: bool,
}

/// (exit success, stdout, stderr)
pub type Snapshot = (bool, String, String);

pub fn snapshot(output: Output) -> Snapshot {
    (
        output.status.success(),
        String::from_utf8(output.stdout).expect("stdout should be UTF-8"),
        String::from_utf8(output.stderr).expect("stderr should be UTF-8"),
    )
}

impl Env {
    pub fn new(tag: &str) -> Self {
        let tmp_root = Path::new(env!("CARGO_MANIFEST_DIR")).join(".tmp");
        fs::create_dir_all(&tmp_root).expect("create project-local temporary directory");
        let root = Builder::new()
            .prefix(&format!("{tag}-"))
            .tempdir_in(tmp_root)
            .expect("create unique e2e fixture");
        let config_dir = root.path().join("config");
        let empty_model_dir = root.path().join("empty-models");
        fs::create_dir_all(&empty_model_dir).expect("create empty model directory");
        fs::create_dir_all(&config_dir).expect("create config directory");
        let daemon_startup_lock = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(config_dir.join("daemon.lock"))
            .expect("create daemon startup lock");
        assert_eq!(
            unsafe { libc::flock(daemon_startup_lock.as_raw_fd(), libc::LOCK_EX) },
            0,
            "hold daemon startup lock"
        );
        Self {
            daemon_startup_lock: Some(daemon_startup_lock),
            real_models: false,
            invalid_model: root.path().join("must-not-be-loaded.gguf"),
            config_dir,
            empty_model_dir,
            root,
        }
    }

    /// An installation that can run a real daemon: searches may start it, and
    /// it loads the models of whoever runs the test. Only for `#[ignore]`d
    /// scenarios; the daemon is stopped when the fixture is dropped.
    pub fn with_real_models(tag: &str) -> Self {
        let mut env = Self::new(tag);
        env.real_models = true;
        env.allow_daemon_start();
        env
    }

    /// Let searches start the daemon in the background, as they do for a user.
    pub fn allow_daemon_start(&mut self) {
        self.daemon_startup_lock = None;
    }

    /// The compiled binary, pointed at this installation. Model lookups are
    /// forced to fail so no scenario can load or download a model.
    pub fn ir(&self) -> Command {
        self.ir_at(Path::new(env!("CARGO_BIN_EXE_ir")))
    }

    /// Like `ir`, for a copy of the binary at another path.
    pub fn ir_at(&self, exe: &Path) -> Command {
        let mut command = Command::new(exe);
        command
            .current_dir(self.root.path())
            .env_remove("XDG_CONFIG_HOME")
            .env_remove("IR_CONFIG_FILE")
            .env_remove("IR_GRAPH_T0_EXPAND")
            .env_remove("IR_GRAPH_BUILD")
            .env_remove("IR_ANN")
            .env("IR_CONFIG_DIR", &self.config_dir);
        if !self.real_models {
            for key in INHERITED_MODEL_ENV_VARS {
                command.env_remove(key);
            }
            command
                .env("IR_MODEL_DIRS", &self.empty_model_dir)
                .env("IR_EMBEDDING_MODEL", &self.invalid_model);
        }
        command
    }

    pub fn run(&self, args: &[&str]) -> Snapshot {
        let mut command = self.ir();
        command.args(args);
        snapshot(command.output().expect("run compiled ir binary"))
    }

    /// Run and require success; returns stdout.
    pub fn ok(&self, args: &[&str]) -> String {
        let (success, stdout, stderr) = self.run(args);
        assert!(
            success,
            "`ir {}` failed:\n{stdout}\n{stderr}",
            args.join(" ")
        );
        stdout
    }

    /// Path under the fixture root (not created).
    pub fn path(&self, rel: &str) -> PathBuf {
        self.root.path().join(rel)
    }

    /// Create a directory and register it as a collection.
    pub fn add_collection(&self, name: &str, extra_args: &[&str]) -> PathBuf {
        let dir = self.path(name);
        fs::create_dir_all(&dir).expect("create collection directory");
        let mut args = vec!["collection", "add", name, dir.to_str().unwrap()];
        args.extend_from_slice(extra_args);
        self.ok(&args);
        dir
    }

    pub fn db_path(&self, collection: &str) -> PathBuf {
        self.config_dir
            .join("collections")
            .join(format!("{collection}.sqlite"))
    }

    /// Paths returned by a BM25 search through the CLI, best first.
    pub fn bm25_paths(&self, collections: &[&str], query: &str) -> Vec<String> {
        let mut args = vec!["search", "--mode", "bm25", "--json", "--quiet"];
        for name in collections {
            args.extend_from_slice(&["-c", name]);
        }
        args.push(query);
        let stdout = self.ok(&args);
        let results: Value = serde_json::from_str(&stdout)
            .unwrap_or_else(|e| panic!("search output is not JSON ({e}): {stdout}"));
        result_paths(&results)
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        if self.daemon_startup_lock.is_none() {
            let _ = self.ir().args(["daemon", "stop"]).output();
        }
    }
}

pub fn result_paths(results: &Value) -> Vec<String> {
    results
        .as_array()
        .expect("search results should be a JSON array")
        .iter()
        .map(|r| r["path"].as_str().expect("result path").to_string())
        .collect()
}

pub fn write(dir: &Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent directory");
    }
    fs::write(path, content).expect("write document");
}

/// The counts line `ir update` prints: (scanned, added, updated, deactivated).
pub fn update_counts(stdout: &str) -> (usize, usize, usize, usize) {
    let line = stdout
        .lines()
        .find(|l| l.contains(" scanned: "))
        .unwrap_or_else(|| panic!("no counts line in update output: {stdout}"));
    let numbers: Vec<usize> = line
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    match numbers[..] {
        [scanned, added, updated, deactivated] => (scanned, added, updated, deactivated),
        _ => panic!("unexpected counts line: {line}"),
    }
}

/// Poll `condition` until it holds or `limit` passes; returns whether it held.
pub fn eventually(limit: Duration, mut condition: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + limit;
    loop {
        if condition() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// A line filter that obeys the preprocessor protocol, written to `name` under
/// the fixture root. `body` is Python run per input line `s`; it prints 0 or 1 line.
pub fn line_filter(env: &Env, name: &str, body: &str) -> String {
    let script = env.path(name);
    fs::write(
        &script,
        format!("import sys, time\nfor line in sys.stdin:\n    s = line.rstrip('\\n')\n{body}\n"),
    )
    .expect("write preprocessor script");
    format!("python3 {}", script.display())
}

/// Register `command` as preprocessor alias `alias`.
pub fn add_preprocessor(env: &Env, alias: &str, command: &str) {
    let mut args = vec!["preprocessor", "add", alias];
    args.extend(command.split(' '));
    env.ok(&args);
}

/// Wait for a child with a deadline; kills it and returns None on timeout.
pub fn wait_within(mut child: Child, limit: Duration) -> Option<Snapshot> {
    let deadline = Instant::now() + limit;
    loop {
        match child.try_wait().expect("poll child") {
            Some(_) => return Some(snapshot(child.wait_with_output().expect("collect output"))),
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    }
}

/// A long-lived `ir mcp` stdio server driven with real JSON-RPC messages.
pub struct Mcp {
    child: Child,
    stdin: ChildStdin,
    responses: Receiver<Value>,
    // Replies read while waiting for a different id (overlapping requests).
    early: std::collections::HashMap<u64, Value>,
    next_id: u64,
}

const MCP_REPLY_TIMEOUT: Duration = Duration::from_secs(60);

impl Mcp {
    pub fn start(env: &Env) -> Self {
        let mut child = env
            .ir()
            .arg("mcp")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("start ir mcp");
        let stdin = child.stdin.take().expect("mcp stdin");
        let stdout = child.stdout.take().expect("mcp stdout");
        let (tx, responses) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                // ^ Anything on stdout that is not JSON-RPC breaks the transport;
                //   forward it so the scenario fails loudly instead of hanging.
                let message = serde_json::from_str(&line)
                    .unwrap_or_else(|_| json!({ "non_json_stdout": line }));
                if tx.send(message).is_err() {
                    break;
                }
            }
        });
        let mut server = Self {
            child,
            stdin,
            responses,
            early: Default::default(),
            next_id: 1,
        };
        server.request(
            "initialize",
            json!({
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": { "name": "ir-e2e", "version": "0" },
            }),
        );
        server.send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }));
        server
    }

    fn send(&mut self, message: &Value) {
        writeln!(self.stdin, "{message}").expect("write to mcp stdin");
        self.stdin.flush().expect("flush mcp stdin");
    }

    /// Send a request without waiting; returns its id.
    pub fn send_request(&mut self, method: &str, params: Value) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }));
        id
    }

    /// Wait for the reply to `id`; returns its `result`.
    pub fn reply(&mut self, id: u64) -> Value {
        loop {
            if let Some(result) = self.early.remove(&id) {
                return result;
            }
            let message = self
                .responses
                .recv_timeout(MCP_REPLY_TIMEOUT)
                .unwrap_or_else(|e| panic!("no MCP reply to request {id}: {e}"));
            assert!(
                message.get("non_json_stdout").is_none(),
                "MCP server wrote non-protocol text to stdout: {message}"
            );
            if let Some(reply_id) = message["id"].as_u64() {
                assert!(message.get("error").is_none(), "MCP error: {message}");
                self.early.insert(reply_id, message["result"].clone());
            }
        }
    }

    pub fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.send_request(method, params);
        self.reply(id)
    }

    pub fn send_tool(&mut self, name: &str, arguments: Value) -> u64 {
        self.send_request(
            "tools/call",
            json!({ "name": name, "arguments": arguments }),
        )
    }

    /// Tool outcome: Ok(parsed JSON payload) or Err(tool error text).
    pub fn tool_reply(&mut self, id: u64) -> Result<Value, String> {
        let result = self.reply(id);
        let text = result["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("tool result without text content: {result}"))
            .to_string();
        if result["isError"].as_bool().unwrap_or(false) {
            return Err(text);
        }
        Ok(serde_json::from_str(&text).unwrap_or(Value::String(text)))
    }

    /// Every text block of a successful tool result, in order.
    pub fn tool_texts(&mut self, name: &str, arguments: Value) -> Vec<String> {
        let id = self.send_tool(name, arguments);
        let result = self.reply(id);
        assert!(
            !result["isError"].as_bool().unwrap_or(false),
            "tool error: {result}"
        );
        result["content"]
            .as_array()
            .unwrap_or_else(|| panic!("tool result without content: {result}"))
            .iter()
            .map(|block| block["text"].as_str().expect("text block").to_string())
            .collect()
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    pub fn tool(&mut self, name: &str, arguments: Value) -> Result<Value, String> {
        let id = self.send_tool(name, arguments);
        self.tool_reply(id)
    }
}

impl Drop for Mcp {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
