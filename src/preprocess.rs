// Preprocessor Plugin Protocol
//
// Any executable that:
// 1. Reads UTF-8 lines from stdin
// 2. Writes 0 or 1 UTF-8 lines to stdout per input line (0 if all tokens filtered)
// 3. Flushes stdout after each line (or after producing no output for a line)
// 4. Passes ASCII-only single-word lines through unchanged (required for sentinel protocol)
// 5. Stays alive between lines (no exit-per-line)
//
// ir uses a sentinel-based protocol to handle case (2): after each content line,
// ir also sends SENTINEL (ASCII-only). The subprocess echoes SENTINEL unchanged.
// ir reads until it sees SENTINEL, so a dropped content line doesn't deadlock.
//
// Register: ir preprocessor add <alias> <command>
// Bind:     ir collection add <name> <path> --preprocessor <alias>
//
// Examples:
//   ir preprocessor install ko   (installs lindera-tokenize, registers as "ko")
//   ir preprocessor add ja "mecab -Owakati"
//   ir collection add wiki ~/wiki --preprocessor ko
//
// Subprocess lifetime: stays alive for batch indexing; spawned per-query for search.
// Lindera startup: <10ms (Rust binary, embedded dictionary).

use crate::config::expand_path;
use crate::error::{Error, Result};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

// ^ ASCII-only, passes through all lindera language filters unchanged (POS = SL/foreign word)
const SENTINEL: &str = "IRSENTINEL";

// ^ Largest line written before reading the reply. Line + sentinel must fit the
//   smallest OS pipe buffer (4 KiB) so the write can never block.
const INLINE_WRITE_MAX: usize = 4000;

fn failed(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Other(format!("preprocessor {what}: {e}"))
}

fn write_line(stdin: &mut BufWriter<ChildStdin>, line: &str) -> std::io::Result<()> {
    stdin.write_all(line.as_bytes())?;
    stdin.write_all(b"\n")?;
    stdin.write_all(SENTINEL.as_bytes())?;
    stdin.write_all(b"\n")?;
    stdin.flush()
}

/// Collect output lines up to the sentinel. The subprocess closing its output
/// first means it died: its reply is incomplete and must not be indexed.
fn read_reply(stdout: &mut BufReader<ChildStdout>) -> Result<String> {
    let mut parts = Vec::new();
    let mut buf = String::new();
    loop {
        buf.clear();
        if stdout
            .read_line(&mut buf)
            .map_err(|e| failed("read failed", e))?
            == 0
        {
            return Err(failed("exited", "no reply for the current line"));
        }
        let s = buf.trim_end_matches('\n');
        if s == SENTINEL {
            return Ok(parts.join(" "));
        }
        parts.push(s.to_string());
    }
}

pub struct PreprocessHandle {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl PreprocessHandle {
    /// Spawn a preprocessor subprocess from a command string (e.g. "mecab -Owakati").
    /// The error names the command, the reason and how to reinstall.
    pub fn spawn(cmd_str: &str) -> Result<Self> {
        // ! paths with spaces unsupported; commands must be simple tokens (e.g. "mecab -Owakati")
        let mut parts = cmd_str.split_whitespace();
        let Some(raw_program) = parts.next() else {
            return Err(failed("failed to start", "empty command"));
        };
        let expanded = expand_path(raw_program);
        let program = expanded.as_os_str();
        // Preprocessor commands embed variable references in args (e.g. --dict $IR_DIR/preprocessors/jieba);
        // Command::args() does not invoke a shell, so variables must be expanded explicitly.
        let args: Vec<std::ffi::OsString> =
            parts.map(|a| expand_path(a).into_os_string()).collect();

        match Command::new(program)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
        {
            Ok(mut child) => {
                let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
                    return Err(failed("failed to start", "no stdio pipes"));
                };
                Ok(Self {
                    child,
                    stdin: BufWriter::new(stdin),
                    stdout: BufReader::new(stdout),
                })
            }
            Err(e) => Err(Error::Other(format!(
                "failed to spawn preprocessor '{cmd_str}': {e} \
                 (run `ir preprocessor install <lang>` to reinstall it)"
            ))),
        }
    }

    pub fn process_line(&mut self, line: &str) -> Result<String> {
        if line.trim().is_empty() {
            return Ok(String::new());
        }
        // Content + sentinel are flushed together; the sentinel ends the reply even
        // when the content line produced no output (see protocol spec at top of file).
        let Self { stdin, stdout, .. } = self;
        if line.len() <= INLINE_WRITE_MAX {
            write_line(stdin, line).map_err(|e| failed("write failed", e))?;
            return read_reply(stdout);
        }
        // ^ A streaming filter echoes while the line is still being written; past
        //   the pipe buffers both sides would block forever. Write from a helper
        //   thread so the reply is drained at the same time.
        std::thread::scope(|scope| {
            let writer = scope.spawn(|| write_line(stdin, line));
            let reply = read_reply(stdout);
            match writer.join() {
                Ok(Ok(())) => reply,
                Ok(Err(e)) => Err(failed("write failed", e)),
                Err(_) => Err(failed("write failed", "writer thread panicked")),
            }
        })
    }

    /// Process multi-line text: split on '\n', process each line, rejoin.
    pub fn process_text(&mut self, text: &str) -> Result<String> {
        let lines: Vec<&str> = text.split('\n').collect();
        let mut out = Vec::with_capacity(lines.len());
        for line in lines {
            out.push(self.process_line(line)?);
        }
        Ok(out.join("\n"))
    }
}

impl Drop for PreprocessHandle {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Chain of preprocessors — output of one feeds into the next.
/// Each specializes in one language and passes other text through unchanged.
/// FTS5's porter unicode61 tokenizer always runs after as the final stage.
pub struct PreprocessChain {
    handles: Vec<PreprocessHandle>,
    /// Why the first command that did not spawn failed.
    spawn_error: Option<String>,
}

impl PreprocessChain {
    /// Spawn a chain from a list of command strings.
    /// Handles that fail to spawn are skipped; `spawn_error` says why.
    pub fn spawn(commands: &[String]) -> Self {
        let mut spawn_error = None;
        let handles = commands
            .iter()
            .filter_map(|cmd| {
                PreprocessHandle::spawn(cmd)
                    .inspect_err(|e| {
                        spawn_error.get_or_insert(e.to_string());
                    })
                    .ok()
            })
            .collect();
        Self {
            handles,
            spawn_error,
        }
    }

    /// Spawn every command or fail. Indexing uses this: text processed by a
    /// partial chain would not match what queries are preprocessed with.
    pub fn spawn_all(commands: &[String]) -> Result<Self> {
        let handles = commands
            .iter()
            .map(|cmd| PreprocessHandle::spawn(cmd))
            .collect::<Result<_>>()?;
        Ok(Self {
            handles,
            spawn_error: None,
        })
    }

    /// Pipe text through all handles in sequence.
    pub fn process_text(&mut self, text: &str) -> Result<String> {
        let mut current = text.to_string();
        for handle in &mut self.handles {
            current = handle.process_text(&current)?;
        }
        Ok(current)
    }

    /// Returns true if at least one preprocessor handle was successfully spawned.
    pub fn is_active(&self) -> bool {
        !self.handles.is_empty()
    }

    pub fn spawn_error(&self) -> Option<&str> {
        self.spawn_error.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Requires: ir preprocessor install zh
    // Run: cargo test -- --ignored zh_preprocessor_tokenizes_chinese
    #[cfg(unix)]
    #[ignore]
    #[test]
    fn zh_preprocessor_tokenizes_chinese() {
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let config = match crate::config::Config::load() {
            Ok(c) => c,
            Err(_) => return,
        };
        let cmd_str = match config.preprocessors.get("zh") {
            Some(cmd) => cmd.clone(),
            None => return, // zh not registered — skip
        };
        // IR_DIR must be set for $IR_DIR/... expansion in the command
        unsafe { std::env::set_var("IR_DIR", crate::config::ir_dir()) };

        let mut handle = match PreprocessHandle::spawn(&cmd_str) {
            Ok(h) => h,
            Err(e) => panic!("zh preprocessor is registered but failed to spawn: {e}"),
        };

        // Segmentation: jieba must split 你好世界 into separate tokens
        let out = handle.process_line("你好世界").unwrap();
        assert_ne!(
            out, "你好世界",
            "jieba must segment Chinese words (got identical output)"
        );
        assert!(
            !out.is_empty(),
            "jieba must produce output for Chinese text"
        );

        // ASCII must pass through unchanged — the sentinel protocol relies on this.
        // (Cannot use IRSENTINEL itself as content: the read loop would treat its echo as the
        // sentinel terminator and return "" — use a different ASCII word instead.)
        let ascii_out = handle.process_line("hello").unwrap();
        assert_eq!(
            ascii_out, "hello",
            "ASCII words must pass through zh preprocessor unchanged"
        );

        // Punctuation line: must not deadlock; result is either passed through or empty
        let _ = handle.process_line("。").unwrap();
    }
}
