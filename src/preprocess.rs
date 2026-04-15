// Preprocessor Plugin Protocol
//
// Any executable that:
// 1. Reads UTF-8 lines from stdin (one line per invocation of process_line)
// 2. Writes exactly one UTF-8 line to stdout per input line
// 3. Flushes stdout after each line
// 4. Handles empty lines (write empty line back)
// 5. Stays alive between lines (no exit-per-line)
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

use crate::error::Result;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

pub struct PreprocessHandle {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl PreprocessHandle {
    /// Spawn a preprocessor subprocess from a command string (e.g. "mecab -Owakati").
    /// Returns None on spawn failure (logs warning).
    pub fn spawn(cmd_str: &str) -> Option<Self> {
        // ! paths with spaces unsupported; commands must be simple tokens (e.g. "mecab -Owakati")
        let mut parts = cmd_str.split_whitespace();
        let program = parts.next()?;
        let args: Vec<&str> = parts.collect();

        match Command::new(program)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
        {
            Ok(mut child) => {
                let stdin = BufWriter::new(child.stdin.take()?);
                let stdout = BufReader::new(child.stdout.take()?);
                Some(Self { child, stdin, stdout })
            }
            Err(e) => {
                eprintln!("warning: failed to spawn preprocessor '{cmd_str}': {e}");
                None
            }
        }
    }

    pub fn process_line(&mut self, line: &str) -> Result<String> {
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let mut out = String::new();
        self.stdout.read_line(&mut out)?;
        Ok(out.trim_end_matches('\n').to_string())
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
}

impl PreprocessChain {
    /// Spawn a chain from a list of command strings.
    /// Handles that fail to spawn are skipped with a warning.
    pub fn spawn(commands: &[String]) -> Self {
        let handles = commands
            .iter()
            .filter_map(|cmd| PreprocessHandle::spawn(cmd))
            .collect();
        Self { handles }
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
}

/// Preprocess a query through a command chain. Falls back to raw query on spawn failure or I/O error.
#[cfg(test)]
pub fn preprocess_query(query: &str, commands: &[String]) -> String {
    if commands.is_empty() {
        return query.to_string();
    }
    let mut chain = PreprocessChain::spawn(commands);
    if !chain.is_active() {
        eprintln!(
            "warning: preprocessor configured ({}) but failed to start; using raw query",
            commands.join(", ")
        );
        return query.to_string();
    }
    chain.process_text(query).unwrap_or_else(|_| query.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_preprocessor_with_cat() {
        let mut chain = PreprocessChain::spawn(&["cat".to_string()]);
        assert!(chain.is_active(), "cat should spawn successfully");
        let out = chain.process_text("hello world").unwrap();
        assert_eq!(out, "hello world");
    }

    #[test]
    fn chain_with_invalid_command_falls_back() {
        let cmds = vec!["__nonexistent_command_xyz__".to_string()];
        let chain = PreprocessChain::spawn(&cmds);
        assert!(!chain.is_active(), "invalid command should not spawn");
    }

    #[test]
    fn multiline_text_preserved() {
        let mut chain = PreprocessChain::spawn(&["cat".to_string()]);
        assert!(chain.is_active());
        let input = "line one\nline two\nline three";
        let out = chain.process_text(input).unwrap();
        assert_eq!(out, input);
    }

    #[cfg(unix)]
    #[test]
    fn rev_transforms_text() {
        let mut chain = PreprocessChain::spawn(&["rev".to_string()]);
        assert!(chain.is_active());
        let out = chain.process_text("hello world").unwrap();
        assert_eq!(out, "dlrow olleh");
    }

    #[cfg(unix)]
    #[test]
    fn chain_pipes_through_multiple() {
        let cmds = vec!["rev".to_string(), "rev".to_string()];
        let mut chain = PreprocessChain::spawn(&cmds);
        assert!(chain.is_active());
        let out = chain.process_text("hello world").unwrap();
        assert_eq!(out, "hello world");
    }

    #[cfg(unix)]
    #[test]
    fn preprocess_query_applies_transformation() {
        let cmds = vec!["rev".to_string()];
        let out = preprocess_query("hello world", &cmds);
        assert_eq!(out, "dlrow olleh");
    }

    #[cfg(unix)]
    #[test]
    fn preprocess_query_empty_commands_returns_raw() {
        let out = preprocess_query("HELLO WORLD", &[]);
        assert_eq!(out, "HELLO WORLD");
    }

    #[cfg(unix)]
    #[test]
    fn preprocess_query_invalid_command_returns_raw() {
        let cmds = vec!["__nonexistent_command_xyz__".to_string()];
        let out = preprocess_query("HELLO WORLD", &cmds);
        assert_eq!(out, "HELLO WORLD");
    }

    #[cfg(unix)]
    #[test]
    fn chain_skips_invalid_handles() {
        let cmds = vec!["__nonexistent_command_xyz__".to_string(), "rev".to_string()];
        let mut chain = PreprocessChain::spawn(&cmds);
        assert!(chain.is_active());
        let out = chain.process_text("hello world").unwrap();
        assert_eq!(out, "dlrow olleh");
    }

    #[cfg(unix)]
    #[test]
    fn multiline_with_transformation() {
        let mut chain = PreprocessChain::spawn(&["rev".to_string()]);
        assert!(chain.is_active());
        let out = chain.process_text("hello\nworld").unwrap();
        assert_eq!(out, "olleh\ndlrow");
    }
}
