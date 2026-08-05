// ~/.config/ir/config.yml — collection registry
// docs: https://docs.rs/serde_yaml

mod context;
pub use context::detect_collection;

use crate::error::{Error, Result};
use crate::types::Collection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Once;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub collections: Vec<Collection>,
    /// alias → command string (e.g. "ko" → "kiwi-tokenize", "ja" → "mecab -Owakati")
    #[serde(default)]
    pub preprocessors: HashMap<String, String>,
    /// Global retrieval-pipeline overrides (top-level block). Used for the
    /// daemon-global `expander` decision; per-collection query knobs live on each
    /// `Collection`. `None` → built-in defaults. See `search::profile`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval: Option<crate::types::RetrievalConfig>,
}

impl Config {
    pub fn load() -> Result<Self> {
        let path = config_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = fs::read_to_string(&path)?;
        let mut config: Self = serde_yaml::from_str(&content)?;
        for col in &mut config.collections {
            if col.path.contains('~') || col.path.contains('$') {
                col.path = expand_path(&col.path).to_string_lossy().into_owned();
            }
        }
        Ok(config)
    }

    pub fn save(&self) -> Result<()> {
        let path = config_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let yaml = serde_yaml::to_string(self)?;
        fs::write(&path, yaml)?;
        Ok(())
    }

    pub fn get_collection(&self, name: &str) -> Option<&Collection> {
        self.collections.iter().find(|c| c.name == name)
    }

    pub fn add_collection(&mut self, collection: Collection) -> Result<()> {
        validate_collection_name(&collection.name)?;
        if self.get_collection(&collection.name).is_some() {
            return Err(Error::CollectionExists(collection.name));
        }
        self.collections.push(collection);
        Ok(())
    }

    pub fn remove_collection(&mut self, name: &str) -> Result<()> {
        let pos = self
            .collections
            .iter()
            .position(|c| c.name == name)
            .ok_or_else(|| Error::CollectionNotFound(name.to_string()))?;
        self.collections.remove(pos);
        Ok(())
    }

    pub fn set_collection_path(&mut self, name: &str, new_path: &str) -> Result<()> {
        let store_path = portable_path(new_path)?;
        let col = self
            .collections
            .iter_mut()
            .find(|c| c.name == name)
            .ok_or_else(|| Error::CollectionNotFound(name.to_string()))?;
        col.path = store_path;
        Ok(())
    }

    pub fn add_preprocessor(&mut self, alias: &str, command: &str) -> Result<()> {
        if alias.is_empty() || alias.contains(' ') {
            return Err(Error::Other(format!(
                "invalid preprocessor alias: {alias:?}"
            )));
        }
        self.preprocessors
            .insert(alias.to_string(), command.to_string());
        Ok(())
    }

    pub fn remove_preprocessor(&mut self, alias: &str) -> Result<()> {
        if self.preprocessors.remove(alias).is_none() {
            return Err(Error::Other(format!(
                "preprocessor alias not found: {alias:?}"
            )));
        }
        Ok(())
    }

    /// Resolve a list of alias names to their command strings.
    /// Aliases not found in the registry are skipped with a warning.
    pub fn resolve_preprocessor_commands(&self, aliases: &[String]) -> Vec<String> {
        aliases
            .iter()
            .filter_map(|alias| match self.preprocessors.get(alias) {
                Some(cmd) => Some(cmd.clone()),
                None => {
                    eprintln!("warning: preprocessor alias '{alias}' not found — skipping");
                    None
                }
            })
            .collect()
    }

    pub fn rename_collection(&mut self, old: &str, new: &str) -> Result<()> {
        validate_collection_name(new)?;
        if self.get_collection(new).is_some() {
            return Err(Error::CollectionExists(new.to_string()));
        }
        let col = self
            .collections
            .iter_mut()
            .find(|c| c.name == old)
            .ok_or_else(|| Error::CollectionNotFound(old.to_string()))?;
        col.name = new.to_string();
        Ok(())
    }
}

/// Base directory for all ir state.
///
/// Precedence: `IR_CONFIG_DIR` > `XDG_CONFIG_HOME/ir` > `~/.config/ir`.
/// Both `IR_CONFIG_DIR` and `XDG_CONFIG_HOME` support `~` and `$VAR` expansion.
/// `XDG_CONFIG_HOME` is deprecated in favor of `IR_CONFIG_DIR`.
pub fn ir_dir() -> PathBuf {
    if let Ok(val) = std::env::var("IR_CONFIG_DIR") {
        return expand_path(&val);
    }
    if let Ok(val) = std::env::var("XDG_CONFIG_HOME") {
        static WARN: Once = Once::new();
        WARN.call_once(|| {
            eprintln!(
                "warning: XDG_CONFIG_HOME is deprecated for ir; use IR_CONFIG_DIR=<path> instead"
            );
        });
        return expand_path(&val).join("ir");
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/"))
        .join(".config")
        .join("ir")
}

/// Parse a boolean env flag: true for `1`/`true`/`yes`/`on` (case-insensitive).
/// Shared so every research flag (`IR_GRAPH_*`, `IR_ANN`, `IR_GRAPH_BUILD`) parses
/// identically — a mismatch across call sites produces quietly wrong A/B sweeps.
pub fn env_flag(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|raw| {
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// Expand `~` and `$VAR`/`${VAR}` in a path string.
///
/// - Leading `~` or `~/` is replaced with the user's home directory.
/// - `$VAR` and `${VAR}` are replaced with the env var value; unknown vars expand to empty string.
/// - No shell is invoked — safe for MCP transport where env values are JSON string literals.
pub fn expand_path(raw: &str) -> PathBuf {
    PathBuf::from(expand_vars(raw))
}

fn expand_vars(input: &str) -> String {
    // Phase 1: leading ~ expansion only
    let s = if input == "~" {
        home_dir_str()
    } else if let Some(rest) = input.strip_prefix("~/") {
        format!("{}/{rest}", home_dir_str())
    } else {
        input.to_owned()
    };

    // Phase 2: $VAR and ${VAR} expansion
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '$' {
            out.push(c);
            continue;
        }
        if chars.peek() == Some(&'{') {
            chars.next(); // consume '{'
            let name: String = chars.by_ref().take_while(|&c| c != '}').collect();
            out.push_str(&std::env::var(&name).unwrap_or_default());
        } else {
            let mut name = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_alphanumeric() || c == '_' {
                    name.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            if name.is_empty() {
                out.push('$');
            } else {
                out.push_str(&std::env::var(&name).unwrap_or_default());
            }
        }
    }
    out
}

fn home_dir_str() -> String {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/"))
        .to_string_lossy()
        .into_owned()
}

/// Determine how to store a user-supplied path in config.yml.
///
/// Paths starting with `~` or containing `$` are stored as-is (portable across machines).
/// Plain paths are canonicalized to absolute. Validates the expanded path exists.
pub fn portable_path(raw: &str) -> Result<String> {
    if raw.starts_with('~') || raw.contains('$') {
        // Validate via expansion but store the original portable form.
        let expanded = expand_path(raw);
        if !expanded.exists() {
            return Err(Error::Other(format!(
                "path does not exist: {}",
                expanded.display()
            )));
        }
        Ok(raw.to_owned())
    } else {
        let resolved = std::fs::canonicalize(raw)
            .map_err(|e| Error::Other(format!("invalid path {raw:?}: {e}")))?;
        Ok(resolved.to_string_lossy().into_owned())
    }
}

/// Path to the `config.yml` file.
///
/// Precedence: `IR_CONFIG_FILE` (set by the `--config-path` CLI arg, and inherited
/// by the spawned daemon) > `ir_dir()/config.yml`. This overrides **only the
/// config file**, not the data dir — collections, caches, and the daemon socket
/// stay under `ir_dir()`, so a benchmark can vary pipeline config across
/// candidates while reusing one embedded corpus.
pub fn config_path() -> PathBuf {
    if let Ok(val) = std::env::var("IR_CONFIG_FILE")
        && !val.is_empty()
    {
        return expand_path(&val);
    }
    ir_dir().join("config.yml")
}

pub fn data_dir() -> PathBuf {
    ir_dir().join("collections")
}

pub fn collection_db_path(name: &str) -> PathBuf {
    data_dir().join(format!("{name}.sqlite"))
}

pub fn daemon_socket_path() -> PathBuf {
    ir_dir().join("daemon.sock")
}

pub fn daemon_pid_path() -> PathBuf {
    ir_dir().join("daemon.pid")
}

pub fn daemon_tier2_path() -> PathBuf {
    ir_dir().join("daemon.tier2")
}

pub fn daemon_lock_path() -> PathBuf {
    ir_dir().join("daemon.lock")
}

pub fn expander_cache_path() -> PathBuf {
    ir_dir().join("expander_cache.sqlite")
}

fn validate_collection_name(name: &str) -> Result<()> {
    if name.is_empty() || name.contains('/') || name.contains('\0') || name.contains("..") {
        return Err(Error::Other(format!("invalid collection name: {name:?}")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Collection;

    fn col(name: &str) -> Collection {
        Collection {
            name: name.to_string(),
            path: "/tmp".into(),
            globs: vec![],
            excludes: vec![],
            description: None,
            preprocessor: None,
            routing: None,
            retrieval: None,
        }
    }

    #[test]
    fn rejects_empty_name() {
        let mut cfg = Config::default();
        assert!(cfg.add_collection(col("")).is_err());
    }

    #[test]
    fn rejects_slash_in_name() {
        let mut cfg = Config::default();
        assert!(cfg.add_collection(col("a/b")).is_err());
    }

    #[test]
    fn rejects_dotdot_in_name() {
        let mut cfg = Config::default();
        assert!(cfg.add_collection(col("..")).is_err());
        assert!(cfg.add_collection(col("a..b")).is_err());
    }

    #[test]
    fn rejects_null_byte() {
        let mut cfg = Config::default();
        assert!(cfg.add_collection(col("a\0b")).is_err());
    }

    #[test]
    fn accepts_valid_names() {
        let mut cfg = Config::default();
        assert!(cfg.add_collection(col("knowledge")).is_ok());
        assert!(cfg.add_collection(col("my-notes_2024")).is_ok());
    }

    #[test]
    fn rename_validates_new_name() {
        let mut cfg = Config::default();
        cfg.add_collection(col("notes")).unwrap();
        assert!(cfg.rename_collection("notes", "a/b").is_err());
        assert!(cfg.rename_collection("notes", "ok-name").is_ok());
    }

    #[test]
    fn expand_tilde() {
        let home = dirs::home_dir().unwrap();
        assert_eq!(expand_path("~"), home);
        assert_eq!(expand_path("~/Documents"), home.join("Documents"));
        assert_eq!(expand_path("~/a/b/c"), home.join("a/b/c"));
    }

    #[test]
    fn expand_no_tilde_in_middle() {
        // ~ not at start is literal
        assert_eq!(expand_path("/a/~/b"), PathBuf::from("/a/~/b"));
    }

    #[test]
    fn expand_env_var() {
        unsafe { std::env::set_var("IR_TEST_EXPAND_VAR", "/custom/path") };
        assert_eq!(
            expand_path("$IR_TEST_EXPAND_VAR/sub"),
            PathBuf::from("/custom/path/sub")
        );
        unsafe { std::env::remove_var("IR_TEST_EXPAND_VAR") };
    }

    #[test]
    fn expand_braced_var() {
        unsafe { std::env::set_var("IR_TEST_EXPAND_BRACED", "/braced") };
        assert_eq!(
            expand_path("${IR_TEST_EXPAND_BRACED}/sub"),
            PathBuf::from("/braced/sub")
        );
        unsafe { std::env::remove_var("IR_TEST_EXPAND_BRACED") };
    }

    #[test]
    fn expand_unknown_var_is_empty() {
        // Unknown vars expand to empty string, leaving the slash.
        let result = expand_path("$IR_THIS_VAR_DOES_NOT_EXIST_IR/sub");
        assert_eq!(result, PathBuf::from("/sub"));
    }

    #[test]
    fn expand_bare_dollar_is_literal() {
        assert_eq!(expand_path("/a/$/b"), PathBuf::from("/a/$/b"));
    }

    #[test]
    fn expand_absolute_unchanged() {
        assert_eq!(
            expand_path("/absolute/path"),
            PathBuf::from("/absolute/path")
        );
    }

    #[test]
    fn expand_tilde_and_var_combined() {
        unsafe { std::env::set_var("IR_TEST_EXPAND_SUBDIR", "vault") };
        let home = dirs::home_dir().unwrap();
        assert_eq!(
            expand_path("~/$IR_TEST_EXPAND_SUBDIR/.config/ir"),
            home.join("vault/.config/ir"),
        );
        unsafe { std::env::remove_var("IR_TEST_EXPAND_SUBDIR") };
    }

    // Mutex to serialize tests that mutate IR_CONFIG_DIR / XDG_CONFIG_HOME.
    // Recover from poison (test panicked while holding lock) to avoid cascading failures.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn ir_dir_respects_ir_config_dir() {
        let _guard = env_lock();
        let saved_xdg = std::env::var("XDG_CONFIG_HOME").ok();
        unsafe {
            std::env::remove_var("XDG_CONFIG_HOME");
            std::env::set_var("IR_CONFIG_DIR", "/custom/ir");
        }
        let result = ir_dir();
        unsafe {
            std::env::remove_var("IR_CONFIG_DIR");
            match saved_xdg {
                Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
        }
        assert_eq!(result, PathBuf::from("/custom/ir"));
    }

    #[test]
    fn ir_dir_ir_config_dir_with_tilde() {
        let _guard = env_lock();
        let home = dirs::home_dir().unwrap();
        let saved_xdg = std::env::var("XDG_CONFIG_HOME").ok();
        unsafe {
            std::env::remove_var("XDG_CONFIG_HOME");
            std::env::set_var("IR_CONFIG_DIR", "~/.config/ir");
        }
        let result = ir_dir();
        unsafe {
            std::env::remove_var("IR_CONFIG_DIR");
            match saved_xdg {
                Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
        }
        assert_eq!(result, home.join(".config/ir"));
    }

    #[test]
    fn ir_dir_xdg_fallback() {
        let _guard = env_lock();
        let saved_ir = std::env::var("IR_CONFIG_DIR").ok();
        let saved_xdg = std::env::var("XDG_CONFIG_HOME").ok();
        unsafe {
            std::env::remove_var("IR_CONFIG_DIR");
            std::env::set_var("XDG_CONFIG_HOME", "/tmp/test-xdg");
        }
        let result = ir_dir();
        unsafe {
            match saved_ir {
                Some(v) => std::env::set_var("IR_CONFIG_DIR", v),
                None => std::env::remove_var("IR_CONFIG_DIR"),
            }
            match saved_xdg {
                Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
        }
        assert_eq!(result, PathBuf::from("/tmp/test-xdg/ir"));
    }

    #[test]
    fn portable_path_preserves_tilde() {
        // ~ paths are stored as-is (not expanded) if the expanded path exists.
        let home = dirs::home_dir().unwrap();
        // Use ~/.config which almost certainly exists on any dev machine.
        let tilde_path = "~/.config";
        if home.join(".config").exists() {
            let result = portable_path(tilde_path).unwrap();
            assert_eq!(
                result, tilde_path,
                "tilde path must be stored as-is, not expanded"
            );
        }
    }

    #[test]
    fn portable_path_nonexistent_returns_err() {
        let result = portable_path("~/__ir_nonexistent_path_xyz__/sub");
        assert!(result.is_err(), "nonexistent tilde path must return Err");
    }
}
