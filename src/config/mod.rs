// ~/.config/ir/config.yml — collection registry
// docs: https://docs.rs/serde_yaml

mod context;
pub use context::detect_collection;

use crate::error::{Error, Result};
use crate::types::Collection;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub collections: Vec<Collection>,
}

impl Config {
    pub fn load() -> Result<Self> {
        let path = config_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = fs::read_to_string(&path)?;
        Ok(serde_yaml::from_str(&content)?)
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
        let resolved = std::fs::canonicalize(new_path)
            .map_err(|e| Error::Other(format!("invalid path {new_path:?}: {e}")))?;
        let col = self
            .collections
            .iter_mut()
            .find(|c| c.name == name)
            .ok_or_else(|| Error::CollectionNotFound(name.to_string()))?;
        col.path = resolved.to_string_lossy().into_owned();
        Ok(())
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

/// Base directory for all ir state: $XDG_CONFIG_HOME/ir or ~/.config/ir.
/// Consistent across platforms; avoids platform-specific paths like ~/Library.
pub(crate) fn ir_dir() -> PathBuf {
    std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("~"))
                .join(".config")
        })
        .join("ir")
}

pub fn config_path() -> PathBuf {
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
}
