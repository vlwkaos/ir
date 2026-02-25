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

    pub fn rename_collection(&mut self, old: &str, new: &str) -> Result<()> {
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

pub fn config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("~/.config"))
        .join("ir")
        .join("config.yml")
}

pub fn data_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("~/.local/share"))
        .join("ir")
        .join("collections")
}

pub fn collection_db_path(name: &str) -> PathBuf {
    data_dir().join(format!("{name}.sqlite"))
}
