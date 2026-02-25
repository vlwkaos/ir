// File discovery using the ignore crate (respects .gitignore).
// docs: https://docs.rs/ignore/latest/ignore/

use crate::error::Result;
use crate::types::Collection;
use ignore::WalkBuilder;
use std::path::PathBuf;

#[derive(Debug)]
pub struct ScannedFile {
    /// Absolute path to file.
    pub abs_path: PathBuf,
    /// Path relative to the collection root.
    pub rel_path: String,
}

pub fn scan(collection: &Collection) -> Result<Vec<ScannedFile>> {
    let root = &collection.path;
    let globs: Vec<&str> = if collection.globs.is_empty() {
        vec!["**/*.md"]
    } else {
        collection.globs.iter().map(|s| s.as_str()).collect()
    };

    let mut override_builder = ignore::overrides::OverrideBuilder::new(root);
    for glob in &globs {
        override_builder
            .add(glob)
            .map_err(|e| crate::error::Error::Other(e.to_string()))?;
    }
    for exclude in &collection.excludes {
        // Prefix with ! to negate
        override_builder
            .add(&format!("!{exclude}"))
            .map_err(|e| crate::error::Error::Other(e.to_string()))?;
    }
    let overrides = override_builder
        .build()
        .map_err(|e| crate::error::Error::Other(e.to_string()))?;

    let walk = WalkBuilder::new(root)
        .overrides(overrides)
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .build();

    let root_path = std::path::Path::new(root);
    let mut files = Vec::new();

    for entry in walk {
        let entry = entry.map_err(|e| crate::error::Error::Other(e.to_string()))?;
        if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            let abs_path = entry.path().to_path_buf();
            let rel_path = abs_path
                .strip_prefix(root_path)
                .unwrap_or(&abs_path)
                .to_string_lossy()
                .into_owned();
            files.push(ScannedFile { abs_path, rel_path });
        }
    }

    Ok(files)
}
