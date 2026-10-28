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
    /// Filesystem modification time, or None if unavailable.
    pub mtime: Option<std::time::SystemTime>,
    /// Filesystem creation/birth time. Not available on all Linux filesystems; falls back to mtime.
    pub birthtime: Option<std::time::SystemTime>,
}

pub fn scan(collection: &Collection) -> Result<Vec<ScannedFile>> {
    let default = ["**/*.md".to_string()];
    let globs: &[String] = if collection.globs.is_empty() {
        &default
    } else {
        &collection.globs
    };
    scan_globs(&collection.path, globs, &collection.excludes)
}

/// Files under `root` matching any of `globs` and none of `excludes`.
/// Respects .gitignore. Also used for auxiliary-domain sources.
pub fn scan_globs(root: &str, globs: &[String], excludes: &[String]) -> Result<Vec<ScannedFile>> {
    let mut override_builder = ignore::overrides::OverrideBuilder::new(root);
    for glob in globs {
        override_builder
            .add(glob)
            .map_err(|e| crate::error::Error::Other(e.to_string()))?;
    }
    for exclude in excludes {
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
            let meta = entry.metadata().ok();
            let mtime = meta.as_ref().and_then(|m| m.modified().ok());
            let birthtime = meta.as_ref().and_then(|m| m.created().ok());
            files.push(ScannedFile {
                abs_path,
                rel_path,
                mtime,
                birthtime,
            });
        }
    }

    Ok(files)
}
