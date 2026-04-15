// Model auto-download via HuggingFace Hub.
// docs: https://docs.rs/hf-hub/latest/hf_hub/
//
// HF_HUB_OFFLINE=1  — skip network, use cache only (handled by hf-hub natively)
// IR_GPU_LAYERS=N   — override GPU layer count

use crate::error::{Error, Result};
use crate::llm::{find_model, hf_repos, models};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// Return the local path for `filename`, downloading from HF if not found locally.
pub fn ensure_model(filename: &str) -> Result<PathBuf> {
    if let Some(p) = find_model(filename) {
        return Ok(p);
    }

    let (repo_id, hf_filename) = hf_repos::for_filename(filename).ok_or_else(|| {
        Error::Other(format!(
            "model '{filename}' not found and no HF repo configured.\n\
             Set the appropriate IR_*_MODEL env var or add a directory to IR_MODEL_DIRS."
        ))
    })?;

    // If a case-variant file already exists (e.g. q4_k_m vs Q4_K_M), reuse it
    // and place a stable alias at ~/.cache/ir/models/{filename}.
    if hf_filename != filename {
        if let Some(existing) = find_model(hf_filename) {
            return Ok(with_local_alias(filename, existing));
        }
    }

    // Approximate sizes for user-visible UX message (rough order of magnitude only).
    let size_hint = match filename {
        models::EMBEDDING => "~300 MB",
        models::RERANKER => "~600 MB",
        models::EXPANDER => "~1 GB",
        models::QWEN35_0_8B => "~800 MB",
        models::QWEN35_2B => "~1.5 GB",
        models::BGE_M3 => "~600 MB",
        _ => "unknown size",
    };
    eprintln!(
        "ir: first run — downloading model\n\
         \x20   model : {filename} ({size_hint})\n\
         \x20   from  : https://huggingface.co/{repo_id}\n\
         \x20   cache : ~/.cache/huggingface  (symlinked to ~/.cache/ir/models/)\n\
         \x20   tip   : set HF_HUB_OFFLINE=1 to skip network; use IR_*_MODEL=<path> for a local file"
    );

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()
        .map_err(|e| Error::Other(format!("hf-hub init: {e}")))?;

    // Alias is only created after api.get() succeeds — partial downloads leave no alias.
    let downloaded = api
        .model(repo_id.to_string())
        .get(hf_filename)
        .map_err(|e| {
            let repo_cache_slug = repo_id.replace('/', "--");
            Error::Other(format!(
                "download failed: {hf_filename} from {repo_id}\n  \
                 cause: {e}\n  \
                 fixes:\n    \
                 - check network and retry\n    \
                 - set HF_HUB_OFFLINE=1 and point IR_*_MODEL at a local .gguf\n    \
                 - manually download from https://huggingface.co/{repo_id} and \
                   put the file in ~/.cache/ir/models/ (or ~/local-models/)\n    \
                 - if a partial download is stuck, delete \
                   ~/.cache/huggingface/hub/models--{repo_cache_slug}/"
            ))
        })?;

    Ok(with_local_alias(filename, downloaded))
}

/// Check whether a string looks like a HuggingFace repo ID (`owner/name`).
///
/// Returns `Some(s)` when `s` has exactly one `/`, both sides non-empty, no
/// whitespace or backslash, and does not start with `.`, `/`, or `~`
/// (those prefixes indicate a local path).
///
/// Path existence (`is_file` / `is_dir`) is checked by the caller BEFORE
/// calling this function, so a real relative path that happens to contain
/// one `/` is handled correctly.
pub fn as_hf_repo_id(s: &str) -> Option<&str> {
    let t = s.trim();
    if t.is_empty() { return None; }
    // Local path prefixes.
    if t.starts_with('.') || t.starts_with('/') || t.starts_with('~') { return None; }
    if t.contains(char::is_whitespace) || t.contains('\\') { return None; }
    let mut parts = t.splitn(3, '/');
    let owner = parts.next()?;
    let repo = parts.next()?;
    if parts.next().is_some() { return None; } // three or more segments
    if owner.is_empty() || repo.is_empty() { return None; }
    Some(t)
}

fn known_repos_display() -> String {
    hf_repos::all_known_repos()
        .iter()
        .map(|(repo_id, _)| format!("    {repo_id}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Resolve one of the env vars in `env_vars` to a local model path.
///
/// Checks (in order) local file → local directory → HF repo ID → error.
/// Returns `Ok(None)` when no env var is set.
///
/// `dir_candidates` lists filenames to look for when the env var points at a
/// directory (e.g. `[models::EMBEDDING, models::BGE_M3]` for the embedding slot).
pub fn resolve_env_hf_or_path(
    env_vars: &[&str],
    dir_candidates: &[&str],
) -> Result<Option<PathBuf>> {
    for key in env_vars {
        let Some(raw_os) = std::env::var_os(key) else {
            continue;
        };
        let raw = raw_os.to_string_lossy().into_owned();
        let path = PathBuf::from(&raw_os);

        if path.is_file() {
            return Ok(Some(path));
        }
        if path.is_dir() {
            for cand in dir_candidates {
                let p = path.join(cand);
                if p.is_file() {
                    return Ok(Some(p));
                }
            }
            return Err(Error::Other(format!(
                "{key}={raw:?} is a directory but contains none of: {}\n  \
                 Use a direct path to a .gguf file to disambiguate.",
                dir_candidates.join(", ")
            )));
        }

        // Check for HF repo ID before emitting an error.
        if let Some(repo_id) = as_hf_repo_id(&raw) {
            if let Some(local_filename) = hf_repos::local_filename_for_repo(repo_id) {
                return Ok(Some(ensure_model(local_filename)?));
            }
            return Err(Error::Other(format!(
                "{key}={raw:?} looks like a HuggingFace repo ID but is not in the known list.\n\
                 Known repos:\n{}\n  \
                 Accepted forms for IR_*_MODEL: path to a .gguf file, directory, or known HF repo ID.",
                known_repos_display()
            )));
        }

        return Err(Error::Other(format!(
            "{key}={raw:?} is not a file, directory, or known HuggingFace repo ID.\n\
             Accepted forms:\n  \
             - path to a .gguf file\n  \
             - directory containing one of: {}\n  \
             - HuggingFace repo ID (owner/name) from the known list:\n{}\n  \
             Unset the env var to use the default model.",
            dir_candidates.join(", "),
            known_repos_display()
        )));
    }
    Ok(None)
}

/// Validate and pre-download all IR_*_MODEL env vars.
///
/// - Errors immediately on unrecognized values (not a file, dir, or known HF repo ID).
/// - Downloads any HF repo IDs that are not yet cached.
/// - Unset vars are silently skipped.
///
/// Call this in the client process BEFORE spawning the daemon so download
/// progress bars are visible in the user's terminal. The daemon then loads
/// from the local cache and starts instantly.
pub fn prepare_model_envs() -> Result<()> {
    use crate::llm::env;
    let _ = resolve_env_hf_or_path(env::EMBEDDING_MODEL, &[models::EMBEDDING, models::BGE_M3])?;
    let _ = resolve_env_hf_or_path(env::RERANKER_MODEL, &[models::RERANKER])?;
    let _ = resolve_env_hf_or_path(env::EXPANDER_MODEL, &[models::EXPANDER])?;
    // IR_COMBINED_MODEL takes priority over deprecated IR_QWEN_MODEL.
    let _ = resolve_env_hf_or_path(
        &[env::COMBINED_MODEL, env::QWEN_MODEL],
        &[models::QWEN35_2B, models::QWEN35_0_8B],
    )?;
    Ok(())
}

fn with_local_alias(filename: &str, source: PathBuf) -> PathBuf {
    match ensure_local_alias(filename, &source) {
        Ok(Some(alias)) => alias,
        Ok(None) => source,
        Err(err) => {
            eprintln!("warning: {err}");
            source
        }
    }
}

fn ensure_local_alias(filename: &str, source: &Path) -> Result<Option<PathBuf>> {
    let Some(cache_dir) = dirs::cache_dir() else {
        return Ok(None);
    };

    let alias = cache_dir.join("ir").join("models").join(filename);
    let Some(parent) = alias.parent() else {
        return Err(Error::Other(format!(
            "invalid model alias path: {}",
            alias.display()
        )));
    };

    std::fs::create_dir_all(parent).map_err(|e| {
        Error::Other(format!(
            "create local model cache dir '{}': {e}",
            parent.display()
        ))
    })?;

    match std::fs::symlink_metadata(&alias) {
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                let existing_target = std::fs::read_link(&alias).map_err(|e| {
                    Error::Other(format!(
                        "read existing model alias '{}': {e}",
                        alias.display()
                    ))
                })?;
                if existing_target == source {
                    return Ok(Some(alias));
                }
                std::fs::remove_file(&alias).map_err(|e| {
                    Error::Other(format!(
                        "remove stale model alias '{}' -> '{}': {e}",
                        alias.display(),
                        existing_target.display()
                    ))
                })?;
            } else if meta.file_type().is_file() {
                // Respect an existing real file in the canonical location.
                return Ok(Some(alias));
            } else {
                return Err(Error::Other(format!(
                    "model alias path exists but is not a file: {}",
                    alias.display()
                )));
            }
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => {
            return Err(Error::Other(format!(
                "inspect model alias path '{}': {e}",
                alias.display()
            )));
        }
    }

    create_link(source, &alias).map_err(|e| {
        Error::Other(format!(
            "create model alias '{}' -> '{}': {e}",
            alias.display(),
            source.display()
        ))
    })?;

    Ok(Some(alias))
}

#[cfg(unix)]
fn create_link(source: &Path, alias: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(source, alias)
}

#[cfg(windows)]
fn create_link(source: &Path, alias: &Path) -> std::io::Result<()> {
    // Fallback to hard links when symlink permissions are restricted.
    std::os::windows::fs::symlink_file(source, alias).or_else(|_| std::fs::hard_link(source, alias))
}

#[cfg(not(any(unix, windows)))]
fn create_link(source: &Path, alias: &Path) -> std::io::Result<()> {
    std::fs::hard_link(source, alias)
}

#[cfg(test)]
mod tests {
    use super::{as_hf_repo_id, resolve_env_hf_or_path};
    use crate::llm::{hf_repos, models};

    #[test]
    fn for_filename_resolves_all_known_models() {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(false)
            .build()
            .expect("hf-hub init");

        for filename in &[models::EMBEDDING, models::RERANKER, models::EXPANDER] {
            let (repo_id, hf_filename) = hf_repos::for_filename(filename)
                .unwrap_or_else(|| panic!("no mapping for {filename}"));
            let url = api.model(repo_id.to_string()).url(hf_filename);
            assert!(
                url.contains(repo_id),
                "url for {filename} missing repo_id: {url}"
            );
        }
    }

    #[test]
    #[ignore] // live network — run with: cargo test download -- --ignored
    fn embedding_repo_is_reachable() {
        let (repo_id, _) = hf_repos::EMBEDDING;
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(false)
            .build()
            .expect("hf-hub init");
        let info = api.model(repo_id.to_string()).info().expect("HF repo info");
        assert!(!info.siblings.is_empty(), "repo has no files?");
    }

    #[test]
    #[ignore] // live network
    fn bge_m3_repo_is_reachable() {
        let (repo_id, _) = hf_repos::BGE_M3;
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(false)
            .build()
            .expect("hf-hub init");
        let info = api.model(repo_id.to_string()).info().expect("HF repo info");
        assert!(!info.siblings.is_empty(), "repo has no files?");
    }

    #[test]
    fn unknown_filename_returns_error() {
        let result = hf_repos::for_filename("no-such-model.gguf");
        assert!(result.is_none());
    }

    // --- as_hf_repo_id ---

    #[test]
    fn as_hf_repo_id_accepts_standard_repo() {
        assert_eq!(
            as_hf_repo_id("ggml-org/bge-m3-Q8_0-GGUF"),
            Some("ggml-org/bge-m3-Q8_0-GGUF")
        );
    }

    #[test]
    fn as_hf_repo_id_rejects_absolute_path() {
        assert_eq!(as_hf_repo_id("/home/user/models/foo.gguf"), None);
    }

    #[test]
    fn as_hf_repo_id_rejects_relative_path() {
        // A relative path that starts with a directory component should be
        // caught by is_file/is_dir at the call site; as_hf_repo_id itself
        // only rejects paths starting with '.', '/', or '~'.
        // "models/foo.gguf" would pass as_hf_repo_id — but callers check
        // is_file first so it only reaches here if the path doesn't exist.
        assert_eq!(as_hf_repo_id("./models/foo.gguf"), None);
    }

    #[test]
    fn as_hf_repo_id_rejects_tilde() {
        assert_eq!(as_hf_repo_id("~/local-models/foo.gguf"), None);
    }

    #[test]
    fn as_hf_repo_id_rejects_triple_segment() {
        assert_eq!(as_hf_repo_id("a/b/c"), None);
    }

    #[test]
    fn as_hf_repo_id_rejects_empty_segment() {
        assert_eq!(as_hf_repo_id("owner/"), None);
        assert_eq!(as_hf_repo_id("/repo"), None);
    }

    #[test]
    fn as_hf_repo_id_rejects_whitespace() {
        assert_eq!(as_hf_repo_id("owner/repo name"), None);
    }

    #[test]
    fn as_hf_repo_id_rejects_backslash() {
        assert_eq!(as_hf_repo_id("owner\\repo"), None);
    }

    // --- resolve_env_hf_or_path ---

    // Use unique env var names per test to avoid parallel-test interference.

    #[test]
    fn resolve_env_hf_or_path_unset_returns_none() {
        // Use a name that won't be set in CI.
        let result = resolve_env_hf_or_path(&["IR_TEST_RESOLVE_UNSET_A"], &[]);
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn resolve_env_hf_or_path_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test-model.gguf");
        std::fs::File::create(&file).unwrap();

        // Safety: single-threaded test; no other thread reads this var.
        unsafe { std::env::set_var("IR_TEST_RESOLVE_FILE_B", &file) };
        let result = resolve_env_hf_or_path(&["IR_TEST_RESOLVE_FILE_B"], &[]);
        unsafe { std::env::remove_var("IR_TEST_RESOLVE_FILE_B") };

        assert_eq!(result.unwrap(), Some(file));
    }

    #[test]
    fn resolve_env_hf_or_path_dir_match() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join(models::EMBEDDING);
        std::fs::File::create(&model_file).unwrap();

        unsafe { std::env::set_var("IR_TEST_RESOLVE_DIR_C", dir.path()) };
        let result =
            resolve_env_hf_or_path(&["IR_TEST_RESOLVE_DIR_C"], &[models::EMBEDDING, models::BGE_M3]);
        unsafe { std::env::remove_var("IR_TEST_RESOLVE_DIR_C") };

        assert_eq!(result.unwrap(), Some(model_file));
    }

    #[test]
    fn resolve_env_hf_or_path_dir_no_match() {
        let dir = tempfile::tempdir().unwrap();
        // Create a file that is NOT in dir_candidates.
        let other = dir.path().join("some-other.gguf");
        std::fs::File::create(other).unwrap();

        unsafe { std::env::set_var("IR_TEST_RESOLVE_DIR_D", dir.path()) };
        let result = resolve_env_hf_or_path(&["IR_TEST_RESOLVE_DIR_D"], &[models::EMBEDDING]);
        unsafe { std::env::remove_var("IR_TEST_RESOLVE_DIR_D") };

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains(models::EMBEDDING),
            "error should name the expected candidates: {err}"
        );
    }

    #[test]
    fn resolve_env_hf_or_path_garbage_errors() {
        unsafe { std::env::set_var("IR_TEST_RESOLVE_GARBAGE_E", "not a path or repo") };
        let result = resolve_env_hf_or_path(&["IR_TEST_RESOLVE_GARBAGE_E"], &[models::EMBEDDING]);
        unsafe { std::env::remove_var("IR_TEST_RESOLVE_GARBAGE_E") };

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Accepted forms"),
            "error should mention accepted forms: {err}"
        );
    }

    #[test]
    fn resolve_env_hf_or_path_unknown_repo_errors() {
        unsafe { std::env::set_var("IR_TEST_RESOLVE_REPO_F", "nobody/fake-model-xyz") };
        let result = resolve_env_hf_or_path(&["IR_TEST_RESOLVE_REPO_F"], &[models::EMBEDDING]);
        unsafe { std::env::remove_var("IR_TEST_RESOLVE_REPO_F") };

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Known repos"),
            "error should list known repos: {err}"
        );
        // Should list at least the BGE-M3 repo.
        assert!(
            err.contains("ggml-org/bge-m3-Q8_0-GGUF"),
            "error should include BGE_M3 repo: {err}"
        );
    }

    #[test]
    fn prepare_model_envs_all_unset_ok() {
        use super::prepare_model_envs;
        // Skip if any real env vars are set — user's dev environment may have them pointing
        // at local files we can't stat in the test environment.
        let any_set = ["IR_EMBEDDING_MODEL", "IR_RERANKER_MODEL", "IR_EXPANDER_MODEL"]
            .iter()
            .any(|k| std::env::var_os(k).is_some());
        if !any_set {
            prepare_model_envs().expect("prepare_model_envs should succeed with no env vars set");
        }
    }

    #[test]
    fn prepare_model_envs_bad_value_errors() {
        // Test resolve_env_hf_or_path directly with a garbage value.
        unsafe { std::env::set_var("IR_TEST_RESOLVE_BAD_G", "::not-valid::") };
        let result = resolve_env_hf_or_path(&["IR_TEST_RESOLVE_BAD_G"], &[models::EMBEDDING]);
        unsafe { std::env::remove_var("IR_TEST_RESOLVE_BAD_G") };
        assert!(result.is_err(), "garbage value should produce an error");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Accepted forms"),
            "error should list accepted forms: {err}"
        );
    }
}
