// LLM model manager: backend init, model path resolution, L2 normalization.
// docs: https://docs.rs/llama-cpp-2/latest/llama_cpp_2/
//
// Model search order:
//   1. Per-model env override (IR_*_MODEL, QMD_*_MODEL aliases)
//   2. Directories from env (IR_MODEL_DIRS / QMD_MODEL_DIRS)
//   3. Built-in defaults:
//      - ~/local-models/
//      - ~/.cache/ir/models/
//      - ~/.cache/qmd/models/

pub mod combined;
pub mod download;
pub mod embedding;
pub mod expander;
pub mod generate;
pub mod reranker;
pub mod scoring;

pub use llama_cpp_2::llama_backend::LlamaBackend;
pub use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::{LlamaBackendDeviceType, list_llama_ggml_backend_devices};

use std::path::PathBuf;
use std::sync::OnceLock;

// LlamaBackend is an empty struct — Send + Sync by default. Store once per process.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
static LOGS_INIT: OnceLock<()> = OnceLock::new();

/// Known model filenames.
pub mod models {
    pub const EMBEDDING: &str = "embeddinggemma-300M-Q8_0.gguf";
    pub const RERANKER: &str = "qwen3-reranker-0.6b-q8_0.gguf";
    pub const EXPANDER: &str = "qmd-query-expansion-1.7B-q4_k_m.gguf";
    pub const QWEN35_0_8B: &str = "Qwen3.5-0.8B-Q8_0.gguf";
    pub const QWEN35_2B: &str = "Qwen3.5-2B-Q4_K_M.gguf";
    pub const BGE_M3: &str = "bge-m3-q8_0.gguf";
}

/// HuggingFace repo + filename for each known model.
/// Tuple: (repo_id, filename_on_hf)
pub mod hf_repos {
    pub const EMBEDDING: (&str, &str) = (
        "ggml-org/embeddinggemma-300M-GGUF",
        "embeddinggemma-300M-Q8_0.gguf",
    );
    pub const RERANKER: (&str, &str) = (
        "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
        "qwen3-reranker-0.6b-q8_0.gguf",
    );
    pub const EXPANDER: (&str, &str) = (
        "tobil/qmd-query-expansion-1.7B",
        // ! uppercase on HF; local name uses lowercase q4_k_m
        "qmd-query-expansion-1.7B-Q4_K_M.gguf",
    );
    pub const QWEN35_0_8B: (&str, &str) = (
        "unsloth/Qwen3.5-0.8B-GGUF",
        "Qwen3.5-0.8B-Q8_0.gguf",
    );
    pub const QWEN35_2B: (&str, &str) = (
        "unsloth/Qwen3.5-2B-GGUF",
        "Qwen3.5-2B-Q4_K_M.gguf",
    );
    pub const BGE_M3: (&str, &str) = (
        "ggml-org/bge-m3-Q8_0-GGUF",
        "bge-m3-q8_0.gguf",
    );

    /// Returns `(repo_id, hf_filename)` for a local model filename, or `None`.
    pub fn for_filename(filename: &str) -> Option<(&'static str, &'static str)> {
        match filename {
            super::models::EMBEDDING => Some(EMBEDDING),
            super::models::RERANKER => Some(RERANKER),
            super::models::EXPANDER => Some(EXPANDER),
            super::models::QWEN35_0_8B => Some(QWEN35_0_8B),
            super::models::QWEN35_2B => Some(QWEN35_2B),
            super::models::BGE_M3 => Some(BGE_M3),
            _ => None,
        }
    }

    /// All known (repo_id, hf_filename) pairs.
    pub fn all_known_repos() -> &'static [(&'static str, &'static str)] {
        &[EMBEDDING, RERANKER, EXPANDER, QWEN35_0_8B, QWEN35_2B, BGE_M3]
    }

    /// Reverse lookup: HF repo ID → local canonical filename. Case-insensitive.
    /// Returns `None` for unknown repos.
    pub fn local_filename_for_repo(repo_id: &str) -> Option<&'static str> {
        let eq = |s: &str| s.eq_ignore_ascii_case(repo_id);
        if eq(EMBEDDING.0) {
            Some(super::models::EMBEDDING)
        } else if eq(RERANKER.0) {
            Some(super::models::RERANKER)
        } else if eq(EXPANDER.0) {
            Some(super::models::EXPANDER)
        } else if eq(QWEN35_0_8B.0) {
            Some(super::models::QWEN35_0_8B)
        } else if eq(QWEN35_2B.0) {
            Some(super::models::QWEN35_2B)
        } else if eq(BGE_M3.0) {
            Some(super::models::BGE_M3)
        } else {
            None
        }
    }
}

/// Number of layers to offload to GPU.
/// Override with `IR_GPU_LAYERS=N`; defaults to 99 (all) on macOS.
pub fn gpu_layers() -> u32 {
    std::env::var("IR_GPU_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if cfg!(target_os = "macos") { 99 } else { 0 })
}

/// CPU-pinned model params: no GPU layers, device list restricted to CPU backends.
pub fn model_load_cpu_params() -> LlamaModelParams {
    let cpu_devices: Vec<usize> = list_llama_ggml_backend_devices()
        .into_iter()
        .filter(|d| d.device_type == LlamaBackendDeviceType::Cpu)
        .map(|d| d.index)
        .collect();

    let base = LlamaModelParams::default().with_n_gpu_layers(0);
    if cpu_devices.is_empty() {
        return base;
    }
    match base.with_devices(&cpu_devices) {
        Ok(pinned) => pinned,
        Err(_) => LlamaModelParams::default().with_n_gpu_layers(0),
    }
}

/// Build model-load params with runtime defaults.
///
/// If `IR_GPU_LAYERS=0` (or `IR_FORCE_CPU_BACKEND=1`), pin to CPU backend devices.
pub fn model_load_params() -> LlamaModelParams {
    let gpu_layers = gpu_layers();
    let base = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);

    if !should_force_cpu_backend() {
        return base;
    }

    let cpu_devices: Vec<usize> = list_llama_ggml_backend_devices()
        .into_iter()
        .filter(|d| d.device_type == LlamaBackendDeviceType::Cpu)
        .map(|d| d.index)
        .collect();

    if cpu_devices.is_empty() {
        return base;
    }

    match LlamaModelParams::default()
        .with_n_gpu_layers(gpu_layers)
        .with_devices(&cpu_devices)
    {
        Ok(pinned) => pinned,
        Err(_) => base,
    }
}

fn should_force_cpu_backend() -> bool {
    match std::env::var("IR_FORCE_CPU_BACKEND") {
        Ok(raw) => {
            let v = raw.to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => gpu_layers() == 0,
    }
}

/// Environment variables for model resolution.
pub mod env {
    pub const MODEL_DIRS: &str = "IR_MODEL_DIRS";
    pub const MODEL_DIRS_QMD: &str = "QMD_MODEL_DIRS";

    pub const EMBEDDING_MODEL: &[&str] = &[
        "IR_EMBEDDING_MODEL",
        "QMD_EMBEDDING_MODEL",
        "QMD_EMBED_MODEL",
    ];
    pub const RERANKER_MODEL: &[&str] = &[
        "IR_RERANKER_MODEL",
        "QMD_RERANKER_MODEL",
        "QMD_RERANK_MODEL",
    ];
    pub const EXPANDER_MODEL: &[&str] = &[
        "IR_EXPANDER_MODEL",
        "QMD_EXPANDER_MODEL",
        "QMD_EXPAND_MODEL",
    ];
    /// Combined model: one GGUF serving both expander + reranker roles.
    pub const COMBINED_MODEL: &str = "IR_COMBINED_MODEL";
    /// Deprecated alias for IR_COMBINED_MODEL. Still accepted; emits a warning.
    pub const QWEN_MODEL: &str = "IR_QWEN_MODEL";
}

/// Env vars that can override the full path for a known model filename.
///
/// ^ BGE_M3 must NOT appear here. `resolve_env_hf_or_path` calls `ensure_model("bge-m3-q8_0.gguf")`
/// ^ which calls `find_model` → `resolve_model_override` → this function. If BGE_M3 were registered,
/// ^ that would re-read `IR_EMBEDDING_MODEL` and call `ensure_model` again — infinite loop.
pub fn model_override_env_vars(filename: &str) -> &'static [&'static str] {
    match filename {
        models::EMBEDDING => env::EMBEDDING_MODEL,
        models::RERANKER => env::RERANKER_MODEL,
        models::EXPANDER => env::EXPANDER_MODEL,
        models::QWEN35_0_8B | models::QWEN35_2B => std::slice::from_ref(&env::QWEN_MODEL),
        _ => &[],
    }
}

/// Search for a model file across the standard locations.
pub fn find_model(filename: &str) -> Option<PathBuf> {
    if let Some(p) = resolve_model_override(filename) {
        return Some(p);
    }

    model_search_paths()
        .into_iter()
        .map(|dir| dir.join(filename))
        .find(|p| p.exists())
}

/// Model search directories in priority order.
pub fn model_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    append_env_dirs(&mut paths, env::MODEL_DIRS);
    append_env_dirs(&mut paths, env::MODEL_DIRS_QMD);

    if let Some(home) = dirs::home_dir() {
        push_unique(&mut paths, home.join("local-models"));
    }
    if let Some(cache) = dirs::cache_dir() {
        push_unique(&mut paths, cache.join("ir").join("models"));
        push_unique(&mut paths, cache.join("qmd").join("models"));
    }
    paths
}

fn resolve_model_override(filename: &str) -> Option<PathBuf> {
    for key in model_override_env_vars(filename) {
        let Some(raw) = std::env::var_os(key) else {
            continue;
        };
        let path = PathBuf::from(raw);
        if path.is_file() {
            return Some(path);
        }
        if path.is_dir() {
            let candidate = path.join(filename);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn append_env_dirs(paths: &mut Vec<PathBuf>, key: &str) {
    let Some(raw) = std::env::var_os(key) else {
        return;
    };
    for dir in std::env::split_paths(&raw) {
        if !dir.as_os_str().is_empty() {
            push_unique(paths, dir);
        }
    }
}

fn push_unique(paths: &mut Vec<PathBuf>, candidate: PathBuf) {
    if !paths.iter().any(|p| p == &candidate) {
        paths.push(candidate);
    }
}

/// Initialize the process-global llama.cpp backend. Safe to call from multiple models.
/// Returns a &'static reference; the backend lives for the lifetime of the process.
pub fn init_backend() -> crate::error::Result<&'static LlamaBackend> {
    init_llama_logs();

    // Fast path: already initialized.
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let b = LlamaBackend::init()
        .map_err(|e: llama_cpp_2::LlamaCppError| crate::error::Error::Other(e.to_string()))?;
    Ok(BACKEND.get_or_init(|| b))
}

fn init_llama_logs() {
    LOGS_INIT.get_or_init(|| {
        let logs_enabled = matches!(
            std::env::var("IR_LLAMA_LOGS")
                .unwrap_or_else(|_| "0".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        );
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default().with_logs_enabled(logs_enabled),
        );
    });
}

/// L2-normalize a float vector in place. No-op if magnitude is zero.
pub fn l2_normalize(v: &mut [f32]) {
    let mag = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 1e-10 {
        v.iter_mut().for_each(|x| *x /= mag);
    }
}

/// Serialize f32 slice to little-endian bytes for sqlite-vec.
pub fn to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_bytes(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn l2_normalize_unit_vector() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (mag - 1.0).abs() < 1e-6,
            "magnitude should be 1.0, got {mag}"
        );
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        l2_normalize(&mut v); // must not panic
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn round_trip_bytes() {
        let orig = vec![1.0f32, 2.5, -0.5, 0.0];
        let bytes = to_bytes(&orig);
        let restored = from_bytes(&bytes);
        assert_eq!(orig, restored);
    }

    #[test]
    fn find_model_returns_none_for_unknown() {
        assert!(find_model("nonexistent-model-xyz.gguf").is_none());
    }

    #[test]
    fn model_override_env_mapping_is_defined_for_known_models() {
        assert_eq!(
            model_override_env_vars(models::EMBEDDING),
            env::EMBEDDING_MODEL
        );
        assert_eq!(
            model_override_env_vars(models::RERANKER),
            env::RERANKER_MODEL
        );
        assert_eq!(
            model_override_env_vars(models::EXPANDER),
            env::EXPANDER_MODEL
        );
    }

    // ^ Regression guard for loop-prevention invariant. See comment on model_override_env_vars.
    #[test]
    fn bge_m3_not_in_model_override_env_vars() {
        assert!(
            model_override_env_vars(models::BGE_M3).is_empty(),
            "BGE_M3 must not appear in model_override_env_vars — it would cause infinite recursion"
        );
    }

    #[test]
    fn bge_m3_is_in_hf_repos() {
        assert!(
            hf_repos::for_filename(models::BGE_M3).is_some(),
            "BGE_M3 must have an HF repo entry"
        );
    }

    #[test]
    fn all_known_repos_contains_bge_m3() {
        let repos = hf_repos::all_known_repos();
        assert!(
            repos.iter().any(|(repo_id, _)| *repo_id == hf_repos::BGE_M3.0),
            "all_known_repos must include BGE_M3"
        );
    }

    #[test]
    fn local_filename_for_repo_maps_all_known() {
        for (repo_id, _) in hf_repos::all_known_repos() {
            assert!(
                hf_repos::local_filename_for_repo(repo_id).is_some(),
                "no local filename for repo: {repo_id}"
            );
        }
    }

    #[test]
    fn local_filename_for_repo_case_insensitive() {
        // HF repo IDs are case-insensitive in practice.
        assert_eq!(
            hf_repos::local_filename_for_repo("GGML-ORG/BGE-M3-Q8_0-GGUF"),
            Some(models::BGE_M3)
        );
        assert_eq!(
            hf_repos::local_filename_for_repo("ggml-org/bge-m3-Q8_0-GGUF"),
            Some(models::BGE_M3)
        );
    }

    #[test]
    fn local_filename_for_repo_returns_none_for_unknown() {
        assert_eq!(hf_repos::local_filename_for_repo("nobody/fake-model"), None);
    }
}
