// LLM model manager: backend init, model path resolution, L2 normalization.
// docs: https://docs.rs/llama-cpp-2/latest/llama_cpp_2/
//
// Model search order:
//   1. ~/local-models/        (central, managed by /local-model skill)
//   2. ~/.cache/ir/models/    (ir-specific cache)
//   3. ~/.cache/qmd/models/   (reuse qmd downloads)

pub mod embedding;
pub mod expander;
pub mod reranker;

pub use llama_cpp_2::llama_backend::LlamaBackend;

use std::path::PathBuf;
use std::sync::OnceLock;

// LlamaBackend is an empty struct — Send + Sync by default. Store once per process.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// Known model filenames.
pub mod models {
    pub const EMBEDDING: &str = "embeddinggemma-300M-Q8_0.gguf";
    pub const RERANKER: &str = "qwen3-reranker-0.6b-q8_0.gguf";
    pub const EXPANDER: &str = "qmd-query-expansion-1.7B-q4_k_m.gguf";
}

/// Search for a model file across the standard locations.
pub fn find_model(filename: &str) -> Option<PathBuf> {
    model_search_paths()
        .into_iter()
        .map(|dir| dir.join(filename))
        .find(|p| p.exists())
}

/// Model search directories in priority order.
pub fn model_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(home) = dirs::home_dir() {
        paths.push(home.join("local-models"));
    }
    if let Some(cache) = dirs::cache_dir() {
        paths.push(cache.join("ir").join("models"));
        paths.push(cache.join("qmd").join("models"));
    }
    paths
}

/// Initialize the process-global llama.cpp backend. Safe to call from multiple models.
/// Returns a &'static reference; the backend lives for the lifetime of the process.
pub fn init_backend() -> crate::error::Result<&'static LlamaBackend> {
    // Fast path: already initialized.
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let b = LlamaBackend::init().map_err(|e: llama_cpp_2::LlamaCppError| {
        crate::error::Error::Other(e.to_string())
    })?;
    Ok(BACKEND.get_or_init(|| b))
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

/// Deserialize little-endian bytes to f32 vec.
#[allow(dead_code)]
pub fn from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_unit_vector() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 1e-6, "magnitude should be 1.0, got {mag}");
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
}
