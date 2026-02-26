// EmbeddingGemma 300M embedding pipeline.
// Model: hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf
// Asymmetric format (required by this model):
//   query: "task: search result | query: {text}"
//   doc:   "title: {title} | text: {text}"

use crate::error::{Error, Result};
use crate::llm::{l2_normalize, models, LlamaBackend};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
};
use std::num::NonZeroU32;
use std::path::Path;

pub const EMBEDDING_DIM: usize = 768;
const CONTEXT_SIZE: u32 = 1024;

pub struct Embedder {
    backend: &'static LlamaBackend,
    model: LlamaModel,
}

impl Embedder {
    pub fn load(model_path: &Path) -> Result<Self> {
        let backend = crate::llm::init_backend()?;
        let model = LlamaModel::load_from_file(&backend, model_path, &LlamaModelParams::default())
            .map_err(|e| Error::Other(format!("load embedding model: {e}")))?;
        Ok(Self { backend, model })
    }

    pub fn load_default() -> Result<Self> {
        let path = crate::llm::find_model(models::EMBEDDING).ok_or_else(|| {
            Error::Other(format!(
                "embedding model '{}' not found.\nAdd to ~/local-models/ or run: ir embed --download",
                models::EMBEDDING
            ))
        })?;
        Self::load(&path)
    }

    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.embed_single(&format_query(query))
    }

    pub fn embed_doc(&self, title: &str, text: &str) -> Result<Vec<f32>> {
        self.embed_single(&format_doc(title, text))
    }

    pub fn embed_doc_batch(&self, chunks: &[(String, String)]) -> Result<Vec<Vec<f32>>> {
        chunks
            .iter()
            .map(|(title, text)| self.embed_doc(title, text))
            .collect()
    }

    fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CONTEXT_SIZE))
            // ! encoder requires n_ubatch >= n_tokens; set equal to ctx so chunks never exceed it
            .with_n_batch(CONTEXT_SIZE)
            .with_n_ubatch(CONTEXT_SIZE)
            .with_n_threads_batch(n_threads)
            .with_embeddings(true);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| Error::Other(format!("embedding context: {e}")))?;

        let tokens = self
            .model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| Error::Other(format!("tokenize: {e}")))?;

        if tokens.is_empty() {
            return Ok(vec![0.0f32; EMBEDDING_DIM]);
        }

        let n = tokens.len().min(CONTEXT_SIZE as usize);
        let mut batch = LlamaBatch::new(n, 1);
        batch
            .add_sequence(&tokens[..n], 0, false)
            .map_err(|e| Error::Other(format!("batch: {e}")))?;

        ctx.decode(&mut batch)
            .map_err(|e| Error::Other(format!("decode: {e}")))?;

        let raw = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| Error::Other(format!("embeddings: {e}")))?;

        let mut emb: Vec<f32> = raw.to_vec();
        l2_normalize(&mut emb);
        emb.resize(EMBEDDING_DIM, 0.0);
        Ok(emb)
    }
}

pub fn format_query(query: &str) -> String {
    format!("task: search result | query: {query}")
}

pub fn format_doc(title: &str, text: &str) -> String {
    format!("title: {title} | text: {text}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_query_prefix() {
        let f = format_query("how does auth work");
        assert!(f.starts_with("task: search result | query:"));
        assert!(f.contains("how does auth work"));
    }

    #[test]
    fn format_doc_prefix() {
        let f = format_doc("My Doc", "some content here");
        assert!(f.starts_with("title: My Doc | text:"));
        assert!(f.contains("some content here"));
    }

    #[test]
    #[ignore]
    fn embed_query_returns_unit_vector() {
        let e = Embedder::load_default().expect("load model");
        let emb = e.embed_query("test query").expect("embed");
        assert_eq!(emb.len(), EMBEDDING_DIM);
        let mag: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 1e-5, "not unit vector: mag={mag}");
    }

    #[test]
    #[ignore]
    fn embed_doc_and_query_similar_for_related_content() {
        let e = Embedder::load_default().expect("load model");
        let q = e.embed_query("rust error handling").expect("embed");
        let d = e
            .embed_doc("Error Handling", "Use Result and thiserror for Rust errors")
            .expect("embed");
        let dot: f32 = q.iter().zip(&d).map(|(a, b)| a * b).sum();
        assert!(dot > 0.3, "expected similarity > 0.3, got {dot}");
    }
}
