// Embedding pipeline supporting EmbeddingGemma and BGE-M3 GGUF variants.
// EmbeddingGemma asymmetric format:
//   query: "task: search result | query: {text}"
//   doc:   "title: {title} | text: {text}"
// BGE-M3 dense mode:
//   query: "Represent this sentence: {text}"
//   doc:   raw text (no title/text prefix)
//   pooling: cls

use crate::error::{Error, Result};
use crate::llm::{LlamaBackend, l2_normalize, model_load_params, models};
use llama_cpp_2::{
    context::LlamaContext,
    context::params::{LlamaContextParams, LlamaPoolingType},
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel},
};
use std::num::NonZeroU32;
use std::path::Path;
use std::str::FromStr;

const CONTEXT_SIZE: u32 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbeddingProfile {
    EmbeddingGemma,
    BgeM3,
    Generic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingPooling {
    None,
    Mean,
    Cls,
    Last,
    Rank,
}

impl EmbeddingPooling {
    fn to_llama(self) -> LlamaPoolingType {
        match self {
            Self::None => LlamaPoolingType::None,
            Self::Mean => LlamaPoolingType::Mean,
            Self::Cls => LlamaPoolingType::Cls,
            Self::Last => LlamaPoolingType::Last,
            Self::Rank => LlamaPoolingType::Rank,
        }
    }
}

impl FromStr for EmbeddingPooling {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "mean" => Ok(Self::Mean),
            "cls" => Ok(Self::Cls),
            "last" => Ok(Self::Last),
            "rank" => Ok(Self::Rank),
            other => Err(format!(
                "unknown pooling '{other}' (use: none, mean, cls, last, rank)"
            )),
        }
    }
}

pub struct Embedder {
    backend: &'static LlamaBackend,
    model: LlamaModel,
    profile: EmbeddingProfile,
    pooling_override: Option<EmbeddingPooling>,
}

impl Embedder {
    pub fn load(model_path: &Path) -> Result<Self> {
        let backend = crate::llm::init_backend()?;
        let model = LlamaModel::load_from_file(&backend, model_path, &model_load_params())
            .map_err(|e| Error::Other(format!("load embedding model: {e}")))?;
        let profile = profile_for_model_path(model_path);
        Ok(Self {
            backend,
            model,
            profile,
            pooling_override: None,
        })
    }

    pub fn load_default() -> Result<Self> {
        let path = crate::llm::download::ensure_model(models::EMBEDDING)?;
        Self::load(&path)
    }

    pub fn embedding_dim(&self) -> usize {
        usize::try_from(self.model.n_embd()).unwrap_or(0)
    }

    pub fn set_pooling_override(&mut self, pooling: Option<EmbeddingPooling>) {
        self.pooling_override = pooling;
    }

    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.embed_single(&self.format_query(query))
    }

    pub fn embed_query_batch(&self, queries: &[String]) -> Result<Vec<Vec<f32>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        let mut ctx = self.new_context(n_threads)?;

        queries
            .iter()
            .map(|q| {
                let formatted = self.format_query(q);
                self.embed_with_context(&mut ctx, &formatted)
            })
            .collect()
    }

    pub fn embed_doc(&self, title: &str, text: &str) -> Result<Vec<f32>> {
        self.embed_single(&self.format_doc(title, text))
    }

    pub fn embed_doc_batch(&self, chunks: &[(String, String)]) -> Result<Vec<Vec<f32>>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        let mut ctx = self.new_context(n_threads)?;

        chunks
            .iter()
            .map(|(title, text)| {
                let formatted = self.format_doc(title, text);
                self.embed_with_context(&mut ctx, &formatted)
            })
            .collect()
    }

    fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        let mut ctx = self.new_context(n_threads)?;
        self.embed_with_context(&mut ctx, text)
    }

    fn new_context(&self, n_threads: i32) -> Result<LlamaContext<'_>> {
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CONTEXT_SIZE))
            // ! encoder requires n_ubatch >= n_tokens; set equal to ctx so chunks never exceed it
            .with_n_batch(CONTEXT_SIZE)
            .with_n_ubatch(CONTEXT_SIZE)
            // Keep embedding path CPU-safe in headless/CI environments.
            .with_offload_kqv(false)
            .with_n_threads_batch(n_threads)
            .with_embeddings(true);

        if let Some(pooling) = self.pooling_override {
            ctx_params = ctx_params.with_pooling_type(pooling.to_llama());
        } else if let Some(pooling) = default_pooling_for_profile(self.profile) {
            ctx_params = ctx_params.with_pooling_type(pooling.to_llama());
        }

        self.model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| Error::Other(format!("embedding context: {e}")))
    }

    fn embed_with_context(&self, ctx: &mut LlamaContext<'_>, text: &str) -> Result<Vec<f32>> {
        ctx.clear_kv_cache();

        let tokens = self
            .model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| Error::Other(format!("tokenize: {e}")))?;

        let dim = self.embedding_dim();
        if tokens.is_empty() {
            return Ok(vec![0.0f32; dim]);
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
        if dim > 0 && emb.len() != dim {
            emb.resize(dim, 0.0);
        }
        Ok(emb)
    }

    fn format_query(&self, query: &str) -> String {
        match self.profile {
            EmbeddingProfile::EmbeddingGemma => format_query(query),
            EmbeddingProfile::BgeM3 => bge_format_query(query),
            EmbeddingProfile::Generic => query.to_string(),
        }
    }

    fn format_doc(&self, title: &str, text: &str) -> String {
        match self.profile {
            EmbeddingProfile::EmbeddingGemma => format_doc(title, text),
            EmbeddingProfile::BgeM3 => bge_format_doc(title, text),
            EmbeddingProfile::Generic => {
                if title.is_empty() {
                    text.to_string()
                } else {
                    format!("{title}\n\n{text}")
                }
            }
        }
    }
}

pub fn format_query(query: &str) -> String {
    format!("task: search result | query: {query}")
}

pub fn format_doc(title: &str, text: &str) -> String {
    format!("title: {title} | text: {text}")
}

fn bge_format_query(query: &str) -> String {
    format!("Represent this sentence: {query}")
}

fn bge_format_doc(_title: &str, text: &str) -> String {
    text.to_string()
}

fn profile_for_model_path(model_path: &Path) -> EmbeddingProfile {
    let name = model_path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if name.contains("embeddinggemma") {
        EmbeddingProfile::EmbeddingGemma
    } else if name.contains("bge-m3") {
        EmbeddingProfile::BgeM3
    } else {
        EmbeddingProfile::Generic
    }
}

fn default_pooling_for_profile(profile: EmbeddingProfile) -> Option<EmbeddingPooling> {
    match profile {
        EmbeddingProfile::EmbeddingGemma => Some(EmbeddingPooling::Last),
        EmbeddingProfile::BgeM3 => Some(EmbeddingPooling::Cls),
        EmbeddingProfile::Generic => None,
    }
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
        assert!(emb.len() > 0);
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
