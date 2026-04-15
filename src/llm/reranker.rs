// Qwen3-Reranker 0.6B cross-encoder scoring.
// Format: ChatML with system instruction + <Instruct>/<Query>/<Document> tags.
// Scores by softmax of logits for "Yes" vs "No" tokens at the last position.
//
// Cache key: sha256(model_id + "\0" + query + "\0" + doc_hash) → cached f64 score

use crate::error::{Error, Result};
use crate::llm::{LlamaBackend, env, model_load_params, models};
use crate::llm::scoring::{self, Scorer};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::LlamaModel;
use std::path::Path;
use std::sync::Mutex;

const CONTEXT_SIZE: u32 = 2048;

pub struct Reranker {
    backend: &'static LlamaBackend,
    model: LlamaModel,
    yes_token_id: i32,
    no_token_id: i32,
    // ! Cached context: model outlives context (same struct, drop order: fields drop in declaration order,
    // ! but we use ManuallyDrop-equivalent via Option to ensure context is dropped before model)
    cached_ctx: Mutex<Option<LlamaContext<'static>>>,
}

// ! Safety: LlamaModel is Send+Sync, LlamaContext access is serialized by Mutex
unsafe impl Send for Reranker {}
unsafe impl Sync for Reranker {}

impl Reranker {
    pub fn load(model_path: &Path) -> Result<Self> {
        let backend = crate::llm::init_backend()?;
        let model = LlamaModel::load_from_file(&backend, model_path, &model_load_params())
            .map_err(|e| Error::Other(format!("load reranker: {e}")))?;
        let (yes_token_id, no_token_id) = scoring::resolve_yes_no_tokens(&model)?;
        Ok(Self {
            backend,
            model,
            yes_token_id,
            no_token_id,
            cached_ctx: Mutex::new(None),
        })
    }

    pub fn load_default() -> Result<Self> {
        let path = match crate::llm::download::resolve_env_hf_or_path(
            env::RERANKER_MODEL,
            &[models::RERANKER],
        )? {
            Some(p) => p,
            None => crate::llm::download::ensure_model(models::RERANKER)?,
        };
        Self::load(&path)
    }

    /// Score relevance of a document to a query. Returns [0, 1].
    /// For scoring multiple documents, prefer `Scorer::score_batch` to avoid per-call context creation.
    #[allow(dead_code)]
    pub fn score(&self, query: &str, doc: &str) -> Result<f64> {
        self.score_batch(query, &[doc])
            .map(|v| v.into_iter().next().unwrap_or(0.0))
    }

    fn get_or_create_ctx(&self) -> Result<std::sync::MutexGuard<'_, Option<LlamaContext<'static>>>> {
        let mut guard = self.cached_ctx.lock().unwrap();
        if guard.is_none() {
            let ctx = scoring::create_scoring_context(&self.model, self.backend, CONTEXT_SIZE)?;
            // ! Safety: model lives in same struct; context is dropped first (Option::take in Drop or field order).
            // Erase lifetime to store in struct alongside its model.
            let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
            *guard = Some(ctx);
        }
        Ok(guard)
    }
}

impl Drop for Reranker {
    fn drop(&mut self) {
        // ! Drop context before model
        let _ = self.cached_ctx.lock().map(|mut g| g.take());
    }
}

impl Scorer for Reranker {
    fn model_id(&self) -> &str {
        models::RERANKER
    }

    fn score_batch(&self, query: &str, docs: &[&str]) -> Result<Vec<f64>> {
        let mut guard = self.get_or_create_ctx()?;
        let ctx = guard.as_mut().unwrap();
        scoring::score_batch_with_ctx(
            ctx,
            &self.model,
            self.yes_token_id,
            self.no_token_id,
            query,
            docs,
            CONTEXT_SIZE,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_key(model: &str, query: &str, doc_hash: &str) -> String {
        crate::index::hasher::hash_bytes(format!("{model}\0{query}\0{doc_hash}").as_bytes())
    }

    #[test]
    fn cache_key_deterministic() {
        let k1 = cache_key("model", "query", "hash123");
        let k2 = cache_key("model", "query", "hash123");
        assert_eq!(k1, k2);
    }

    #[test]
    fn cache_key_differs_on_different_inputs() {
        let k1 = cache_key("model", "query1", "hash");
        let k2 = cache_key("model", "query2", "hash");
        assert_ne!(k1, k2);
        let k3 = cache_key("model_a", "query", "hash");
        let k4 = cache_key("model_b", "query", "hash");
        assert_ne!(k3, k4);
    }

    #[test]
    #[ignore]
    fn score_relevant_doc_higher() {
        let r = Reranker::load_default().expect("load reranker");
        let relevant = r
            .score(
                "rust memory management",
                "Rust uses ownership and borrowing to manage memory without a garbage collector",
            )
            .expect("score");
        let irrelevant = r
            .score(
                "rust memory management",
                "Python uses a garbage collector. JavaScript also has automatic memory management.",
            )
            .expect("score");
        assert!(
            relevant > irrelevant,
            "relevant={relevant:.3} should > irrelevant={irrelevant:.3}"
        );
    }
}
