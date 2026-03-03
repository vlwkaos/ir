// Qwen3-Reranker 0.6B cross-encoder scoring.
// Format: ChatML with system instruction + <Instruct>/<Query>/<Document> tags.
// Scores by softmax of logits for "Yes" vs "No" tokens at the last position.
//
// Cache key: sha256(query + "\0" + doc_hash) → cached f64 score

use crate::error::{Error, Result};
use crate::llm::{LlamaBackend, model_load_params, models};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel},
};
use std::num::NonZeroU32;
use std::path::Path;

const CONTEXT_SIZE: u32 = 2048;
/// Reserve tokens for the query + framing
const MAX_DOC_CHARS: usize = 6000;

pub struct Reranker {
    backend: &'static LlamaBackend,
    model: LlamaModel,
    yes_token_id: i32,
    no_token_id: i32,
}

impl Reranker {
    pub fn load(model_path: &Path) -> Result<Self> {
        let backend = crate::llm::init_backend()?;
        let model = LlamaModel::load_from_file(&backend, model_path, &model_load_params())
            .map_err(|e| Error::Other(format!("load reranker: {e}")))?;

        // Resolve "Yes" and "No" token IDs from the model vocabulary.
        let yes_tokens = model
            .str_to_token("Yes", AddBos::Never)
            .map_err(|e| Error::Other(format!("tokenize 'Yes': {e}")))?;
        let no_tokens = model
            .str_to_token("No", AddBos::Never)
            .map_err(|e| Error::Other(format!("tokenize 'No': {e}")))?;

        // Use last token of each (handles BPE subword splits)
        let yes_id = yes_tokens.last().map(|t| t.0).unwrap_or(0);
        let no_id = no_tokens.last().map(|t| t.0).unwrap_or(1);

        Ok(Self {
            backend,
            model,
            yes_token_id: yes_id,
            no_token_id: no_id,
        })
    }

    pub fn load_default() -> Result<Self> {
        let path = crate::llm::download::ensure_model(models::RERANKER)?;
        Self::load(&path)
    }

    /// Score relevance of a document to a query. Returns [0, 1].
    pub fn score(&self, query: &str, doc: &str) -> Result<f64> {
        let doc_truncated = if doc.len() > MAX_DOC_CHARS {
            &doc[..doc.floor_char_boundary(MAX_DOC_CHARS)]
        } else {
            doc
        };

        let prompt = format!(
            "<|im_start|>system\n\
             Judge whether the Document meets the requirements based on the Query and the Instruct provided. \
             Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n\
             <|im_start|>user\n\
             <Instruct>: Given a web search query, retrieve relevant passages that answer the query\n\
             <Query>: {query}\n\
             <Document>: {doc_truncated}<|im_end|>\n\
             <|im_start|>assistant\n\
             <think>\n\
             </think>\n"
        );

        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CONTEXT_SIZE))
            .with_offload_kqv(false)
            .with_n_threads(n_threads)
            .with_n_threads_batch(n_threads);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| Error::Other(format!("reranker context: {e}")))?;

        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Never) // ! ChatML starts with <|im_start|>; extra BOS confuses model
            .map_err(|e| Error::Other(format!("tokenize: {e}")))?;

        if tokens.is_empty() {
            return Ok(0.0);
        }

        let n = tokens.len().min(CONTEXT_SIZE as usize - 1);
        let mut batch = LlamaBatch::new(n, 1);
        for (i, &tok) in tokens[..n].iter().enumerate() {
            let logits = i == n - 1; // only last token needs logits
            batch
                .add(tok, i as i32, &[0], logits)
                .map_err(|e| Error::Other(format!("batch add: {e}")))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| Error::Other(format!("decode: {e}")))?;

        let logits = ctx.get_logits_ith((n - 1) as i32);
        let yes_idx = self.yes_token_id as usize;
        let no_idx = self.no_token_id as usize;
        if yes_idx >= logits.len() || no_idx >= logits.len() {
            return Err(Error::Other(format!(
                "token id out of range: yes={yes_idx}, no={no_idx}, vocab={}",
                logits.len()
            )));
        }
        let yes_logit = logits[yes_idx];
        let no_logit = logits[no_idx];

        // Softmax over just Yes/No to get P(Yes)
        let max_logit = yes_logit.max(no_logit);
        let yes_exp = (yes_logit - max_logit).exp() as f64;
        let no_exp = (no_logit - max_logit).exp() as f64;
        let score = yes_exp / (yes_exp + no_exp);

        Ok(score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_key(query: &str, doc_hash: &str) -> String {
        crate::index::hasher::hash_bytes(format!("{query}\0{doc_hash}").as_bytes())
    }

    #[test]
    fn cache_key_deterministic() {
        let k1 = cache_key("query", "hash123");
        let k2 = cache_key("query", "hash123");
        assert_eq!(k1, k2);
    }

    #[test]
    fn cache_key_differs_on_different_inputs() {
        let k1 = cache_key("query1", "hash");
        let k2 = cache_key("query2", "hash");
        assert_ne!(k1, k2);
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
