// Combined model: one GGUF load serving both query expansion and reranking.
// Activated via IR_COMBINED_MODEL (preferred) or IR_QWEN_MODEL (deprecated).
// Currently tuned for instruction-following models with ChatML format (Qwen3.5).
// docs: https://docs.rs/llama-cpp-2/latest/llama_cpp_2/
//
// Reranker role:  Yes/No logit scoring (same protocol as reranker.rs)
// Expander role:  autoregressive generation → lex:/vec:/hyde: lines
//
// Prompts are DSPy-MIPROv2 optimized (see research/dspy_optimize.py).
// Run research/dspy_optimize.py to re-optimize; hardcode results here.

use crate::error::{Error, Result};
use crate::llm::{LlamaBackend, model_load_params, models};
use crate::llm::expander::{QueryExpander, SubQuery, fallback, parse_output};
use crate::llm::generate::{self, GenerateParams};
use crate::llm::scoring::{self, Scorer};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::{AddBos, LlamaModel};
use std::path::Path;
use std::sync::Mutex;

const RERANK_CONTEXT_SIZE: u32 = 2048;
const EXPAND_CONTEXT_SIZE: u32 = 2048;
const MAX_EXPAND_TOKENS: usize = 300;

pub struct Combined {
    backend: &'static LlamaBackend,
    model: LlamaModel,
    model_filename: String,
    yes_token_id: i32,
    no_token_id: i32,
    // ! Cached rerank context: model outlives context (same struct)
    cached_rerank_ctx: Mutex<Option<LlamaContext<'static>>>,
}

// ! Safety: LlamaModel is Send+Sync, LlamaContext access is serialized by Mutex
unsafe impl Send for Combined {}
unsafe impl Sync for Combined {}

impl Combined {
    pub fn load(path: &Path) -> Result<Self> {
        let backend = crate::llm::init_backend()?;
        let model = LlamaModel::load_from_file(backend, path, &model_load_params())
            .map_err(|e| Error::Other(format!("load combined model: {e}")))?;
        let model_filename = path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        let (yes_token_id, no_token_id) = scoring::resolve_yes_no_tokens(&model)?;
        Ok(Self {
            backend,
            model,
            model_filename,
            yes_token_id,
            no_token_id,
            cached_rerank_ctx: Mutex::new(None),
        })
    }

    pub fn name(&self) -> &str {
        &self.model_filename
    }

    /// Resolve the combined model path from env vars, or search local model dirs.
    /// Returns `Ok(None)` when no env var is set and no known model file is found locally.
    pub fn try_load_default() -> Result<Option<Self>> {
        use crate::llm::env;

        // Priority: IR_COMBINED_MODEL > IR_QWEN_MODEL (deprecated).
        let env_key: &[&str] = if std::env::var_os(env::COMBINED_MODEL).is_some() {
            &[env::COMBINED_MODEL]
        } else if std::env::var_os(env::QWEN_MODEL).is_some() {
            eprintln!("ir: IR_QWEN_MODEL is deprecated — use IR_COMBINED_MODEL instead");
            &[env::QWEN_MODEL]
        } else {
            &[]
        };

        if env_key.is_empty() {
            // No env var — check whether a known file exists locally (no download).
            for filename in &[models::QWEN35_2B, models::QWEN35_0_8B] {
                if let Some(path) = crate::llm::find_model(filename) {
                    return Ok(Some(Self::load(&path)?));
                }
            }
            return Ok(None);
        }

        match crate::llm::download::resolve_env_hf_or_path(
            env_key,
            &[models::QWEN35_2B, models::QWEN35_0_8B],
        )? {
            Some(p) => Ok(Some(Self::load(&p)?)),
            None => Ok(None),
        }
    }

    /// Expand a query into typed sub-queries (lex/vec/hyde). Falls back on parse failure.
    ///
    /// // ! DSPy-optimized prompt — do not edit manually; re-run research/dspy_optimize.py
    pub fn expand(&self, query: &str) -> Result<Vec<SubQuery>> {
        let prompt = build_expand_prompt(query);
        let raw = generate::generate(
            &self.model,
            self.backend,
            &prompt,
            &GenerateParams {
                ctx_size: EXPAND_CONTEXT_SIZE,
                max_tokens: MAX_EXPAND_TOKENS,
                add_bos: AddBos::Never, // ! ChatML prompt; no extra BOS
                temp: 0.7,
                top_k: 20,
                top_p: 0.8,
                seed: 42,
            },
        )?;
        let parsed = parse_output(&raw);

        let query_lower = query.to_lowercase();
        let valid = parsed.iter().any(|s| {
            s.text
                .split_whitespace()
                .any(|w| query_lower.contains(&w.to_lowercase()))
        });

        if parsed.is_empty() || !valid {
            Ok(fallback(query))
        } else {
            Ok(parsed)
        }
    }

    fn get_or_create_rerank_ctx(&self) -> Result<std::sync::MutexGuard<'_, Option<LlamaContext<'static>>>> {
        let mut guard = self.cached_rerank_ctx.lock().unwrap();
        if guard.is_none() {
            let ctx = scoring::create_scoring_context(&self.model, self.backend, RERANK_CONTEXT_SIZE)?;
            // ! Safety: model lives in same struct; context is dropped first via Drop impl
            let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
            *guard = Some(ctx);
        }
        Ok(guard)
    }
}

impl Drop for Combined {
    fn drop(&mut self) {
        // ! Drop context before model
        let _ = self.cached_rerank_ctx.lock().map(|mut g| g.take());
    }
}

impl Scorer for Combined {
    fn model_id(&self) -> &str {
        &self.model_filename
    }

    /// // ! DSPy-optimized prompt — do not edit manually; re-run research/dspy_optimize.py
    fn score_batch(&self, query: &str, docs: &[&str]) -> Result<Vec<f64>> {
        let mut guard = self.get_or_create_rerank_ctx()?;
        let ctx = guard.as_mut().unwrap();
        scoring::score_batch_with_ctx(
            ctx,
            &self.model,
            self.yes_token_id,
            self.no_token_id,
            query,
            docs,
            RERANK_CONTEXT_SIZE,
        )
    }
}

impl QueryExpander for Combined {
    fn expand_query(&self, query: &str) -> Result<Vec<SubQuery>> {
        self.expand(query)
    }
    fn model_id(&self) -> &str {
        &self.model_filename
    }
}

/// // ! DSPy-optimized prompt — do not edit manually; re-run research/dspy_optimize.py
fn build_expand_prompt(query: &str) -> String {
    format!(
        "<|im_start|>system\n\
         Generate search sub-queries for document retrieval. \
         Output exactly three lines: lex (2-5 keywords for BM25), \
         vec (natural language reformulation), hyde (1-2 sentence hypothetical answer passage).<|im_end|>\n\
         <|im_start|>user\n\
         Query: {query}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn load_0_8b_and_tokenize() {
        let path = dirs::home_dir()
            .unwrap()
            .join("local-models")
            .join(models::QWEN35_0_8B);
        assert!(path.exists(), "model not found: {}", path.display());

        let q = Combined::load(&path).expect("load 0.8B");

        let tokens = q
            .model
            .str_to_token("hello world", AddBos::Never)
            .expect("tokenize");
        assert!(!tokens.is_empty(), "tokenization returned empty");

        let result = q.expand("hello");
        assert!(result.is_ok(), "expand failed: {:?}", result.err());
        println!("0.8B output: {:?}", result.unwrap());
    }

    #[test]
    #[ignore]
    fn load_2b_and_tokenize() {
        let path = dirs::home_dir()
            .unwrap()
            .join("local-models")
            .join(models::QWEN35_2B);
        assert!(path.exists(), "model not found: {}", path.display());

        let q = Combined::load(&path).expect("load 2B");
        let tokens = q
            .model
            .str_to_token("hello world", AddBos::Never)
            .expect("tokenize");
        assert!(!tokens.is_empty());
        println!("2B vocab size: {}", q.model.n_vocab());
    }

    #[test]
    #[ignore]
    fn expand_returns_valid_subqueries() {
        use crate::llm::expander::SubQueryKind;
        let q = Combined::try_load_default().expect("load model").unwrap();
        let subs = q.expand("rust memory management").expect("expand");
        assert!(!subs.is_empty());
        let any_relevant = subs
            .iter()
            .any(|s| s.text.contains("rust") || s.text.contains("memory"));
        assert!(any_relevant, "no relevant sub-query in: {subs:?}");

        let kinds: Vec<SubQueryKind> = subs.iter().map(|s| s.kind).collect();
        assert!(kinds.contains(&SubQueryKind::Lex));
        assert!(kinds.contains(&SubQueryKind::Vec));
    }

    #[test]
    #[ignore]
    fn score_batch_orders_correctly() {
        use crate::llm::scoring::Scorer;
        let q = Combined::try_load_default().expect("load model").unwrap();
        let scores = q
            .score_batch(
                "rust memory management",
                &[
                    "Rust uses ownership and borrowing to manage memory without a garbage collector",
                    "Python uses a garbage collector. JavaScript also has automatic memory management.",
                ],
            )
            .expect("score_batch");
        assert_eq!(scores.len(), 2);
        assert!(
            scores[0] > scores[1],
            "relevant={:.3} should > irrelevant={:.3}",
            scores[0],
            scores[1]
        );
    }
}
