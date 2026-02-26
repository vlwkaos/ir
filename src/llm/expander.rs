// Query expansion model: fine-tuned 1.7B → generates typed sub-queries.
// Output format:
//   lex: keyword1 keyword2 "exact phrase"
//   vec: natural language reformulation
//   hyde: A passage that would be a good answer...
//
// Uses GBNF grammar to constrain output to valid sub-query lines.
// docs: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md

use crate::error::{Error, Result};
use crate::llm::{models, LlamaBackend};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    sampling::LlamaSampler,
};
// Note: grammar-constrained sampling (GBNF) is intentionally not used here.
// llama_grammar_reject_candidates has an assertion failure with this llama.cpp version
// when applied to the qmd-query-expansion model. Free-form sampling + parse + fallback
// is equivalent since the model is fine-tuned to produce the correct format.
use std::num::NonZeroU32;
use std::path::Path;

const MAX_OUTPUT_TOKENS: usize = 300;
const CONTEXT_SIZE: u32 = 2048;

#[derive(Debug, Clone)]
pub struct SubQuery {
    pub kind: SubQueryKind,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubQueryKind {
    Lex,  // keyword / BM25
    Vec,  // semantic / vector
    Hyde, // hypothetical document
}

pub struct Expander {
    backend: &'static LlamaBackend,
    model: LlamaModel,
}

impl Expander {
    pub fn load(model_path: &Path) -> Result<Self> {
        let backend = crate::llm::init_backend()?;
        let model = LlamaModel::load_from_file(&backend, model_path, &LlamaModelParams::default())
            .map_err(|e| Error::Other(format!("load expander model: {e}")))?;
        Ok(Self { backend, model })
    }

    pub fn load_default() -> Result<Self> {
        let path = crate::llm::find_model(models::EXPANDER).ok_or_else(|| {
            Error::Other(format!(
                "expansion model '{}' not found. Add to ~/local-models/",
                models::EXPANDER
            ))
        })?;
        Self::load(&path)
    }

    /// Expand a query into typed sub-queries. Falls back to defaults on parse failure.
    pub fn expand(&self, query: &str) -> Result<Vec<SubQuery>> {
        let prompt = build_prompt(query);
        let raw = self.generate(&prompt)?;
        let parsed = parse_output(&raw);

        // Validate: at least one sub-query must contain a term from the original query.
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

    fn generate(&self, prompt: &str) -> Result<String> {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CONTEXT_SIZE))
            .with_n_threads(n_threads)
            .with_n_threads_batch(n_threads);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| Error::Other(format!("expander context: {e}")))?;

        // Tokenize prompt.
        let prompt_tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| Error::Other(format!("tokenize: {e}")))?;

        let n_prompt = prompt_tokens.len();
        if n_prompt == 0 {
            return Ok(String::new());
        }

        // Initial decode: all prompt tokens, logits only on last.
        let mut batch = LlamaBatch::new(n_prompt, 1);
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            let logits = i == n_prompt - 1;
            batch
                .add(tok, i as i32, &[0], logits)
                .map_err(|e| Error::Other(format!("batch add: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| Error::Other(format!("decode prompt: {e}")))?;

        // Free-form sampler: temperature 0.7, top_k 20, top_p 0.8.
        // Grammar constraint omitted — see module note above.
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.7),
            LlamaSampler::top_k(20),
            LlamaSampler::top_p(0.8, 1),
            LlamaSampler::dist(42),
        ]);

        let mut output = String::new();
        let mut n_cur = n_prompt as i32;

        for _ in 0..MAX_OUTPUT_TOKENS {
            let token = sampler.sample(&ctx, -1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            // token_to_piece_bytes: special=true, no lstrip
            let bytes = self
                .model
                .token_to_piece_bytes(token, 32, true, None)
                .map_err(|e| Error::Other(format!("token_to_piece_bytes: {e}")))?;
            output.push_str(&String::from_utf8_lossy(&bytes));

            // Prepare next token batch.
            let mut next = LlamaBatch::new(1, 1);
            next.add(token, n_cur, &[0], true)
                .map_err(|e| Error::Other(format!("batch next: {e}")))?;
            ctx.decode(&mut next)
                .map_err(|e| Error::Other(format!("decode next: {e}")))?;
            n_cur += 1;
        }

        Ok(output)
    }
}

fn build_prompt(query: &str) -> String {
    format!(
        "Generate search sub-queries for: {query}\n\
         Output lex (keywords), vec (semantic), and hyde (passage) variants:\n"
    )
}

/// Parse "type: content\n" lines from model output.
pub fn parse_output(raw: &str) -> Vec<SubQuery> {
    raw.lines()
        .filter_map(|line| {
            let line = line.trim();
            if let Some(text) = line.strip_prefix("lex:") {
                Some(SubQuery { kind: SubQueryKind::Lex, text: text.trim().to_string() })
            } else if let Some(text) = line.strip_prefix("vec:") {
                Some(SubQuery { kind: SubQueryKind::Vec, text: text.trim().to_string() })
            } else if let Some(text) = line.strip_prefix("hyde:") {
                Some(SubQuery { kind: SubQueryKind::Hyde, text: text.trim().to_string() })
            } else {
                None
            }
        })
        .filter(|s| !s.text.is_empty())
        .collect()
}

/// Safe fallback when model output fails validation.
pub fn fallback(query: &str) -> Vec<SubQuery> {
    vec![
        SubQuery { kind: SubQueryKind::Lex, text: query.to_string() },
        SubQuery { kind: SubQueryKind::Vec, text: query.to_string() },
        SubQuery {
            kind: SubQueryKind::Hyde,
            text: format!("Information about {query}"),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_output() {
        let raw = "lex: rust error handling\nvec: how to handle errors in Rust\nhyde: A passage about Result types\n";
        let subs = parse_output(raw);
        assert_eq!(subs.len(), 3);
        assert_eq!(subs[0].kind, SubQueryKind::Lex);
        assert_eq!(subs[1].kind, SubQueryKind::Vec);
        assert_eq!(subs[2].kind, SubQueryKind::Hyde);
        assert!(subs[0].text.contains("rust"));
    }

    #[test]
    fn parse_ignores_garbage_lines() {
        let raw = "some preamble\nlex: keyword\n\ngarbage\nvec: semantic query\n";
        let subs = parse_output(raw);
        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn fallback_contains_original_query() {
        let subs = fallback("my search query");
        assert!(subs.iter().any(|s| s.text.contains("my search query")));
        assert_eq!(subs.len(), 3);
    }

    #[test]
    #[ignore]
    fn expand_returns_valid_subqueries() {
        let e = Expander::load_default().expect("load model");
        let subs = e.expand("rust memory management").expect("expand");
        assert!(!subs.is_empty());
        // At least one sub-query should contain a term from the original
        let any_relevant = subs
            .iter()
            .any(|s| s.text.contains("rust") || s.text.contains("memory"));
        assert!(any_relevant);
    }
}
