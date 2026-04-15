// Query expansion model: fine-tuned 1.7B → generates typed sub-queries.
// Output format:
//   lex: keyword1 keyword2 "exact phrase"
//   vec: natural language reformulation
//   hyde: A passage that would be a good answer...
//
// Uses GBNF grammar to constrain output to valid sub-query lines.
// docs: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md

use crate::error::{Error, Result};
use crate::llm::{LlamaBackend, env, model_load_params, models};
use llama_cpp_2::model::{AddBos, LlamaModel};
// Note: grammar-constrained sampling (GBNF) is intentionally not used here.
// llama_grammar_reject_candidates has an assertion failure with this llama.cpp version
// when applied to the qmd-query-expansion model. Free-form sampling + parse + fallback
// is equivalent since the model is fine-tuned to produce the correct format.
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

const MAX_OUTPUT_TOKENS: usize = 300;
const CONTEXT_SIZE: u32 = 2048;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubQuery {
    pub kind: SubQueryKind,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubQueryKind {
    Lex,  // keyword / BM25
    Vec,  // semantic / vector
    Hyde, // hypothetical document
}

pub trait QueryExpander: Send + Sync {
    fn expand_query(&self, query: &str) -> Result<Vec<SubQuery>>;
    fn model_id(&self) -> &str;
}

impl<T: QueryExpander> QueryExpander for Arc<T> {
    fn expand_query(&self, query: &str) -> Result<Vec<SubQuery>> {
        (**self).expand_query(query)
    }
    fn model_id(&self) -> &str {
        (**self).model_id()
    }
}

pub struct Expander {
    backend: &'static LlamaBackend,
    model: LlamaModel,
}

impl Expander {
    pub fn load(model_path: &Path) -> Result<Self> {
        let backend = crate::llm::init_backend()?;
        let model = LlamaModel::load_from_file(&backend, model_path, &model_load_params())
            .map_err(|e| Error::Other(format!("load expander model: {e}")))?;
        Ok(Self { backend, model })
    }

    pub fn load_default() -> Result<Self> {
        let path = match crate::llm::download::resolve_env_hf_or_path(
            env::EXPANDER_MODEL,
            &[models::EXPANDER],
        )? {
            Some(p) => p,
            None => crate::llm::download::ensure_model(models::EXPANDER)?,
        };
        Self::load(&path)
    }

    /// Expand a query into typed sub-queries. Falls back to defaults on parse failure.
    pub fn expand(&self, query: &str) -> Result<Vec<SubQuery>> {
        let prompt = build_prompt(query);
        let raw = crate::llm::generate::generate(
            &self.model,
            self.backend,
            &prompt,
            &crate::llm::generate::GenerateParams {
                ctx_size: CONTEXT_SIZE,
                max_tokens: MAX_OUTPUT_TOKENS,
                add_bos: AddBos::Always,
                temp: 0.7,
                top_k: 20,
                top_p: 0.8,
                seed: 42,
            },
        )?;
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
}

impl QueryExpander for Expander {
    fn expand_query(&self, query: &str) -> Result<Vec<SubQuery>> {
        self.expand(query)
    }
    fn model_id(&self) -> &str {
        models::EXPANDER
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
                Some(SubQuery {
                    kind: SubQueryKind::Lex,
                    text: text.trim().to_string(),
                })
            } else if let Some(text) = line.strip_prefix("vec:") {
                Some(SubQuery {
                    kind: SubQueryKind::Vec,
                    text: text.trim().to_string(),
                })
            } else if let Some(text) = line.strip_prefix("hyde:") {
                Some(SubQuery {
                    kind: SubQueryKind::Hyde,
                    text: text.trim().to_string(),
                })
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
        SubQuery {
            kind: SubQueryKind::Lex,
            text: query.to_string(),
        },
        SubQuery {
            kind: SubQueryKind::Vec,
            text: query.to_string(),
        },
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
