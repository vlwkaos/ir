// Shared yes/no logit scoring: ChatML prompt → tokenize → decode → softmax.
// Used by Reranker and Combined (identical protocol).
// docs: https://docs.rs/llama-cpp-2/latest/llama_cpp_2/

use crate::error::{Error, Result};
use crate::llm::LlamaBackend;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel},
};
use std::num::NonZeroU32;
use std::sync::Arc;

const MAX_DOC_CHARS: usize = 6000;

pub trait Scorer: Send + Sync {
    fn score_batch(&self, query: &str, docs: &[&str]) -> Result<Vec<f64>>;
    fn model_id(&self) -> &str;
}

impl<T: Scorer> Scorer for Arc<T> {
    fn score_batch(&self, query: &str, docs: &[&str]) -> Result<Vec<f64>> {
        (**self).score_batch(query, docs)
    }
    fn model_id(&self) -> &str {
        (**self).model_id()
    }
}

/// Resolve yes/no token IDs from model vocabulary. Uses last subword token of each.
pub fn resolve_yes_no_tokens(model: &LlamaModel) -> Result<(i32, i32)> {
    let yes_tokens = model
        .str_to_token("Yes", AddBos::Never)
        .map_err(|e| Error::Other(format!("tokenize 'Yes': {e}")))?;
    let no_tokens = model
        .str_to_token("No", AddBos::Never)
        .map_err(|e| Error::Other(format!("tokenize 'No': {e}")))?;
    let yes_id = yes_tokens.last().map(|t| t.0).unwrap_or(0);
    let no_id = no_tokens.last().map(|t| t.0).unwrap_or(1);
    Ok((yes_id, no_id))
}

/// Score a single (query, doc) pair using an existing context.
/// Caller is responsible for clearing KV cache between calls.
pub fn score_yes_no(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    model: &LlamaModel,
    yes_id: i32,
    no_id: i32,
    query: &str,
    doc: &str,
    ctx_size: u32,
) -> Result<f64> {
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

    let tokens = model
        .str_to_token(&prompt, AddBos::Never) // ! ChatML starts with <|im_start|>; extra BOS confuses model
        .map_err(|e| Error::Other(format!("tokenize: {e}")))?;

    if tokens.is_empty() {
        return Ok(0.0);
    }

    let n = tokens.len().min(ctx_size as usize - 1);
    let mut batch = LlamaBatch::new(n, 1);
    for (i, &tok) in tokens[..n].iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], i == n - 1)
            .map_err(|e| Error::Other(format!("batch add: {e}")))?;
    }

    ctx.decode(&mut batch)
        .map_err(|e| Error::Other(format!("decode: {e}")))?;

    let logits = ctx.get_logits_ith((n - 1) as i32);
    let yes_idx = yes_id as usize;
    let no_idx = no_id as usize;
    if yes_idx >= logits.len() || no_idx >= logits.len() {
        return Err(Error::Other(format!(
            "token id out of range: yes={yes_idx}, no={no_idx}, vocab={}",
            logits.len()
        )));
    }

    let yes_logit = logits[yes_idx];
    let no_logit = logits[no_idx];
    let max_logit = yes_logit.max(no_logit);
    let yes_exp = (yes_logit - max_logit).exp() as f64;
    let no_exp = (no_logit - max_logit).exp() as f64;
    Ok(yes_exp / (yes_exp + no_exp))
}

/// Score all docs against query using a new context (convenience wrapper).
#[allow(dead_code)]
pub fn score_batch_yes_no(
    model: &LlamaModel,
    backend: &'static LlamaBackend,
    yes_id: i32,
    no_id: i32,
    query: &str,
    docs: &[&str],
    ctx_size: u32,
) -> Result<Vec<f64>> {
    let mut ctx = create_scoring_context(model, backend, ctx_size)?;
    score_batch_with_ctx(&mut ctx, model, yes_id, no_id, query, docs, ctx_size)
}

/// Score all docs against query using a caller-provided context (avoids context creation overhead).
/// Uses prefix caching: shared prompt prefix (system+instruct+query) is decoded once,
/// then only doc-specific suffix is decoded per document.
pub fn score_batch_with_ctx(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    model: &LlamaModel,
    yes_id: i32,
    no_id: i32,
    query: &str,
    docs: &[&str],
    ctx_size: u32,
) -> Result<Vec<f64>> {
    if docs.is_empty() {
        return Ok(vec![]);
    }

    // Build and decode the shared prefix once
    let prefix = format!(
        "<|im_start|>system\n\
         Judge whether the Document meets the requirements based on the Query and the Instruct provided. \
         Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n\
         <|im_start|>user\n\
         <Instruct>: Given a web search query, retrieve relevant passages that answer the query\n\
         <Query>: {query}\n\
         <Document>: "
    );

    let prefix_tokens = model
        .str_to_token(&prefix, AddBos::Never)
        .map_err(|e| Error::Other(format!("tokenize prefix: {e}")))?;

    let prefix_len = prefix_tokens.len();
    if prefix_len == 0 {
        // Fallback to non-prefixed scoring
        return docs
            .iter()
            .map(|doc| {
                ctx.clear_kv_cache();
                score_yes_no(ctx, model, yes_id, no_id, query, doc, ctx_size)
            })
            .collect();
    }

    // Decode prefix
    ctx.clear_kv_cache();
    let mut batch = LlamaBatch::new(prefix_len, 1);
    for (i, &tok) in prefix_tokens.iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], false)
            .map_err(|e| Error::Other(format!("batch add prefix: {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| Error::Other(format!("decode prefix: {e}")))?;

    let suffix_template = "<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n";

    docs.iter()
        .map(|doc| {
            // Clear KV cache after prefix, keeping prefix entries
            let _ = ctx.clear_kv_cache_seq(Some(0), Some(prefix_len as u32), None);

            let doc_truncated = if doc.len() > MAX_DOC_CHARS {
                &doc[..doc.floor_char_boundary(MAX_DOC_CHARS)]
            } else {
                doc
            };

            let suffix = format!("{doc_truncated}{suffix_template}");
            let suffix_tokens = model
                .str_to_token(&suffix, AddBos::Never)
                .map_err(|e| Error::Other(format!("tokenize suffix: {e}")))?;

            if suffix_tokens.is_empty() {
                return Ok(0.0);
            }

            let total_len = prefix_len + suffix_tokens.len();
            let n = total_len.min(ctx_size as usize - 1);
            let suffix_n = n.saturating_sub(prefix_len);
            if suffix_n == 0 {
                return Ok(0.0);
            }

            let mut suffix_batch = LlamaBatch::new(suffix_n, 1);
            for (i, &tok) in suffix_tokens[..suffix_n].iter().enumerate() {
                let pos = (prefix_len + i) as i32;
                let is_last = i == suffix_n - 1;
                suffix_batch
                    .add(tok, pos, &[0], is_last)
                    .map_err(|e| Error::Other(format!("batch add suffix: {e}")))?;
            }

            ctx.decode(&mut suffix_batch)
                .map_err(|e| Error::Other(format!("decode suffix: {e}")))?;

            let logits = ctx.get_logits_ith((suffix_n - 1) as i32);
            let yes_idx = yes_id as usize;
            let no_idx = no_id as usize;
            if yes_idx >= logits.len() || no_idx >= logits.len() {
                return Err(Error::Other(format!(
                    "token id out of range: yes={yes_idx}, no={no_idx}, vocab={}",
                    logits.len()
                )));
            }

            let yes_logit = logits[yes_idx];
            let no_logit = logits[no_idx];
            let max_logit = yes_logit.max(no_logit);
            let yes_exp = (yes_logit - max_logit).exp() as f64;
            let no_exp = (no_logit - max_logit).exp() as f64;
            Ok(yes_exp / (yes_exp + no_exp))
        })
        .collect()
}

/// Create a reusable scoring context.
pub fn create_scoring_context<'a>(
    model: &'a LlamaModel,
    backend: &'static LlamaBackend,
    ctx_size: u32,
) -> Result<llama_cpp_2::context::LlamaContext<'a>> {
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(ctx_size))
        .with_offload_kqv(false)
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads);
    model
        .new_context(backend, ctx_params)
        .map_err(|e| Error::Other(format!("scoring context: {e}")))
}
