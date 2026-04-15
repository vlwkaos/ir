// Shared autoregressive generation: tokenize prompt → sampling loop → string output.
// Used by Expander and Combined.
// docs: https://docs.rs/llama-cpp-2/latest/llama_cpp_2/

use crate::error::{Error, Result};
use crate::llm::LlamaBackend;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel},
    sampling::LlamaSampler,
};
use std::num::NonZeroU32;

pub struct GenerateParams {
    pub ctx_size: u32,
    pub max_tokens: usize,
    pub add_bos: AddBos,
    pub temp: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub seed: u32,
}

pub fn generate(
    model: &LlamaModel,
    backend: &'static LlamaBackend,
    prompt: &str,
    params: &GenerateParams,
) -> Result<String> {
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(params.ctx_size))
        .with_offload_kqv(false)
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| Error::Other(format!("generate context: {e}")))?;

    let prompt_tokens = model
        .str_to_token(prompt, params.add_bos)
        .map_err(|e| Error::Other(format!("tokenize: {e}")))?;
    let n_prompt = prompt_tokens.len();
    if n_prompt == 0 {
        return Ok(String::new());
    }

    let mut batch = LlamaBatch::new(n_prompt, 1);
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], i == n_prompt - 1)
            .map_err(|e| Error::Other(format!("batch add: {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| Error::Other(format!("decode prompt: {e}")))?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(params.temp),
        LlamaSampler::top_k(params.top_k),
        LlamaSampler::top_p(params.top_p, 1),
        LlamaSampler::dist(params.seed),
    ]);

    let mut output = String::new();
    let mut n_cur = n_prompt as i32;

    for _ in 0..params.max_tokens {
        let token = sampler.sample(&ctx, -1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let bytes = model
            .token_to_piece_bytes(token, 32, true, None)
            .map_err(|e| Error::Other(format!("token_to_piece_bytes: {e}")))?;
        output.push_str(&String::from_utf8_lossy(&bytes));

        let mut next = LlamaBatch::new(1, 1);
        next.add(token, n_cur, &[0], true)
            .map_err(|e| Error::Other(format!("batch next: {e}")))?;
        ctx.decode(&mut next)
            .map_err(|e| Error::Other(format!("decode next: {e}")))?;
        n_cur += 1;
    }

    Ok(output)
}
