// `ir rerank-eval` (research, research/reranker-eval.md): rank frozen tier-2 windows with the
// configured reranker (src/llm/mod.rs load_scorer) or with the scores stored in the dump, and
// with search's own blend (hybrid::blend), so a reranker is measured apart from retrieval.
// Input: IR_RERANK_DUMP lines (hybrid::dump_line), each optionally with an `id`.

use crate::error::{Error, Result};
use crate::search::hybrid;
use crate::types::SearchResult;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Deserialize)]
struct Doc {
    collection: String,
    path: String,
    hash: String,
    fused: f64,
    text: Option<String>,
    score: Option<f64>,
    ms: Option<f64>,
    tokens: Option<usize>,
}

// ^ Gate lines (hybrid::dump_gate) carry only `gate` and `query`: they skipped tier 2.
#[derive(Deserialize)]
struct Line {
    gate: Option<String>,
    id: Option<String>,
    query: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    limit: usize,
    #[serde(default)]
    keep_window: bool,
    #[serde(default)]
    window: Vec<Doc>,
    #[serde(default)]
    rest: Vec<Doc>,
}

impl Doc {
    fn result(&self) -> SearchResult {
        SearchResult {
            collection: self.collection.clone(),
            path: self.path.clone(),
            title: String::new(),
            score: self.fused,
            snippet: None,
            hash: self.hash.clone(),
            doc_id: String::new(),
            content: None,
            chunk_seq: None,
            evidence: None,
            kind: None,
        }
    }
}

pub fn run(input: &Path, output: &Path, window: usize, from_dump: bool) -> Result<()> {
    let open_err = |p: &Path, e: std::io::Error| Error::Other(format!("{}: {e}", p.display()));
    let reader = BufReader::new(File::open(input).map_err(|e| open_err(input, e))?);
    let scorer = if from_dump {
        None
    } else {
        Some(crate::llm::load_scorer()?)
    };
    let mut out = BufWriter::new(File::create(output).map_err(|e| open_err(output, e))?);
    for (n, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let at =
            |e: &dyn std::fmt::Display| Error::Other(format!("{}:{}: {e}", input.display(), n + 1));
        let rec: Line = serde_json::from_str(&line).map_err(|e| at(&e))?;
        if rec.gate.as_deref().is_some_and(|g| g != "t2") {
            continue;
        }
        let win = &rec.window[..window.min(rec.window.len())];
        // ^ Listwise scores are ranks within the whole window, and its cost sits on document 0.
        if scorer.is_none() && win.len() < rec.window.len() && rec.model.starts_with("listwise:") {
            return Err(at(&format!(
                "{} ranked the whole window of {n}; --from-dump needs --window >= {n}",
                rec.model,
                n = rec.window.len()
            )));
        }
        let (scores, ms, tokens): (Vec<Option<f64>>, f64, usize) = match &scorer {
            None => (
                win.iter().map(|d| d.score).collect(),
                win.iter().filter_map(|d| d.ms).sum(),
                win.iter().filter_map(|d| d.tokens).sum(),
            ),
            Some(s) => {
                let texts: Vec<&str> = win
                    .iter()
                    .map(|d| d.text.as_deref().unwrap_or(""))
                    .collect();
                let c = s
                    .score_batch_costed(&rec.query, &texts)
                    .map_err(|e| at(&e))?;
                if c.scores.len() != win.len() {
                    let n = (c.scores.len(), win.len());
                    return Err(at(&format!(
                        "scorer returned {} scores for {} documents",
                        n.0, n.1
                    )));
                }
                // ^ As in search: a document without text gets no score.
                let scores = win
                    .iter()
                    .zip(&c.scores)
                    .map(|(d, &s)| d.text.as_ref().map(|_| s))
                    .collect();
                (scores, c.doc_ms.iter().sum(), c.doc_tokens.iter().sum())
            }
        };
        let candidates = rec
            .window
            .iter()
            .chain(&rec.rest)
            .map(Doc::result)
            .collect();
        let ranked = hybrid::blend(candidates, &scores, rec.keep_window, rec.limit);
        let row = serde_json::json!({
            "id": rec.id,
            "window": win.len(),
            "ms": ms,
            "tokens": tokens,
            "scores": scores,
            "blend": ranked.iter().map(|r| r.path.as_str()).collect::<Vec<_>>(),
        });
        writeln!(out, "{row}")?;
    }
    out.flush()?;
    Ok(())
}
