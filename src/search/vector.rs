// Vector search: embed query → kNN via sqlite-vec → join documents.

use crate::db::{CollectionDb, vectors};
use crate::error::Result;
use crate::llm::embedding::Embedder;
use crate::types::SearchResult;

pub struct VecSearchRequest<'a> {
    pub query: &'a str,
    pub limit: usize,
    pub min_score: Option<f64>,
}

pub fn search(
    embedder: &Embedder,
    dbs: &[CollectionDb],
    req: &VecSearchRequest,
) -> Result<Vec<SearchResult>> {
    let query_emb = embedder.embed_query(req.query)?;

    let mut all: Vec<SearchResult> = dbs
        .iter()
        .flat_map(|db| {
            vectors::search(db.conn(), &query_emb, &db.name, req.limit * 2).unwrap_or_else(|e| {
                eprintln!("warn: vector search on '{}' failed: {e}", db.name);
                vec![]
            })
        })
        .collect();

    if let Some(min) = req.min_score {
        all.retain(|r| r.score >= min);
    }

    SearchResult::sort_desc(&mut all);
    all.truncate(req.limit);
    Ok(all)
}
