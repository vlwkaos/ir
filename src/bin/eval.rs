// NFCorpus relevancy evaluation: measures nDCG@10 and Recall@10 for BM25, Vector, and Hybrid.
//
// Dataset: BEIR/NFCorpus — 3632 medical documents, 323 test queries with graded relevance
// Download: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip
//           Unzip to test-data/nfcorpus/ relative to the workspace root.
//
// Usage:
//   cargo run --bin eval -- --data test-data/nfcorpus [--limit N] [--mode bm25|vector|hybrid|all]
//
// Modes:
//   bm25   — FTS5 BM25 only (no model required)
//   vector — embedding kNN only (requires embedding model)
//   hybrid — BM25 probe → fallback sub-queries → RRF fusion (requires embedding model)
//   all    — run all modes and print comparison table (default)
//
// Primary metric: nDCG@10 (graded relevance, log-discounted cumulative gain)

use rusqlite::Connection;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use ir::db::{fts, schema, vectors};
use ir::error::{Error, Result};
use ir::index::{chunker, hasher};
use ir::llm::{
    embedding::{Embedder, EmbeddingPooling},
    expander::{Expander, SubQueryKind},
    models,
    reranker::Reranker,
};
use ir::types::SearchResult;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum ExpanderFusionMode {
    Rrf,
    Score,
}

struct Args {
    data_dir: PathBuf,
    cache_db: PathBuf,
    limit: usize,
    mode: EvalMode,
    pooling: Option<EmbeddingPooling>,
    alpha: f64,
    use_rerank: bool,
    rerank_weight: f64,
    rerank_top_n: usize,
    use_expander: bool,
    expander_fusion: ExpanderFusionMode,
    rrf_k: f64,
    rrf_lex_weight: f64,
    rrf_vec_weight: f64,
    rrf_hyde_weight: f64,
    adaptive_alpha: bool,
    alpha_normalizer: f64,
    use_prf: bool,
    prf_weight: f64,
    prf_docs: usize,
    prf_terms: usize,
    tune_alpha: bool,
    tune_rerank: bool,
    chunk_size_tokens: Option<usize>,
    max_docs: Option<usize>,
    max_queries: Option<usize>,
    rrf_no_expander: bool,
    title_weight: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EvalMode {
    Bm25,
    Vector,
    Hybrid,
    All,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut data_dir = PathBuf::from("test-data/nfcorpus");
    let mut cache_db = PathBuf::from("test-data/nfcorpus-eval.sqlite");
    let mut limit = 10;
    let mut mode = EvalMode::All;
    let mut pooling: Option<EmbeddingPooling> = None;
    let mut alpha = DEFAULT_SCORE_FUSION_ALPHA;
    let mut use_rerank = false;
    let mut rerank_weight = DEFAULT_RERANK_WEIGHT;
    let mut rerank_top_n = DEFAULT_RERANK_TOP_N;
    let mut use_expander = false;
    let mut expander_fusion = ExpanderFusionMode::Rrf;
    let mut rrf_k = DEFAULT_RRF_K;
    let mut rrf_lex_weight = DEFAULT_RRF_LEX_WEIGHT;
    let mut rrf_vec_weight = DEFAULT_RRF_VEC_WEIGHT;
    let mut rrf_hyde_weight = DEFAULT_RRF_HYDE_WEIGHT;
    let mut adaptive_alpha = false;
    let mut alpha_normalizer = DEFAULT_ALPHA_NORMALIZER;
    let mut use_prf = false;
    let mut prf_weight = DEFAULT_PRF_WEIGHT;
    let mut prf_docs = DEFAULT_PRF_DOCS;
    let mut prf_terms = DEFAULT_PRF_TERMS;
    let mut tune_alpha = false;
    let mut tune_rerank = false;
    let mut chunk_size_tokens: Option<usize> = None;
    let mut max_docs: Option<usize> = None;
    let mut max_queries: Option<usize> = None;
    let mut rrf_no_expander = false;
    let mut title_weight = 1.0f64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" | "-d" => {
                i += 1;
                if i < args.len() {
                    data_dir = PathBuf::from(&args[i]);
                }
            }
            "--cache-db" => {
                i += 1;
                if i < args.len() {
                    cache_db = PathBuf::from(&args[i]);
                }
            }
            "--limit" | "-k" => {
                i += 1;
                if i < args.len() {
                    limit = args[i].parse().expect("--limit must be a number");
                }
            }
            "--mode" | "-m" => {
                i += 1;
                if i < args.len() {
                    mode = match args[i].as_str() {
                        "bm25" => EvalMode::Bm25,
                        "vector" | "vec" => EvalMode::Vector,
                        "hybrid" => EvalMode::Hybrid,
                        "all" => EvalMode::All,
                        other => {
                            eprintln!("unknown mode '{other}'. Use: bm25, vector, hybrid, all");
                            std::process::exit(1);
                        }
                    };
                }
            }
            "--pooling" => {
                i += 1;
                if i < args.len() {
                    pooling = Some(
                        args[i]
                            .parse()
                            .unwrap_or_else(|e: String| panic!("--pooling: {e}")),
                    );
                }
            }
            "--alpha" => {
                i += 1;
                if i < args.len() {
                    alpha = args[i].parse().expect("--alpha must be a number");
                }
            }
            "--rerank" => {
                use_rerank = true;
            }
            "--rerank-weight" => {
                i += 1;
                if i < args.len() {
                    rerank_weight = args[i].parse().expect("--rerank-weight must be a number");
                }
            }
            "--rerank-fusion-weight" => {
                i += 1;
                if i < args.len() {
                    let fusion_weight: f64 = args[i]
                        .parse()
                        .expect("--rerank-fusion-weight must be a number");
                    rerank_weight = 1.0 - fusion_weight;
                }
            }
            "--rerank-top-n" => {
                i += 1;
                if i < args.len() {
                    rerank_top_n = args[i].parse().expect("--rerank-top-n must be a number");
                }
            }
            "--expander" => {
                use_expander = true;
            }
            "--expander-fusion" => {
                i += 1;
                if i < args.len() {
                    expander_fusion = match args[i].as_str() {
                        "rrf" => ExpanderFusionMode::Rrf,
                        "score" => ExpanderFusionMode::Score,
                        other => {
                            eprintln!("unknown --expander-fusion '{other}'. Use: rrf, score");
                            std::process::exit(1);
                        }
                    };
                }
            }
            "--rrf-k" => {
                i += 1;
                if i < args.len() {
                    rrf_k = args[i].parse().expect("--rrf-k must be a number");
                }
            }
            "--rrf-lex-weight" => {
                i += 1;
                if i < args.len() {
                    rrf_lex_weight = args[i].parse().expect("--rrf-lex-weight must be a number");
                }
            }
            "--rrf-vec-weight" => {
                i += 1;
                if i < args.len() {
                    rrf_vec_weight = args[i].parse().expect("--rrf-vec-weight must be a number");
                }
            }
            "--rrf-hyde-weight" => {
                i += 1;
                if i < args.len() {
                    rrf_hyde_weight = args[i].parse().expect("--rrf-hyde-weight must be a number");
                }
            }
            "--adaptive-alpha" => {
                adaptive_alpha = true;
            }
            "--alpha-normalizer" => {
                i += 1;
                if i < args.len() {
                    alpha_normalizer = args[i]
                        .parse()
                        .expect("--alpha-normalizer must be a number");
                }
            }
            "--prf" => {
                use_prf = true;
            }
            "--prf-weight" => {
                i += 1;
                if i < args.len() {
                    prf_weight = args[i].parse().expect("--prf-weight must be a number");
                }
            }
            "--prf-docs" => {
                i += 1;
                if i < args.len() {
                    prf_docs = args[i].parse().expect("--prf-docs must be a number");
                }
            }
            "--prf-terms" => {
                i += 1;
                if i < args.len() {
                    prf_terms = args[i].parse().expect("--prf-terms must be a number");
                }
            }
            "--tune-alpha" => {
                tune_alpha = true;
            }
            "--tune-rerank" => {
                tune_rerank = true;
            }
            "--chunk-size" => {
                i += 1;
                if i < args.len() {
                    chunk_size_tokens =
                        Some(args[i].parse().expect("--chunk-size must be a number"));
                }
            }
            "--max-docs" => {
                i += 1;
                if i < args.len() {
                    max_docs = Some(args[i].parse().expect("--max-docs must be a number"));
                }
            }
            "--max-queries" => {
                i += 1;
                if i < args.len() {
                    max_queries = Some(args[i].parse().expect("--max-queries must be a number"));
                }
            }
            "--rrf-no-expander" => {
                rrf_no_expander = true;
            }
            "--title-weight" => {
                i += 1;
                if i < args.len() {
                    title_weight = args[i].parse().expect("--title-weight must be a number");
                }
            }
            "--help" | "-h" => {
                println!(
                    "Usage: eval [--data DIR] [--limit K] [--mode bm25|vector|hybrid|all] \
                     [--pooling none|mean|cls|last|rank] \
                     [--cache-db PATH] [--alpha A] [--adaptive-alpha] [--alpha-normalizer N] \
                     [--rerank] [--rerank-weight W] [--rerank-top-n N] \
                     [--expander] [--expander-fusion rrf|score] [--rrf-k K] \
                     [--rrf-lex-weight W] [--rrf-vec-weight W] [--rrf-hyde-weight W] \
                     [--prf] [--prf-weight W] [--prf-docs N] [--prf-terms N] \
                     [--tune-alpha] [--tune-rerank] [--chunk-size TOKENS] [--max-docs N] [--max-queries N] \
                     [--rrf-no-expander] [--title-weight W]"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    Args {
        data_dir,
        cache_db,
        limit,
        mode,
        pooling,
        alpha,
        use_rerank,
        rerank_weight,
        rerank_top_n,
        use_expander,
        expander_fusion,
        rrf_k,
        rrf_lex_weight,
        rrf_vec_weight,
        rrf_hyde_weight,
        adaptive_alpha,
        alpha_normalizer,
        use_prf,
        prf_weight,
        prf_docs,
        prf_terms,
        tune_alpha,
        tune_rerank,
        chunk_size_tokens,
        max_docs,
        max_queries,
        rrf_no_expander,
        title_weight,
    }
}

// ── Dataset types ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct CorpusDoc {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct Query {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

/// qrel score 2 = highly relevant, 1 = relevant (both count as relevant)
type Qrels = HashMap<String, HashMap<String, u32>>; // query_id → {doc_id → score}

// ── Data loading ──────────────────────────────────────────────────────────────

fn load_corpus(path: &Path) -> Result<Vec<CorpusDoc>> {
    let f = std::fs::File::open(path).map_err(Error::Io)?;
    let reader = BufReader::new(f);
    let mut docs = Vec::new();
    for line in reader.lines() {
        let line: String = line.map_err(Error::Io)?;
        if line.trim().is_empty() {
            continue;
        }
        let doc: CorpusDoc =
            serde_json::from_str(&line).map_err(|e| Error::Other(format!("parse corpus: {e}")))?;
        docs.push(doc);
    }
    Ok(docs)
}

fn load_queries(path: &Path) -> Result<Vec<Query>> {
    let f = std::fs::File::open(path).map_err(Error::Io)?;
    let reader = BufReader::new(f);
    let mut queries = Vec::new();
    for line in reader.lines() {
        let line: String = line.map_err(Error::Io)?;
        if line.trim().is_empty() {
            continue;
        }
        let q: Query =
            serde_json::from_str(&line).map_err(|e| Error::Other(format!("parse query: {e}")))?;
        queries.push(q);
    }
    Ok(queries)
}

fn load_qrels(path: &Path) -> Result<Qrels> {
    let f = std::fs::File::open(path).map_err(Error::Io)?;
    let reader = BufReader::new(f);
    let mut qrels: Qrels = HashMap::new();
    let mut first = true;
    for line in reader.lines() {
        let line: String = line.map_err(Error::Io)?;
        if first {
            first = false;
            if line.starts_with("query-id") {
                continue;
            }
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 3 {
            continue;
        }
        let query_id = parts[0].to_string();
        let doc_id = parts[1].to_string();
        let score: u32 = parts[2].trim().parse().unwrap_or(0);
        if score > 0 {
            qrels.entry(query_id).or_default().insert(doc_id, score);
        }
    }
    Ok(qrels)
}

// ── Indexing ──────────────────────────────────────────────────────────────────

fn index_corpus(conn: &Connection, docs: &[CorpusDoc]) -> Result<()> {
    let now = "2024-01-01T00:00:00Z";

    for doc in docs {
        let text = doc_text(doc);
        let hash = hasher::hash_bytes(format!("{}:{}", doc.id, text).as_bytes());

        conn.execute(
            "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?1, ?2, ?3)",
            rusqlite::params![hash, text, now],
        )?;

        // path = doc_id so search results expose the BEIR corpus id
        conn.execute(
            "INSERT OR IGNORE INTO documents (path, title, hash, created_at, modified_at, active)
             VALUES (?1, ?2, ?3, ?4, ?5, 1)",
            rusqlite::params![doc.id, doc.title, hash, now, now],
        )?;
    }

    Ok(())
}

fn current_vector_dim(conn: &Connection) -> Option<usize> {
    let ddl: Option<String> = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name = 'vectors_vec' AND sql IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .ok();
    let ddl = ddl?;

    let marker = "float[";
    let start = ddl.find(marker)? + marker.len();
    let rest = &ddl[start..];
    let end = rest.find(']')?;
    rest[..end].trim().parse::<usize>().ok()
}

fn ensure_vector_dimension(conn: &Connection, dim: usize) -> Result<()> {
    if dim == 0 {
        return Err(Error::Other(
            "embedding dimension resolved to 0".to_string(),
        ));
    }

    match current_vector_dim(conn) {
        Some(existing) if existing == dim => return Ok(()),
        Some(existing) => {
            println!("  vector dimension mismatch ({existing} -> {dim}), rebuilding vector table");
            conn.execute_batch(
                "DROP TABLE IF EXISTS vectors_vec;
                 DELETE FROM content_vectors;",
            )?;
        }
        None => {
            conn.execute_batch("DROP TABLE IF EXISTS vectors_vec;")?;
        }
    }

    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vectors_vec USING vec0(
            hash_seq TEXT PRIMARY KEY,
            embedding float[{dim}] distance_metric=cosine
        );"
    ))?;

    Ok(())
}

fn embed_corpus(conn: &Connection, docs: &[CorpusDoc], embedder: &Embedder) -> Result<()> {
    // Collect all pending docs first, then embed in a single batch (one Metal context).
    // ! Per-doc batching creates/destroys thousands of GPU contexts and exhausts the Metal heap.
    struct PendingDoc {
        hash: String,
        title: String,
        chunks: Vec<ir::index::chunker::Chunk>,
    }

    let mut pending: Vec<PendingDoc> = Vec::new();
    for doc in docs {
        let text = doc_text(doc);
        let hash = hasher::hash_bytes(format!("{}:{}", doc.id, text).as_bytes());
        let already: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM content_vectors WHERE hash = ?1",
                [&hash],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if already > 0 {
            continue;
        }
        pending.push(PendingDoc {
            hash,
            title: doc.title.clone(),
            chunks: chunker::chunk_document(&text),
        });
    }

    if pending.is_empty() {
        println!("  all embeddings cached");
        return Ok(());
    }

    let total_chunks: usize = pending.iter().map(|d| d.chunks.len()).sum();
    println!("  embedding {} chunks from {} documents...", total_chunks, pending.len());

    let inputs: Vec<(String, String)> = pending
        .iter()
        .flat_map(|d| d.chunks.iter().map(|c| (d.title.clone(), c.text.clone())))
        .collect();

    let all_embeddings = embedder.embed_doc_batch(&inputs)?;

    let mut emb_iter = all_embeddings.iter();
    for doc in &pending {
        for chunk in &doc.chunks {
            let emb = emb_iter.next().expect("embedding count mismatch");
            let hash_seq = format!("{}_{}", doc.hash, chunk.seq);
            vectors::insert(conn, &hash_seq, emb)?;
            vectors::mark_embedded(
                conn,
                &doc.hash,
                chunk.seq as i64,
                chunk.pos as i64,
                models::EMBEDDING,
            )?;
        }
    }
    Ok(())
}

fn embed_queries(embedder: &Embedder, queries: &[Query]) -> Result<HashMap<String, Vec<f32>>> {
    let inputs: Vec<String> = queries.iter().map(|q| q.text.clone()).collect();
    let embeddings = embedder.embed_query_batch(&inputs)?;
    if embeddings.len() != queries.len() {
        return Err(Error::Other(format!(
            "query embedding count mismatch: got {}, expected {}",
            embeddings.len(),
            queries.len()
        )));
    }
    Ok(queries
        .iter()
        .zip(embeddings)
        .map(|(q, emb)| (q.id.clone(), emb))
        .collect())
}

fn doc_text(doc: &CorpusDoc) -> String {
    if doc.title.is_empty() {
        doc.text.clone()
    } else {
        format!("{}\n\n{}", doc.title, doc.text)
    }
}

// ── Metrics ───────────────────────────────────────────────────────────────────

fn ndcg_at_k(ranked: &[String], relevant: &HashMap<String, u32>, k: usize) -> f64 {
    let dcg_val = dcg(ranked, relevant, k);
    let ideal = ideal_dcg(relevant, k);
    if ideal == 0.0 { 0.0 } else { dcg_val / ideal }
}

fn dcg(ranked: &[String], relevant: &HashMap<String, u32>, k: usize) -> f64 {
    ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, doc_id)| {
            let rel = *relevant.get(doc_id).unwrap_or(&0) as f64;
            (2.0_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2()
        })
        .sum()
}

fn ideal_dcg(relevant: &HashMap<String, u32>, k: usize) -> f64 {
    let mut scores: Vec<f64> = relevant.values().map(|&s| s as f64).collect();
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    scores
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &rel)| (2.0_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2())
        .sum()
}

fn recall_at_k(ranked: &[String], relevant: &HashMap<String, u32>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let hits = ranked
        .iter()
        .take(k)
        .filter(|id| relevant.contains_key(*id))
        .count();
    hits as f64 / relevant.len() as f64
}

// ── Search ────────────────────────────────────────────────────────────────────

fn bm25_search(conn: &Connection, query: &str, limit: usize, title_weight: f64) -> Vec<SearchResult> {
    let fts_query = fts::build_query(query);
    if fts_query.is_empty() {
        return vec![];
    }
    let q = fts::BM25Query {
        fts_query,
        collection: "eval",
        limit,
        title_weight: if title_weight == 1.0 { None } else { Some(title_weight) },
    };
    fts::search(conn, &q).unwrap_or_default()
}

fn vector_search_from_embedding(
    conn: &Connection,
    embedding: &[f32],
    limit: usize,
) -> Vec<SearchResult> {
    vectors::search(conn, embedding, "eval", limit).unwrap_or_default()
}

const DEFAULT_SCORE_FUSION_ALPHA: f64 = 0.80;
const DEFAULT_RERANK_WEIGHT: f64 = 0.60;
const DEFAULT_RERANK_TOP_N: usize = 20;
const DEFAULT_RRF_K: f64 = 60.0;
const DEFAULT_RRF_LEX_WEIGHT: f64 = 1.0;
const DEFAULT_RRF_VEC_WEIGHT: f64 = 1.5;
const DEFAULT_RRF_HYDE_WEIGHT: f64 = 1.0;
const DEFAULT_ALPHA_NORMALIZER: f64 = 0.20;
const DEFAULT_PRF_WEIGHT: f64 = 0.30;
const DEFAULT_PRF_DOCS: usize = 3;
const DEFAULT_PRF_TERMS: usize = 10;

#[derive(Clone, Copy)]
struct RrfWeights {
    lex: f64,
    vec: f64,
    hyde: f64,
}

#[derive(Clone, Copy)]
struct PrfConfig {
    weight: f64,
    top_docs: usize,
    top_terms: usize,
}

#[derive(Clone, Copy)]
struct HybridConfig {
    alpha: f64,
    adaptive_alpha: bool,
    alpha_normalizer: f64,
    use_expander: bool,
    expander_fusion: ExpanderFusionMode,
    rrf_k: f64,
    rrf_weights: RrfWeights,
    prf: Option<PrfConfig>,
    rrf_no_expander: bool,
    title_weight: f64,
}

struct HybridRun {
    results: Vec<SearchResult>,
    bm25_seed: Vec<SearchResult>,
}

struct TermStats {
    df: HashMap<String, usize>,
    n_docs: usize,
}

fn hybrid_search(
    conn: &Connection,
    query: &str,
    query_embedding: &[f32],
    limit: usize,
    embedder: &Embedder,
    expander: Option<&Expander>,
    cfg: &HybridConfig,
) -> HybridRun {
    let fetch_n = limit * 3;
    let bm25_results = bm25_search(conn, query, fetch_n, cfg.title_weight);
    let vec_results = vector_search_from_embedding(conn, query_embedding, fetch_n);

    let alpha = if cfg.adaptive_alpha {
        adaptive_alpha(
            query,
            cfg.alpha,
            &bm25_results,
            &vec_results,
            cfg.alpha_normalizer,
        )
    } else {
        cfg.alpha
    };

    let results = if cfg.use_expander {
        if let Some(expander) = expander {
            expanded_fusion(
                conn,
                query,
                limit,
                embedder,
                expander,
                cfg,
                alpha,
                &bm25_results,
                &vec_results,
            )
        } else if cfg.rrf_no_expander {
            rrf_fuse_weighted(
                &[
                    (vec_results.clone(), cfg.rrf_weights.vec),
                    (bm25_results.clone(), cfg.rrf_weights.lex),
                ],
                cfg.rrf_k,
                limit,
            )
        } else {
            score_fusion_from_lists(&bm25_results, &vec_results, limit, alpha)
        }
    } else if cfg.rrf_no_expander {
        rrf_fuse_weighted(
            &[
                (vec_results.clone(), cfg.rrf_weights.vec),
                (bm25_results.clone(), cfg.rrf_weights.lex),
            ],
            cfg.rrf_k,
            limit,
        )
    } else {
        score_fusion_from_lists(&bm25_results, &vec_results, limit, alpha)
    };

    HybridRun {
        results,
        bm25_seed: bm25_results,
    }
}

fn score_fusion_from_lists(
    bm25_results: &[SearchResult],
    vec_results: &[SearchResult],
    limit: usize,
    alpha: f64,
) -> Vec<SearchResult> {
    let mut scores: HashMap<String, (f64, f64, SearchResult)> = HashMap::new();

    for r in bm25_results {
        let entry = scores
            .entry(r.path.clone())
            .or_insert((0.0, 0.0, r.clone()));
        entry.0 = entry.0.max(r.score);
    }
    for r in vec_results {
        let entry = scores
            .entry(r.path.clone())
            .or_insert((0.0, 0.0, r.clone()));
        entry.1 = entry.1.max(r.score);
    }

    let mut merged: Vec<SearchResult> = scores
        .into_values()
        .map(|(bm25_score, vec_score, mut result)| {
            result.score = alpha * vec_score + (1.0 - alpha) * bm25_score;
            result
        })
        .collect();

    SearchResult::sort_desc(&mut merged);
    merged.truncate(limit);
    merged
}

fn expanded_fusion(
    conn: &Connection,
    query: &str,
    limit: usize,
    embedder: &Embedder,
    expander: &Expander,
    cfg: &HybridConfig,
    alpha: f64,
    fallback_bm25: &[SearchResult],
    fallback_vec: &[SearchResult],
) -> Vec<SearchResult> {
    let sub_queries = expander
        .expand(query)
        .unwrap_or_else(|_| ir::llm::expander::fallback(query));
    let fetch_n = limit * 3;

    match cfg.expander_fusion {
        ExpanderFusionMode::Rrf => {
            let mut lists: Vec<(Vec<SearchResult>, f64)> = Vec::new();
            for sub in &sub_queries {
                let weight = match sub.kind {
                    SubQueryKind::Lex => cfg.rrf_weights.lex,
                    SubQueryKind::Vec => cfg.rrf_weights.vec,
                    SubQueryKind::Hyde => cfg.rrf_weights.hyde,
                };

                let results = match sub.kind {
                    SubQueryKind::Lex => bm25_search(conn, &sub.text, fetch_n, cfg.title_weight),
                    SubQueryKind::Vec | SubQueryKind::Hyde => {
                        if let Ok(embedding) = embedder.embed_query(&sub.text) {
                            vector_search_from_embedding(conn, &embedding, fetch_n)
                        } else {
                            vec![]
                        }
                    }
                };

                if !results.is_empty() {
                    lists.push((results, weight));
                }
            }

            if lists.is_empty() {
                return score_fusion_from_lists(fallback_bm25, fallback_vec, limit, alpha);
            }
            rrf_fuse_weighted(&lists, cfg.rrf_k, limit)
        }
        ExpanderFusionMode::Score => {
            let mut bm25_pool: Vec<SearchResult> = Vec::new();
            let mut vec_pool: Vec<SearchResult> = Vec::new();

            for sub in &sub_queries {
                let weight = match sub.kind {
                    SubQueryKind::Lex => cfg.rrf_weights.lex,
                    SubQueryKind::Vec => cfg.rrf_weights.vec,
                    SubQueryKind::Hyde => cfg.rrf_weights.hyde,
                };
                match sub.kind {
                    SubQueryKind::Lex => {
                        let mut results = bm25_search(conn, &sub.text, fetch_n, cfg.title_weight);
                        results.iter_mut().for_each(|r| r.score *= weight);
                        bm25_pool.extend(results);
                    }
                    SubQueryKind::Vec | SubQueryKind::Hyde => {
                        if let Ok(embedding) = embedder.embed_query(&sub.text) {
                            let mut results =
                                vector_search_from_embedding(conn, &embedding, fetch_n);
                            results.iter_mut().for_each(|r| r.score *= weight);
                            vec_pool.extend(results);
                        }
                    }
                }
            }

            if bm25_pool.is_empty() {
                bm25_pool.extend_from_slice(fallback_bm25);
            }
            if vec_pool.is_empty() {
                vec_pool.extend_from_slice(fallback_vec);
            }
            score_fusion_from_lists(&bm25_pool, &vec_pool, limit, alpha)
        }
    }
}

fn rrf_fuse_weighted(
    lists: &[(Vec<SearchResult>, f64)],
    rrf_k: f64,
    limit: usize,
) -> Vec<SearchResult> {
    let k = if rrf_k <= 0.0 { DEFAULT_RRF_K } else { rrf_k };
    let mut scores: HashMap<String, (f64, SearchResult)> = HashMap::new();

    for (results, weight) in lists {
        for (rank, result) in results.iter().enumerate() {
            let base = weight / (k + rank as f64 + 1.0);
            let bonus = match rank {
                0 => 0.05,
                1 | 2 => 0.02,
                _ => 0.0,
            };
            let contribution = base + bonus;

            scores
                .entry(result.path.clone())
                .and_modify(|(score, _)| *score += contribution)
                .or_insert((contribution, result.clone()));
        }
    }

    let mut merged: Vec<SearchResult> = scores
        .into_values()
        .map(|(rrf_score, mut result)| {
            result.score = rrf_score;
            result
        })
        .collect();
    SearchResult::sort_desc(&mut merged);
    merged.truncate(limit);
    merged
}

fn adaptive_alpha(
    query: &str,
    base_alpha: f64,
    bm25_results: &[SearchResult],
    vec_results: &[SearchResult],
    normalizer: f64,
) -> f64 {
    let bm25_gap = top_gap(bm25_results);
    let dense_margin = top_gap(vec_results);
    let mut alpha = base_alpha + 0.15 * (dense_margin - bm25_gap) / normalizer.max(1e-6);

    let query_len = query
        .split_whitespace()
        .filter(|t| !t.trim().is_empty())
        .count();
    if query_len <= 3 {
        alpha -= 0.05;
    } else if query_len >= 8 {
        alpha += 0.03;
    }

    alpha.clamp(0.55, 0.95)
}

fn top_gap(results: &[SearchResult]) -> f64 {
    match (results.first(), results.get(1)) {
        (Some(a), Some(b)) => (a.score - b.score).max(0.0),
        (Some(a), None) => a.score.max(0.0),
        _ => 0.0,
    }
}

fn fuse_linear(
    primary: &[SearchResult],
    secondary: &[SearchResult],
    primary_weight: f64,
    secondary_weight: f64,
    limit: usize,
) -> Vec<SearchResult> {
    let mut scores: HashMap<String, (f64, f64, SearchResult)> = HashMap::new();

    for r in primary {
        let entry = scores
            .entry(r.path.clone())
            .or_insert((0.0, 0.0, r.clone()));
        entry.0 = entry.0.max(r.score);
    }
    for r in secondary {
        let entry = scores
            .entry(r.path.clone())
            .or_insert((0.0, 0.0, r.clone()));
        entry.1 = entry.1.max(r.score);
    }

    let mut merged: Vec<SearchResult> = scores
        .into_values()
        .map(|(primary_score, secondary_score, mut result)| {
            result.score = primary_weight * primary_score + secondary_weight * secondary_score;
            result
        })
        .collect();
    SearchResult::sort_desc(&mut merged);
    merged.truncate(limit);
    merged
}

fn build_term_stats(doc_texts: &HashMap<String, String>) -> TermStats {
    let mut df: HashMap<String, usize> = HashMap::new();
    for text in doc_texts.values() {
        let mut seen: HashMap<String, bool> = HashMap::new();
        for term in tokenize_terms(text) {
            seen.insert(term, true);
        }
        for term in seen.into_keys() {
            *df.entry(term).or_insert(0) += 1;
        }
    }
    TermStats {
        df,
        n_docs: doc_texts.len(),
    }
}

fn tokenize_terms(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter_map(|token| {
            let term = token.trim().to_ascii_lowercase();
            if term.len() < 3 || is_stopword(&term) {
                None
            } else {
                Some(term)
            }
        })
        .collect()
}

fn is_stopword(term: &str) -> bool {
    matches!(
        term,
        "the"
            | "and"
            | "for"
            | "with"
            | "from"
            | "that"
            | "this"
            | "are"
            | "was"
            | "were"
            | "has"
            | "have"
            | "had"
            | "but"
            | "you"
            | "your"
            | "about"
            | "into"
            | "than"
            | "then"
            | "they"
            | "them"
            | "their"
            | "can"
            | "will"
            | "would"
            | "should"
            | "could"
            | "there"
            | "what"
            | "when"
            | "where"
            | "which"
            | "while"
            | "using"
            | "use"
            | "used"
            | "also"
            | "such"
            | "these"
            | "those"
            | "our"
            | "his"
            | "her"
            | "its"
            | "who"
            | "why"
            | "how"
    )
}

fn prf_expand_query(
    query: &str,
    bm25_seed: &[SearchResult],
    doc_texts: &HashMap<String, String>,
    term_stats: &TermStats,
    top_docs: usize,
    top_terms: usize,
) -> Option<String> {
    let query_terms: HashMap<String, bool> = tokenize_terms(query)
        .into_iter()
        .map(|t| (t, true))
        .collect();

    let mut tf: HashMap<String, usize> = HashMap::new();
    for result in bm25_seed.iter().take(top_docs) {
        let Some(text) = doc_texts.get(&result.path) else {
            continue;
        };
        for term in tokenize_terms(text) {
            if query_terms.contains_key(&term) {
                continue;
            }
            *tf.entry(term).or_insert(0) += 1;
        }
    }

    if tf.is_empty() {
        return None;
    }

    let mut scored_terms: Vec<(String, f64)> = tf
        .into_iter()
        .map(|(term, freq)| {
            let df = *term_stats.df.get(&term).unwrap_or(&1) as f64;
            let idf = ((term_stats.n_docs as f64 + 1.0) / (df + 1.0)).ln() + 1.0;
            (term, idf * freq as f64)
        })
        .collect();
    scored_terms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let expansion_terms: Vec<String> = scored_terms
        .into_iter()
        .take(top_terms)
        .map(|(term, _)| term)
        .collect();
    if expansion_terms.is_empty() {
        return None;
    }

    Some(format!("{query} {}", expansion_terms.join(" ")))
}

fn apply_prf(
    conn: &Connection,
    query: &str,
    limit: usize,
    base_results: &[SearchResult],
    bm25_seed: &[SearchResult],
    doc_texts: &HashMap<String, String>,
    term_stats: &TermStats,
    prf: PrfConfig,
) -> Vec<SearchResult> {
    let Some(expanded_query) = prf_expand_query(
        query,
        bm25_seed,
        doc_texts,
        term_stats,
        prf.top_docs,
        prf.top_terms,
    ) else {
        return base_results.to_vec();
    };

    let prf_results = bm25_search(conn, &expanded_query, limit * 3, 1.0);
    let primary_weight = (1.0 - prf.weight).clamp(0.0, 1.0);
    let secondary_weight = prf.weight.clamp(0.0, 1.0);
    fuse_linear(
        base_results,
        &prf_results,
        primary_weight,
        secondary_weight,
        limit,
    )
}

fn rerank_with_blend(
    conn: &Connection,
    query: &str,
    mut candidates: Vec<SearchResult>,
    reranker: &Reranker,
    doc_texts: &HashMap<String, String>,
    rerank_weight: f64,
    top_n: usize,
    cache: &mut HashMap<(String, String), f64>,
) -> Vec<SearchResult> {
    let top_n = top_n.min(candidates.len());

    for result in candidates.iter_mut().take(top_n) {
        let Some(text) = doc_texts.get(&result.path) else {
            continue;
        };

        let key = (query.to_string(), result.path.clone());
        let rerank_score = if let Some(cached) = cache.get(&key) {
            *cached
        } else {
            let cache_hash = hasher::hash_bytes(format!("{}\0{}", query, result.path).as_bytes());
            let cached_db: Option<f64> = conn
                .query_row(
                    "SELECT result FROM llm_cache WHERE hash = ?1",
                    [&cache_hash],
                    |row| row.get::<_, String>(0),
                )
                .ok()
                .and_then(|v| v.parse::<f64>().ok());

            if let Some(score) = cached_db {
                cache.insert(key, score);
                score
            } else {
                match reranker.score(query, text) {
                    Ok(score) => {
                        cache.insert(key, score);
                        let _ = conn.execute(
                            "INSERT OR REPLACE INTO llm_cache (hash, result, created_at)
                             VALUES (?1, ?2, ?3)",
                            rusqlite::params![
                                cache_hash,
                                score.to_string(),
                                chrono::Utc::now().to_rfc3339()
                            ],
                        );
                        score
                    }
                    Err(_) => continue,
                }
            }
        };

        result.score = (1.0 - rerank_weight) * result.score + rerank_weight * rerank_score;
    }

    SearchResult::sort_desc(&mut candidates);
    candidates
}

fn maybe_log_query_progress(phase: &str, index: usize, total: usize) {
    let done = index + 1;
    if done % 50 == 0 || done == total {
        println!("  {phase}: {done}/{total}");
    }
}

// ── Evaluation ────────────────────────────────────────────────────────────────

struct EvalResult {
    mode: &'static str,
    ndcg: f64,
    recall: f64,
    n_queries: usize,
}

fn evaluate_bm25(conn: &Connection, queries: &[Query], qrels: &Qrels, k: usize) -> EvalResult {
    let (mut total_ndcg, mut total_recall, mut n) = (0.0f64, 0.0f64, 0usize);
    for (i, q) in queries.iter().enumerate() {
        maybe_log_query_progress("bm25", i, queries.len());
        let Some(relevant) = qrels.get(&q.id) else {
            continue;
        };
        let results = bm25_search(conn, &q.text, k, 1.0);
        let ranked: Vec<String> = results.iter().map(|r| r.path.clone()).collect();
        total_ndcg += ndcg_at_k(&ranked, relevant, k);
        total_recall += recall_at_k(&ranked, relevant, k);
        n += 1;
    }
    EvalResult {
        mode: "bm25",
        ndcg: if n > 0 { total_ndcg / n as f64 } else { 0.0 },
        recall: if n > 0 { total_recall / n as f64 } else { 0.0 },
        n_queries: n,
    }
}

fn evaluate_vector(
    conn: &Connection,
    queries: &[Query],
    qrels: &Qrels,
    k: usize,
    query_embeddings: &HashMap<String, Vec<f32>>,
) -> EvalResult {
    let (mut total_ndcg, mut total_recall, mut n) = (0.0f64, 0.0f64, 0usize);
    for (i, q) in queries.iter().enumerate() {
        maybe_log_query_progress("vector", i, queries.len());
        let Some(relevant) = qrels.get(&q.id) else {
            continue;
        };
        let Some(embedding) = query_embeddings.get(&q.id) else {
            continue;
        };
        let results = vector_search_from_embedding(conn, embedding, k);
        let ranked: Vec<String> = results.iter().map(|r| r.path.clone()).collect();
        total_ndcg += ndcg_at_k(&ranked, relevant, k);
        total_recall += recall_at_k(&ranked, relevant, k);
        n += 1;
    }
    EvalResult {
        mode: "vector",
        ndcg: if n > 0 { total_ndcg / n as f64 } else { 0.0 },
        recall: if n > 0 { total_recall / n as f64 } else { 0.0 },
        n_queries: n,
    }
}

fn evaluate_hybrid(
    conn: &Connection,
    queries: &[Query],
    qrels: &Qrels,
    k: usize,
    query_embeddings: &HashMap<String, Vec<f32>>,
    embedder: &Embedder,
    expander: Option<&Expander>,
    hybrid_cfg: &HybridConfig,
    term_stats: &TermStats,
    reranker: Option<&Reranker>,
    rerank_weight: f64,
    rerank_top_n: usize,
    doc_texts: &HashMap<String, String>,
    mode_name: &'static str,
) -> EvalResult {
    let (mut total_ndcg, mut total_recall, mut n) = (0.0f64, 0.0f64, 0usize);
    let mut rerank_cache: HashMap<(String, String), f64> = HashMap::new();
    for (i, q) in queries.iter().enumerate() {
        maybe_log_query_progress(mode_name, i, queries.len());
        let Some(relevant) = qrels.get(&q.id) else {
            continue;
        };
        let Some(embedding) = query_embeddings.get(&q.id) else {
            continue;
        };
        let fusion_limit = if reranker.is_some() {
            (k * 2).max(rerank_top_n)
        } else {
            k
        };
        let mut run = hybrid_search(
            conn,
            &q.text,
            embedding,
            fusion_limit,
            embedder,
            expander,
            hybrid_cfg,
        );

        if let Some(prf_cfg) = hybrid_cfg.prf {
            run.results = apply_prf(
                conn,
                &q.text,
                fusion_limit,
                &run.results,
                &run.bm25_seed,
                doc_texts,
                term_stats,
                prf_cfg,
            );
        }

        let mut results = run.results;
        if let Some(rr) = reranker {
            results = rerank_with_blend(
                conn,
                &q.text,
                results,
                rr,
                doc_texts,
                rerank_weight,
                rerank_top_n,
                &mut rerank_cache,
            );
        }
        results.truncate(k);
        let ranked: Vec<String> = results.iter().map(|r| r.path.clone()).collect();
        total_ndcg += ndcg_at_k(&ranked, relevant, k);
        total_recall += recall_at_k(&ranked, relevant, k);
        n += 1;
    }
    EvalResult {
        mode: mode_name,
        ndcg: if n > 0 { total_ndcg / n as f64 } else { 0.0 },
        recall: if n > 0 { total_recall / n as f64 } else { 0.0 },
        n_queries: n,
    }
}

/// Grid search over score-fusion alpha values and report best.
fn tune_score_fusion(
    conn: &Connection,
    queries: &[Query],
    qrels: &Qrels,
    k: usize,
    query_embeddings: &HashMap<String, Vec<f32>>,
    embedder: &Embedder,
    expander: Option<&Expander>,
    base_cfg: &HybridConfig,
    doc_texts: &HashMap<String, String>,
    term_stats: &TermStats,
    reranker: Option<&Reranker>,
    rerank_weight: f64,
    rerank_top_n: usize,
) {
    println!("\n── Score-fusion tuning (α*vec + (1-α)*bm25) ──────────────────");
    println!("{:<8}  {:>10}  {:>12}", "alpha", "nDCG@10", "Recall@10");
    println!("{}", "─".repeat(36));

    let alphas = [0.70f64, 0.75, 0.80, 0.85, 0.90];
    let mut best = (0.0f64, 0.0f64, 0.0f64);

    for &alpha in &alphas {
        let mut cfg = *base_cfg;
        cfg.alpha = alpha;
        cfg.adaptive_alpha = false;
        let r = evaluate_hybrid(
            conn,
            queries,
            qrels,
            k,
            query_embeddings,
            embedder,
            expander,
            &cfg,
            term_stats,
            reranker,
            rerank_weight,
            rerank_top_n,
            doc_texts,
            "hybrid",
        );
        let ndcg = r.ndcg;
        let recall = r.recall;
        println!("{:<8.2}  {:>10.4}  {:>12.4}", alpha, ndcg, recall);
        if ndcg > best.0 {
            best = (ndcg, recall, alpha);
        }
    }
    println!("{}", "─".repeat(36));
    println!(
        "best: alpha={:.2} → nDCG@10={:.4}, Recall@10={:.4}",
        best.2, best.0, best.1
    );
}

/// Grid search over reranker blend:
/// final_score = fusion_weight * fusion + (1-fusion_weight) * rerank
fn tune_rerank_blend(
    conn: &Connection,
    queries: &[Query],
    qrels: &Qrels,
    k: usize,
    query_embeddings: &HashMap<String, Vec<f32>>,
    embedder: &Embedder,
    expander: Option<&Expander>,
    hybrid_cfg: &HybridConfig,
    term_stats: &TermStats,
    reranker: &Reranker,
    rerank_top_n: usize,
    doc_texts: &HashMap<String, String>,
) {
    println!("\n── Reranker blend tuning ───────────────────────────────────────");
    println!(
        "{:<14}  {:>10}  {:>12}",
        "rerank/fusion", "nDCG@10", "Recall@10"
    );
    println!("{}", "─".repeat(42));

    let rerank_weights = [0.30f64, 0.40, 0.50, 0.60, 0.70, 0.80];
    let mut best = (0.0f64, 0.0f64, 0.0f64);

    for &rerank_weight in &rerank_weights {
        let r = evaluate_hybrid(
            conn,
            queries,
            qrels,
            k,
            query_embeddings,
            embedder,
            expander,
            hybrid_cfg,
            term_stats,
            Some(reranker),
            rerank_weight,
            rerank_top_n,
            doc_texts,
            "hybrid-rerank",
        );
        println!(
            "{:<14}  {:>10.4}  {:>12.4}",
            format!("{:.2}/{:.2}", rerank_weight, 1.0 - rerank_weight),
            r.ndcg,
            r.recall
        );
        if r.ndcg > best.0 {
            best = (r.ndcg, r.recall, rerank_weight);
        }
    }
    println!("{}", "─".repeat(42));
    println!(
        "best: rerank/fusion = {:.2}/{:.2} → nDCG@10={:.4}, Recall@10={:.4}",
        best.2,
        1.0 - best.2,
        best.0,
        best.1
    );
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = parse_args();
    if !(0.0..=1.0).contains(&args.alpha) {
        return Err(Error::Other(format!(
            "--alpha must be between 0 and 1, got {}",
            args.alpha
        )));
    }
    if !(0.0..=1.0).contains(&args.rerank_weight) {
        return Err(Error::Other(format!(
            "--rerank-weight must be between 0 and 1, got {}",
            args.rerank_weight
        )));
    }
    if !(0.0..=1.0).contains(&args.prf_weight) {
        return Err(Error::Other(format!(
            "--prf-weight must be between 0 and 1, got {}",
            args.prf_weight
        )));
    }
    if args.rrf_k <= 0.0 {
        return Err(Error::Other(format!(
            "--rrf-k must be > 0, got {}",
            args.rrf_k
        )));
    }
    if args.alpha_normalizer <= 0.0 {
        return Err(Error::Other(format!(
            "--alpha-normalizer must be > 0, got {}",
            args.alpha_normalizer
        )));
    }

    let corpus_name = args
        .data_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("corpus")
        .to_string();

    // If cache_db was not explicitly set, derive from corpus name.
    let cache_db = if args.cache_db == PathBuf::from("test-data/nfcorpus-eval.sqlite") {
        PathBuf::from(format!("test-data/{corpus_name}-eval.sqlite"))
    } else {
        args.cache_db.clone()
    };

    let corpus_path = args.data_dir.join("corpus.jsonl");
    let queries_path = args.data_dir.join("queries.jsonl");
    let qrels_path = args.data_dir.join("qrels/test.tsv");

    for p in [&corpus_path, &queries_path, &qrels_path] {
        if !p.exists() {
            eprintln!(
                "error: {} not found.\n\
                 Download BEIR datasets from:\n  \
                 https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/",
                p.display()
            );
            std::process::exit(1);
        }
    }

    println!("loading corpus...");
    let mut corpus = load_corpus(&corpus_path)?;
    if let Some(max_docs) = args.max_docs {
        corpus.truncate(max_docs);
    }
    println!("  {} documents", corpus.len());
    let doc_texts: HashMap<String, String> =
        corpus.iter().map(|d| (d.id.clone(), doc_text(d))).collect();
    let term_stats = build_term_stats(&doc_texts);

    println!("loading queries...");
    let all_queries = load_queries(&queries_path)?;

    println!("loading qrels (test split)...");
    let qrels = load_qrels(&qrels_path)?;

    // Only evaluate queries that have test judgments
    let mut queries: Vec<Query> = all_queries
        .into_iter()
        .filter(|q| qrels.contains_key(&q.id))
        .collect();
    if let Some(max_queries) = args.max_queries {
        queries.truncate(max_queries);
    }
    println!("  {} queries with test judgments\n", queries.len());

    if let Some(chunk_size) = args.chunk_size_tokens {
        chunker::set_chunk_size_tokens_override(Some(chunk_size));
        println!("chunk size override: {chunk_size} tokens");
    }
    println!("cache DB: {}", cache_db.display());
    if let Some(pooling) = args.pooling {
        println!("embedding pooling override: {:?}", pooling);
    }
    println!("score-fusion alpha: {:.2}", args.alpha);
    if args.adaptive_alpha {
        println!(
            "adaptive alpha: enabled (normalizer={:.3})",
            args.alpha_normalizer
        );
    }
    if args.use_expander {
        println!(
            "expander: {:?}, rrf_k={:.1}, weights(lex/vec/hyde)={:.2}/{:.2}/{:.2}",
            args.expander_fusion,
            args.rrf_k,
            args.rrf_lex_weight,
            args.rrf_vec_weight,
            args.rrf_hyde_weight
        );
    }
    if args.use_prf {
        println!(
            "prf: enabled (weight={:.2}, docs={}, terms={})",
            args.prf_weight, args.prf_docs, args.prf_terms
        );
    }
    if args.use_rerank || args.tune_rerank {
        println!(
            "rerank blend: rerank={:.2}, fusion={:.2}, top_n={}",
            args.rerank_weight,
            1.0 - args.rerank_weight,
            args.rerank_top_n
        );
    }
    println!();

    println!("indexing corpus into cache DB ({})...", cache_db.display());
    ir::db::ensure_sqlite_vec();
    if let Some(parent) = cache_db.parent() {
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
    }
    let conn = Connection::open(&cache_db)?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous  = NORMAL;
         PRAGMA cache_size   = -64000;
         PRAGMA foreign_keys = ON;",
    )?;
    schema::init(&conn, &corpus_name)?;
    index_corpus(&conn, &corpus)?;
    println!("  done\n");

    let needs_embedder = matches!(
        args.mode,
        EvalMode::Vector | EvalMode::Hybrid | EvalMode::All
    );

    let embedder_opt: Option<Embedder> = if needs_embedder {
        match Embedder::load_default() {
            Ok(mut emb) => {
                if let Some(pooling) = args.pooling {
                    emb.set_pooling_override(Some(pooling));
                }
                println!("embedding model loaded");
                let embedding_dim = emb.embedding_dim();
                println!("embedding dim: {embedding_dim}");
                ensure_vector_dimension(&conn, embedding_dim)?;
                println!(
                    "embedding corpus ({} docs, may take several minutes)...",
                    corpus.len()
                );
                embed_corpus(&conn, &corpus, &emb)?;
                println!("  done\n");
                Some(emb)
            }
            Err(e) => {
                eprintln!(
                    "warning: embedding model not found ({e})\n\
                     Vector and Hybrid modes will be skipped.\n\
                     Set IR_EMBEDDING_MODEL (or QMD_EMBEDDING_MODEL) for '{}', or add a model directory to IR_MODEL_DIRS.\n",
                    models::EMBEDDING
                );
                None
            }
        }
    } else {
        None
    };

    let query_embeddings_opt: Option<HashMap<String, Vec<f32>>> =
        if let Some(ref emb) = embedder_opt {
            println!("embedding queries ({} queries)...", queries.len());
            let q = embed_queries(emb, &queries)?;
            println!("  done\n");
            Some(q)
        } else {
            None
        };

    let expander_opt: Option<Expander> = if args.use_expander {
        match Expander::load_default() {
            Ok(expander) => {
                println!("expander model loaded");
                Some(expander)
            }
            Err(e) => {
                eprintln!(
                    "warning: expander model not found ({e})\n\
                     Expander-enabled evaluations will fall back to base hybrid.\n\
                     Set IR_EXPANDER_MODEL (or QMD_EXPANDER_MODEL) for '{}', or add a model directory to IR_MODEL_DIRS.\n",
                    models::EXPANDER
                );
                None
            }
        }
    } else {
        None
    };

    let needs_reranker = args.use_rerank || args.tune_rerank;
    let reranker_opt: Option<Reranker> = if needs_reranker {
        match Reranker::load_default() {
            Ok(r) => {
                println!("reranker model loaded");
                Some(r)
            }
            Err(e) => {
                eprintln!(
                    "warning: reranker model not found ({e})\n\
                     Reranker-enabled evaluations will be skipped.\n\
                     Set IR_RERANKER_MODEL (or QMD_RERANKER_MODEL) for '{}', or add a model directory to IR_MODEL_DIRS.\n",
                    models::RERANKER
                );
                None
            }
        }
    } else {
        None
    };

    let hybrid_cfg = HybridConfig {
        alpha: args.alpha,
        adaptive_alpha: args.adaptive_alpha,
        alpha_normalizer: args.alpha_normalizer,
        use_expander: args.use_expander,
        expander_fusion: args.expander_fusion,
        rrf_k: args.rrf_k,
        rrf_weights: RrfWeights {
            lex: args.rrf_lex_weight,
            vec: args.rrf_vec_weight,
            hyde: args.rrf_hyde_weight,
        },
        prf: if args.use_prf {
            Some(PrfConfig {
                weight: args.prf_weight,
                top_docs: args.prf_docs,
                top_terms: args.prf_terms,
            })
        } else {
            None
        },
        rrf_no_expander: args.rrf_no_expander,
        title_weight: args.title_weight,
    };

    let k = args.limit;
    let mut results: Vec<EvalResult> = Vec::new();

    if matches!(args.mode, EvalMode::Bm25 | EvalMode::All) {
        print!("evaluating bm25 ({} queries)... ", queries.len());
        let _ = std::io::stdout().flush();
        results.push(evaluate_bm25(&conn, &queries, &qrels, k));
        println!("done");
    }

    if let Some(query_embeddings) = query_embeddings_opt.as_ref() {
        let embedder = embedder_opt
            .as_ref()
            .ok_or_else(|| Error::Other("embedder missing for vector/hybrid".to_string()))?;

        if matches!(args.mode, EvalMode::Vector | EvalMode::All) {
            print!("evaluating vector ({} queries)... ", queries.len());
            let _ = std::io::stdout().flush();
            results.push(evaluate_vector(
                &conn,
                &queries,
                &qrels,
                k,
                query_embeddings,
            ));
            println!("done");
        }

        if matches!(args.mode, EvalMode::Hybrid | EvalMode::All) {
            print!("evaluating hybrid ({} queries)... ", queries.len());
            let _ = std::io::stdout().flush();
            let reranker_ref = if args.use_rerank {
                reranker_opt.as_ref()
            } else {
                None
            };
            let mode_name = if reranker_ref.is_some() {
                "hybrid-rerank"
            } else {
                "hybrid"
            };
            results.push(evaluate_hybrid(
                &conn,
                &queries,
                &qrels,
                k,
                query_embeddings,
                embedder,
                expander_opt.as_ref(),
                &hybrid_cfg,
                &term_stats,
                reranker_ref,
                args.rerank_weight,
                args.rerank_top_n,
                &doc_texts,
                mode_name,
            ));
            println!("done");

            if args.tune_alpha {
                tune_score_fusion(
                    &conn,
                    &queries,
                    &qrels,
                    k,
                    query_embeddings,
                    embedder,
                    expander_opt.as_ref(),
                    &hybrid_cfg,
                    &doc_texts,
                    &term_stats,
                    reranker_ref,
                    args.rerank_weight,
                    args.rerank_top_n,
                );
            }

            if args.tune_rerank {
                if let Some(r) = reranker_opt.as_ref() {
                    tune_rerank_blend(
                        &conn,
                        &queries,
                        &qrels,
                        k,
                        query_embeddings,
                        embedder,
                        expander_opt.as_ref(),
                        &hybrid_cfg,
                        &term_stats,
                        r,
                        args.rerank_top_n,
                        &doc_texts,
                    );
                } else {
                    println!("\n── Reranker blend tuning ───────────────────────────────────────");
                    println!("skipped: reranker unavailable");
                }
            }
        }
    }

    // Results table
    println!("\n── NFCorpus evaluation (k={k}) ────────────────────────────────");
    println!(
        "{:<10}  {:>10}  {:>12}  {:>10}",
        "mode", "nDCG@10", "Recall@10", "queries"
    );
    println!("{}", "─".repeat(50));
    for r in &results {
        println!(
            "{:<10}  {:>10.4}  {:>12.4}  {:>10}",
            r.mode, r.ndcg, r.recall, r.n_queries
        );
    }
    println!("{}", "─".repeat(50));

    // Summary
    let get = |mode: &str| {
        results
            .iter()
            .find(|r| r.mode == mode)
            .map(|r| r.ndcg)
            .unwrap_or(0.0)
    };
    let bm25_ndcg = get("bm25");
    let vec_ndcg = get("vector");
    let hybrid_ndcg = get("hybrid").max(get("hybrid-rerank"));

    if hybrid_ndcg > 0.0 && (bm25_ndcg > 0.0 || vec_ndcg > 0.0) {
        let beats_bm25 = hybrid_ndcg > bm25_ndcg;
        let beats_vec = vec_ndcg == 0.0 || hybrid_ndcg > vec_ndcg;

        if beats_bm25 && beats_vec {
            let bm25_delta = if bm25_ndcg > 0.0 {
                (hybrid_ndcg - bm25_ndcg) / bm25_ndcg * 100.0
            } else {
                0.0
            };
            let vec_delta = if vec_ndcg > 0.0 {
                (hybrid_ndcg - vec_ndcg) / vec_ndcg * 100.0
            } else {
                0.0
            };
            println!(
                "hybrid beats bm25 (+{:.1}%) and vector (+{:.1}%) on nDCG@10",
                bm25_delta, vec_delta,
            );
        } else if beats_bm25 {
            println!(
                "hybrid beats bm25 (+{:.1}%) but not vector ({:+.1}%)",
                (hybrid_ndcg - bm25_ndcg) / bm25_ndcg * 100.0,
                (hybrid_ndcg - vec_ndcg) / vec_ndcg * 100.0,
            );
        } else {
            println!(
                "note: hybrid ({:.4}) does not beat all individual modes. \
                 consider tuning RRF k or fusion weights.",
                hybrid_ndcg,
            );
        }
    }

    Ok(())
}
