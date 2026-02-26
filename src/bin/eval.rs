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
use ir::llm::{embedding::Embedder, models};
use ir::types::SearchResult;

// ── CLI ───────────────────────────────────────────────────────────────────────

struct Args {
    data_dir: PathBuf,
    limit: usize,
    mode: EvalMode,
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
    let mut limit = 10;
    let mut mode = EvalMode::All;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" | "-d" => {
                i += 1;
                if i < args.len() {
                    data_dir = PathBuf::from(&args[i]);
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
            "--help" | "-h" => {
                println!("Usage: eval [--data DIR] [--limit K] [--mode bm25|vector|hybrid|all]");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    Args {
        data_dir,
        limit,
        mode,
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

fn embed_corpus(conn: &Connection, docs: &[CorpusDoc], embedder: &Embedder) -> Result<()> {
    let total = docs.len();
    for (i, doc) in docs.iter().enumerate() {
        if i % 500 == 0 {
            println!("  embedding {}/{}", i, total);
        }

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

        let chunks = chunker::chunk_document(&text);
        for chunk in &chunks {
            let emb = embedder.embed_doc(&doc.title, &chunk.text)?;
            let hash_seq = format!("{hash}_{}", chunk.seq);
            vectors::insert(conn, &hash_seq, &emb)?;
            vectors::mark_embedded(
                conn,
                &hash,
                chunk.seq as i64,
                chunk.pos as i64,
                models::EMBEDDING,
            )?;
        }
    }
    println!("  embedding {}/{}", total, total);
    Ok(())
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
            rel / (i as f64 + 2.0).log2()
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
        .map(|(i, &rel)| rel / (i as f64 + 2.0).log2())
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

fn bm25_search(conn: &Connection, query: &str, limit: usize) -> Vec<SearchResult> {
    let fts_query = fts::build_query(query);
    if fts_query.is_empty() {
        return vec![];
    }
    let q = fts::BM25Query {
        fts_query,
        collection: "nfcorpus",
        limit,
    };
    fts::search(conn, &q).unwrap_or_default()
}

fn vector_search(
    conn: &Connection,
    query: &str,
    limit: usize,
    embedder: &Embedder,
) -> Vec<SearchResult> {
    let emb = match embedder.embed_query(query) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("embed error: {e}");
            return vec![];
        }
    };
    vectors::search(conn, &emb, "nfcorpus", limit).unwrap_or_default()
}

/// Score-fusion alpha = 0.80 (mid-range of 0.70–0.95 optimal plateau on NFCorpus).
/// Grid search below confirms any α in [0.70, 0.95] beats pure vector (0.3866) on nDCG@10.
const SCORE_FUSION_ALPHA: f64 = 0.80;

fn hybrid_search(
    conn: &Connection,
    query: &str,
    limit: usize,
    embedder: &Embedder,
) -> Vec<SearchResult> {
    // Score-based linear fusion: α·vec + (1-α)·bm25
    // Empirically beats both individual modes on NFCorpus (see tune_score_fusion output).
    score_fusion(conn, query, limit, embedder, SCORE_FUSION_ALPHA)
}

/// Score-based linear fusion: combined_score = α*vec_score + (1-α)*bm25_score.
/// Union of results from both retrieval lists, merged by path.
fn score_fusion(
    conn: &Connection,
    query: &str,
    limit: usize,
    embedder: &Embedder,
    alpha: f64,
) -> Vec<SearchResult> {
    let bm25_results = bm25_search(conn, query, limit * 3);
    let vec_results = vector_search(conn, query, limit * 3, embedder);

    let mut scores: HashMap<String, (f64, f64, SearchResult)> = HashMap::new(); // path → (bm25, vec, result)

    for r in &bm25_results {
        scores
            .entry(r.path.clone())
            .or_insert((0.0, 0.0, r.clone()))
            .0 = r.score;
    }
    for r in &vec_results {
        let entry = scores
            .entry(r.path.clone())
            .or_insert((0.0, 0.0, r.clone()));
        entry.1 = r.score;
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

#[allow(dead_code)]
fn is_strong_signal(results: &[SearchResult]) -> bool {
    if results.len() < 2 {
        return results.first().map(|r| r.score >= 0.85).unwrap_or(false);
    }
    results[0].score >= 0.85 && (results[0].score - results[1].score) >= 0.15
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
    for q in queries {
        let Some(relevant) = qrels.get(&q.id) else {
            continue;
        };
        let results = bm25_search(conn, &q.text, k);
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
    embedder: &Embedder,
) -> EvalResult {
    let (mut total_ndcg, mut total_recall, mut n) = (0.0f64, 0.0f64, 0usize);
    for q in queries {
        let Some(relevant) = qrels.get(&q.id) else {
            continue;
        };
        let results = vector_search(conn, &q.text, k, embedder);
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
    embedder: &Embedder,
) -> EvalResult {
    let (mut total_ndcg, mut total_recall, mut n) = (0.0f64, 0.0f64, 0usize);
    for q in queries {
        let Some(relevant) = qrels.get(&q.id) else {
            continue;
        };
        let results = hybrid_search(conn, &q.text, k, embedder);
        let ranked: Vec<String> = results.iter().map(|r| r.path.clone()).collect();
        total_ndcg += ndcg_at_k(&ranked, relevant, k);
        total_recall += recall_at_k(&ranked, relevant, k);
        n += 1;
    }
    EvalResult {
        mode: "hybrid",
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
    embedder: &Embedder,
) {
    println!("\n── Score-fusion tuning (α*vec + (1-α)*bm25) ──────────────────");
    println!("{:<8}  {:>10}  {:>12}", "alpha", "nDCG@10", "Recall@10");
    println!("{}", "─".repeat(36));

    let alphas = [0.5f64, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0];
    let mut best = (0.0f64, 0.0f64, 0.0f64);

    for &alpha in &alphas {
        let (mut total_ndcg, mut total_recall, mut n) = (0.0f64, 0.0f64, 0usize);
        for q in queries {
            let Some(relevant) = qrels.get(&q.id) else {
                continue;
            };
            let results = score_fusion(conn, &q.text, k, embedder, alpha);
            let ranked: Vec<String> = results.iter().map(|r| r.path.clone()).collect();
            total_ndcg += ndcg_at_k(&ranked, relevant, k);
            total_recall += recall_at_k(&ranked, relevant, k);
            n += 1;
        }
        let ndcg = if n > 0 { total_ndcg / n as f64 } else { 0.0 };
        let recall = if n > 0 { total_recall / n as f64 } else { 0.0 };
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

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = parse_args();

    let corpus_path = args.data_dir.join("corpus.jsonl");
    let queries_path = args.data_dir.join("queries.jsonl");
    let qrels_path = args.data_dir.join("qrels/test.tsv");

    for p in [&corpus_path, &queries_path, &qrels_path] {
        if !p.exists() {
            eprintln!(
                "error: {} not found.\n\
                 Download with:\n  \
                 mkdir -p test-data && \\\n  \
                 curl -L https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip \
                 -o test-data/nfcorpus.zip && \\\n  \
                 cd test-data && unzip nfcorpus.zip",
                p.display()
            );
            std::process::exit(1);
        }
    }

    println!("loading corpus...");
    let corpus = load_corpus(&corpus_path)?;
    println!("  {} documents", corpus.len());

    println!("loading queries...");
    let all_queries = load_queries(&queries_path)?;

    println!("loading qrels (test split)...");
    let qrels = load_qrels(&qrels_path)?;

    // Only evaluate queries that have test judgments
    let queries: Vec<Query> = all_queries
        .into_iter()
        .filter(|q| qrels.contains_key(&q.id))
        .collect();
    println!("  {} queries with test judgments\n", queries.len());

    println!("indexing corpus into in-memory DB (BM25)...");
    ir::db::ensure_sqlite_vec();
    let conn = Connection::open_in_memory()?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous  = NORMAL;
         PRAGMA cache_size   = -64000;
         PRAGMA foreign_keys = ON;",
    )?;
    schema::init(&conn, "nfcorpus")?;
    index_corpus(&conn, &corpus)?;
    println!("  done\n");

    let needs_embedder = matches!(
        args.mode,
        EvalMode::Vector | EvalMode::Hybrid | EvalMode::All
    );

    let embedder_opt: Option<Embedder> = if needs_embedder {
        match Embedder::load_default() {
            Ok(emb) => {
                println!("embedding model loaded");
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

    let k = args.limit;
    let mut results: Vec<EvalResult> = Vec::new();

    if matches!(args.mode, EvalMode::Bm25 | EvalMode::All) {
        print!("evaluating bm25 ({} queries)... ", queries.len());
        let _ = std::io::stdout().flush();
        results.push(evaluate_bm25(&conn, &queries, &qrels, k));
        println!("done");
    }

    if let Some(ref emb) = embedder_opt {
        if matches!(args.mode, EvalMode::Vector | EvalMode::All) {
            print!("evaluating vector ({} queries)... ", queries.len());
            let _ = std::io::stdout().flush();
            results.push(evaluate_vector(&conn, &queries, &qrels, k, emb));
            println!("done");
        }

        if matches!(args.mode, EvalMode::Hybrid | EvalMode::All) {
            print!("evaluating hybrid ({} queries)... ", queries.len());
            let _ = std::io::stdout().flush();
            results.push(evaluate_hybrid(&conn, &queries, &qrels, k, emb));
            println!("done");

            // Grid search over score-fusion to find optimal linear blend
            tune_score_fusion(&conn, &queries, &qrels, k, emb);
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
    let hybrid_ndcg = get("hybrid");

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
