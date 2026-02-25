pub mod chunker;
pub mod diff;
pub mod hasher;
pub mod scanner;

use crate::db::CollectionDb;
use crate::error::Result;
use crate::types::Collection;
use chrono::Utc;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;

pub struct UpdateOptions {
    pub force: bool,
}

/// Scan, diff, and update the FTS index for a collection.
/// Returns (added, updated, deactivated) counts.
pub fn update(db: &CollectionDb, collection: &Collection, opts: &UpdateOptions) -> Result<(usize, usize, usize)> {
    let conn = db.conn();

    // 1. Load current DB state: {rel_path → hash}
    if opts.force {
        // Wipe existing data so everything is re-indexed from scratch.
        conn.execute_batch(
            "DELETE FROM documents;
             DELETE FROM content;
             DELETE FROM content_vectors;
             DELETE FROM llm_cache;",
        )?;
    }
    let stored: HashMap<String, String> = {
        let mut stmt = conn.prepare("SELECT path, hash FROM documents WHERE active = 1")?;
        stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?
            .filter_map(|r| r.ok())
            .collect()
    };

    // 2. Scan filesystem
    let scanned_files = scanner::scan(collection)?;
    let pb = ProgressBar::new(scanned_files.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    // 3. Hash scanned files: {rel_path → hash}
    let mut scanned: HashMap<String, String> = HashMap::with_capacity(scanned_files.len());
    for f in &scanned_files {
        let content = std::fs::read(&f.abs_path)?;
        let hash = hasher::hash_bytes(&content);
        scanned.insert(f.rel_path.clone(), hash);
    }

    // 4. Compute diff
    let d = diff::compute(&scanned, &stored);
    let (n_add, n_update, n_deactivate) = (d.to_add.len(), d.to_update.len(), d.to_deactivate.len());

    pb.set_length((n_add + n_update + n_deactivate) as u64);

    // Build a rel_path → abs_path lookup
    let path_lookup: HashMap<String, _> = scanned_files
        .iter()
        .map(|f| (f.rel_path.clone(), &f.abs_path))
        .collect();

    // 5. Deactivate removed files
    for rel_path in &d.to_deactivate {
        conn.execute(
            "UPDATE documents SET active = 0 WHERE path = ?1",
            [rel_path],
        )?;
        pb.inc(1);
        pb.set_message(format!("deactivate {rel_path}"));
    }

    // 6. Add new files
    for rel_path in &d.to_add {
        let abs_path = path_lookup[rel_path];
        let content = std::fs::read(abs_path)?;
        let text = String::from_utf8_lossy(&content).into_owned();
        let hash = &scanned[rel_path];
        let title = chunker::extract_title(&text, rel_path);
        let now = Utc::now().to_rfc3339();

        store_document(conn, rel_path, &title, hash, &text, &now, &now)?;
        pb.inc(1);
        pb.set_message(format!("add {rel_path}"));
    }

    // 7. Update changed files
    for rel_path in &d.to_update {
        let abs_path = path_lookup[rel_path];
        let content = std::fs::read(abs_path)?;
        let text = String::from_utf8_lossy(&content).into_owned();
        let hash = &scanned[rel_path];
        let title = chunker::extract_title(&text, rel_path);
        let now = Utc::now().to_rfc3339();
        let created_at: String = conn
            .query_row(
                "SELECT created_at FROM documents WHERE path = ?1",
                [rel_path],
                |row| row.get(0),
            )
            .unwrap_or_else(|_| now.clone());

        conn.execute("DELETE FROM documents WHERE path = ?1", [rel_path])?;
        store_document(conn, rel_path, &title, hash, &text, &created_at, &now)?;
        pb.inc(1);
        pb.set_message(format!("update {rel_path}"));
    }

    pb.finish_with_message("done");
    Ok((n_add, n_update, n_deactivate))
}

fn store_document(
    conn: &rusqlite::Connection,
    rel_path: &str,
    title: &str,
    hash: &str,
    text: &str,
    created_at: &str,
    modified_at: &str,
) -> Result<()> {
    // Upsert content (content-addressed, may already exist from another file)
    conn.execute(
        "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?1, ?2, ?3)",
        rusqlite::params![hash, text, created_at],
    )?;

    conn.execute(
        "INSERT INTO documents (path, title, hash, created_at, modified_at, active)
         VALUES (?1, ?2, ?3, ?4, ?5, 1)",
        rusqlite::params![rel_path, title, hash, created_at, modified_at],
    )?;

    Ok(())
}
