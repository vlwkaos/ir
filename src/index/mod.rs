pub mod chunker;
pub mod diff;
pub mod embed;
pub mod hasher;
pub mod scanner;
pub mod units;

use crate::config::Config;
use crate::db::CollectionDb;
use crate::error::Result;
use crate::preprocess::PreprocessChain;
use crate::types::Collection;
use chrono::Utc;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;

pub struct UpdateOptions {
    pub force: bool,
}

pub fn new_progress_bar(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {per_sec} {msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb
}

/// Scan, diff, and update the FTS index for a collection.
/// Returns (added, updated, deactivated) counts.
pub fn update(
    db: &CollectionDb,
    collection: &Collection,
    opts: &UpdateOptions,
    config: &Config,
) -> Result<(usize, usize, usize)> {
    let conn = db.conn();

    // Resolve preprocessor aliases to command strings.
    let pp_aliases = collection.preprocessor.as_deref().unwrap_or(&[]);
    let pp_commands = config.resolve_preprocessor_commands(pp_aliases);
    let has_preprocessor = !pp_commands.is_empty();

    // Check stored has_preprocessor to detect migration.
    let stored_has_preprocessor: bool = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'has_preprocessor'",
            [],
            |row| row.get::<_, String>(0),
        )
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false);

    let force = opts.force || (stored_has_preprocessor != has_preprocessor);
    if !opts.force && stored_has_preprocessor != has_preprocessor {
        eprintln!(
            "preprocessor config changed (was={stored_has_preprocessor}, now={has_preprocessor}) — forcing re-index"
        );
    }

    // 1. Load current DB state: {rel_path → hash}
    if force {
        // Wipe existing data so everything is re-indexed from scratch.
        conn.execute_batch(
            "DELETE FROM documents;
             DELETE FROM content;
             DELETE FROM content_vectors;
             DELETE FROM vectors_vec;
             DELETE FROM llm_cache;
             DELETE FROM content_units;
             DELETE FROM unit_links;",
        )?;
        // Re-init schema with correct trigger state.
        crate::db::schema::init(conn, &db.name, has_preprocessor)?;
    }
    let stored: HashMap<String, String> = {
        let mut stmt = conn.prepare("SELECT path, hash FROM documents WHERE active = 1")?;
        stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<std::result::Result<HashMap<_, _>, _>>()?
    };

    // 2. Scan filesystem
    let scanned_files = scanner::scan(collection)?;

    // 3. Hash scanned files: {rel_path → (hash, content_bytes, mtime_rfc3339, birthtime_rfc3339)}
    // Progress bar runs during hashing so operators can see activity from the first second.
    let pb = new_progress_bar(scanned_files.len() as u64);
    pb.set_message("hashing");
    let mut scanned: HashMap<String, (String, Vec<u8>, String, String)> =
        HashMap::with_capacity(scanned_files.len());
    for f in &scanned_files {
        let content = std::fs::read(&f.abs_path)?;
        let hash = hasher::hash_bytes(&content);
        // ^ use filesystem timestamps so date filters work correctly.
        //   birthtime falls back to mtime on Linux filesystems that don't track it.
        let now = Utc::now();
        let mtime = f.mtime.map(chrono::DateTime::<Utc>::from).unwrap_or(now);
        let birthtime = f
            .birthtime
            .map(chrono::DateTime::<Utc>::from)
            .unwrap_or(mtime);
        scanned.insert(
            f.rel_path.clone(),
            (hash, content, mtime.to_rfc3339(), birthtime.to_rfc3339()),
        );
        pb.inc(1);
    }

    // 4. Compute diff — pass hash-only view
    pb.set_message("diffing");
    let hash_only: HashMap<String, String> = scanned
        .iter()
        .map(|(path, (hash, _, _, _))| (path.clone(), hash.clone()))
        .collect();
    let d = diff::compute(&hash_only, &stored);
    let (n_add, n_update, n_deactivate) =
        (d.to_add.len(), d.to_update.len(), d.to_deactivate.len());

    pb.set_position(0);
    pb.set_length((n_add + n_update + n_deactivate) as u64);
    pb.set_message("applying");

    // Spawn preprocessor chain once for the whole batch.
    let mut chain = if has_preprocessor {
        let c = PreprocessChain::spawn(&pp_commands);
        if !c.is_active() {
            eprintln!("warning: all preprocessors failed to spawn — indexing raw text");
        }
        Some(c)
    } else {
        None
    };

    // 5–7. Apply diff atomically so a crash leaves the DB consistent.
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let mut apply = || -> Result<()> {
        // 5. Deactivate removed files
        for rel_path in &d.to_deactivate {
            if has_preprocessor {
                // ! Triggers disabled — must manually remove from FTS.
                let id: Option<i64> = conn
                    .query_row(
                        "SELECT id FROM documents WHERE path = ?1",
                        [rel_path],
                        |r| r.get(0),
                    )
                    .ok();
                conn.execute(
                    "UPDATE documents SET active = 0 WHERE path = ?1",
                    [rel_path],
                )?;
                if let Some(id) = id {
                    conn.execute("DELETE FROM documents_fts WHERE rowid = ?1", [id])?;
                }
            } else {
                conn.execute(
                    "UPDATE documents SET active = 0 WHERE path = ?1",
                    [rel_path],
                )?;
            }
            pb.inc(1);
            pb.set_message(format!("deactivate {rel_path}"));
        }

        // 6. Add new files
        for rel_path in &d.to_add {
            let (hash, content, file_mtime, file_birthtime) =
                scanned.get(rel_path).ok_or_else(|| {
                    crate::error::Error::Other(format!("missing scan entry: {rel_path}"))
                })?;
            let raw_text = String::from_utf8_lossy(content).into_owned();
            let text = raw_text.replace("\r\n", "\n");
            let title = chunker::extract_title(&text, rel_path);

            store_document(
                conn,
                rel_path,
                &title,
                hash,
                &text,
                file_birthtime,
                file_mtime,
                chain.as_mut(),
            )?;
            pb.inc(1);
            pb.set_message(format!("add {rel_path}"));
        }

        // 7. Update changed files
        for rel_path in &d.to_update {
            let (hash, content, file_mtime, _file_birthtime) =
                scanned.get(rel_path).ok_or_else(|| {
                    crate::error::Error::Other(format!("missing scan entry: {rel_path}"))
                })?;
            let raw_text = String::from_utf8_lossy(content).into_owned();
            let text = raw_text.replace("\r\n", "\n");
            let title = chunker::extract_title(&text, rel_path);
            let created_at: String = conn
                .query_row(
                    "SELECT created_at FROM documents WHERE path = ?1",
                    [rel_path],
                    |row| row.get(0),
                )
                .unwrap_or_else(|_| file_mtime.clone());

            if has_preprocessor {
                // ! Triggers disabled — must manually remove from FTS before delete.
                let id: Option<i64> = conn
                    .query_row(
                        "SELECT id FROM documents WHERE path = ?1",
                        [rel_path],
                        |r| r.get(0),
                    )
                    .ok();
                if let Some(id) = id {
                    conn.execute("DELETE FROM documents_fts WHERE rowid = ?1", [id])?;
                }
            }
            conn.execute("DELETE FROM documents WHERE path = ?1", [rel_path])?;
            // ^ ON DELETE CASCADE removes document_metadata rows for this document
            store_document(
                conn,
                rel_path,
                &title,
                hash,
                &text,
                &created_at,
                file_mtime,
                chain.as_mut(),
            )?;
            pb.inc(1);
            pb.set_message(format!("update {rel_path}"));
        }

        Ok(())
    };
    match apply() {
        Ok(()) => conn.execute_batch("COMMIT")?,
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            return Err(e);
        }
    }

    pb.finish_with_message("done");
    Ok((n_add, n_update, n_deactivate))
}

#[allow(clippy::too_many_arguments)]
fn store_document(
    conn: &rusqlite::Connection,
    rel_path: &str,
    title: &str,
    hash: &str,
    text: &str,
    created_at: &str,
    modified_at: &str,
    chain: Option<&mut PreprocessChain>,
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

    let doc_id = conn.last_insert_rowid();

    // Extract and store frontmatter metadata
    if let Some(mapping) = chunker::extract_frontmatter(text) {
        for (key, value) in chunker::flatten_frontmatter(&mapping) {
            conn.execute(
                "INSERT OR IGNORE INTO document_metadata (document_id, key, value) \
                 VALUES (?1, ?2, ?3)",
                rusqlite::params![doc_id, key, value],
            )?;
        }
    }

    units::store_units(conn, doc_id, rel_path, hash, text)?;

    // When chain is active, triggers are disabled — explicitly insert preprocessed text into FTS.
    if let Some(chain) = chain
        && chain.is_active()
    {
        let processed = chain.process_text(text)?;
        conn.execute(
            "INSERT INTO documents_fts(rowid, path, title, body) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![doc_id, rel_path, title, processed],
        )?;
    }

    Ok(())
}
