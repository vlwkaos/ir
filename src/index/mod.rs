pub mod chunker;
pub mod diff;
pub mod embed;
pub mod hasher;
pub mod scanner;

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

    // 1. Load current DB state: {rel_path → hash}. Force mode intentionally
    // presents an empty state, but does not mutate the DB until the scan succeeds.
    let stored: HashMap<String, String> = if force {
        HashMap::new()
    } else {
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
    let mut chain = if has_preprocessor && (n_add != 0 || n_update != 0) {
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
        if force {
            // A failed scan or preprocessing pass must not destroy the last good index.
            conn.execute_batch(
                "DELETE FROM documents_fts;
                 DELETE FROM documents;
                 DELETE FROM content;
                 DELETE FROM content_vectors;
                 DELETE FROM vectors_vec;
                 DELETE FROM llm_cache;",
            )?;
        } else {
            // ^ v0.15 and older retained tombstones. Purging them before inserts
            // makes a path that disappeared and returned self-healing.
            if has_preprocessor {
                conn.execute(
                    "DELETE FROM documents_fts
                     WHERE rowid IN (SELECT id FROM documents WHERE active = 0)",
                    [],
                )?;
            }
            conn.execute("DELETE FROM documents WHERE active = 0", [])?;
        }

        // 5. Delete removed files. Content and vectors remain hash-addressed caches.
        for rel_path in &d.to_deactivate {
            // ! Triggers disabled — must manually remove from FTS.
            delete_document(conn, rel_path, has_preprocessor)?;
            pb.inc(1);
            pb.set_message(format!("remove {rel_path}"));
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

            // ! Triggers disabled — must manually remove from FTS before delete.
            // ^ ON DELETE CASCADE removes document_metadata rows for this document
            delete_document(conn, rel_path, has_preprocessor)?;
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

        // Record the preprocessing state only after the matching FTS content is ready.
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('has_preprocessor', ?1)",
            [if has_preprocessor { "1" } else { "0" }],
        )?;

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

fn delete_document(
    conn: &rusqlite::Connection,
    rel_path: &str,
    has_preprocessor: bool,
) -> Result<()> {
    if has_preprocessor {
        let id: Option<i64> = conn
            .query_row(
                "SELECT id FROM documents WHERE path = ?1",
                [rel_path],
                |row| row.get(0),
            )
            .ok();
        if let Some(id) = id {
            conn.execute("DELETE FROM documents_fts WHERE rowid = ?1", [id])?;
        }
    }
    conn.execute("DELETE FROM documents WHERE path = ?1", [rel_path])?;
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    struct Fixture {
        _temp: TempDir,
        root: PathBuf,
        db: CollectionDb,
        collection: Collection,
        config: Config,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct CommittedIndex {
        documents: Vec<(String, String, String, i64)>,
        fts: Vec<(String, String)>,
    }

    impl Fixture {
        fn new() -> Self {
            fs::create_dir_all(".tmp").unwrap();
            let temp = tempfile::Builder::new()
                .prefix("index-update-test-")
                .tempdir_in(".tmp")
                .unwrap();
            let root = temp.path().join("collection");
            fs::create_dir(&root).unwrap();
            let db = CollectionDb::open("test", &temp.path().join("test.sqlite"), false).unwrap();
            let collection = Collection {
                name: "test".to_string(),
                path: root.to_string_lossy().into_owned(),
                globs: vec![],
                excludes: vec![],
                description: None,
                preprocessor: None,
                routing: None,
                retrieval: None,
            };
            Self {
                _temp: temp,
                root,
                db,
                collection,
                config: Config::default(),
            }
        }

        fn write(&self, path: &str, content: &str) {
            let absolute = self.root.join(path);
            if let Some(parent) = absolute.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(absolute, content).unwrap();
        }

        fn run_update(&self) -> crate::error::Result<(usize, usize, usize)> {
            self.run_update_with_force(false)
        }

        fn run_update_with_force(
            &self,
            force: bool,
        ) -> crate::error::Result<(usize, usize, usize)> {
            update(
                &self.db,
                &self.collection,
                &UpdateOptions { force },
                &self.config,
            )
        }
    }

    fn committed_index(db: &CollectionDb) -> CommittedIndex {
        let documents = {
            let mut stmt = db
                .conn()
                .prepare(
                    "SELECT d.path, d.hash, c.doc, d.active
                     FROM documents d
                     JOIN content c ON c.hash = d.hash
                     ORDER BY d.path",
                )
                .unwrap();
            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                })
                .unwrap();
            rows.collect::<rusqlite::Result<Vec<_>>>().unwrap()
        };
        let fts = {
            let mut stmt = db
                .conn()
                .prepare("SELECT path, body FROM documents_fts ORDER BY path")
                .unwrap();
            let rows = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .unwrap();
            rows.collect::<rusqlite::Result<Vec<_>>>().unwrap()
        };
        CommittedIndex { documents, fts }
    }

    fn retrieved_body(db: &CollectionDb, path: &str) -> Option<String> {
        crate::get::lookup_in_conn(db.conn(), &db.name, path)
            .unwrap()
            .map(|doc| doc.content)
    }

    fn bm25_paths(db: &CollectionDb, query: &str) -> Vec<String> {
        let query = crate::db::fts::BM25Query {
            fts_query: crate::db::fts::build_query(query),
            collection: &db.name,
            limit: 10,
            title_weight: None,
        };
        crate::db::fts::search(db.conn(), &query)
            .unwrap()
            .into_iter()
            .map(|result| result.path)
            .collect()
    }

    fn path_row_counts(db: &CollectionDb, path: &str) -> (i64, i64) {
        let document_count = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM documents WHERE path = ?1 AND active = 1",
                [path],
                |row| row.get(0),
            )
            .unwrap();
        let fts_count = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM documents_fts WHERE path = ?1",
                [path],
                |row| row.get(0),
            )
            .unwrap();
        (document_count, fts_count)
    }

    fn document_id(db: &CollectionDb, path: &str) -> i64 {
        db.conn()
            .query_row("SELECT id FROM documents WHERE path = ?1", [path], |row| {
                row.get(0)
            })
            .unwrap()
    }

    #[test]
    fn given_empty_collection_when_update_runs_then_no_documents_are_changed() {
        let fixture = Fixture::new();

        let counts = fixture.run_update().unwrap();

        assert_eq!(
            (counts, committed_index(&fixture.db)),
            (
                (0, 0, 0),
                CommittedIndex {
                    documents: vec![],
                    fts: vec![],
                }
            )
        );
    }

    #[test]
    fn given_new_path_when_update_runs_then_document_and_fts_are_committed() {
        let fixture = Fixture::new();
        fixture.write("new.md", "# New\n\nfirstneedle body");

        let counts = fixture.run_update().unwrap();

        assert_eq!(
            (
                counts,
                retrieved_body(&fixture.db, "new.md"),
                bm25_paths(&fixture.db, "firstneedle"),
                path_row_counts(&fixture.db, "new.md"),
            ),
            (
                (1, 0, 0),
                Some("# New\n\nfirstneedle body".to_string()),
                vec!["new.md".to_string()],
                (1, 1),
            )
        );
    }

    #[test]
    fn given_changed_active_path_when_update_runs_then_committed_body_is_replaced() {
        let fixture = Fixture::new();
        fixture.write("change.md", "# Before\n\noldneedle body");
        fixture.run_update().unwrap();
        fixture.write("change.md", "# After\n\nnewneedle body");

        let counts = fixture.run_update().unwrap();

        assert_eq!(
            (
                counts,
                retrieved_body(&fixture.db, "change.md"),
                bm25_paths(&fixture.db, "newneedle"),
                bm25_paths(&fixture.db, "oldneedle"),
                path_row_counts(&fixture.db, "change.md"),
            ),
            (
                (0, 1, 0),
                Some("# After\n\nnewneedle body".to_string()),
                vec!["change.md".to_string()],
                vec![],
                (1, 1),
            )
        );
    }

    #[test]
    fn given_unchanged_path_when_update_runs_again_then_content_remains_retrievable() {
        let fixture = Fixture::new();
        fixture.write("same.md", "# Same\n\nunchangedneedle body");
        fixture.run_update().unwrap();

        let counts = fixture.run_update().unwrap();

        assert_eq!(
            (
                counts,
                retrieved_body(&fixture.db, "same.md"),
                bm25_paths(&fixture.db, "unchangedneedle"),
            ),
            (
                (0, 0, 0),
                Some("# Same\n\nunchangedneedle body".to_string()),
                vec!["same.md".to_string()],
            )
        );
    }

    #[test]
    fn given_removed_then_recreated_path_when_update_runs_then_path_returns_once() {
        let fixture = Fixture::new();
        fixture.write("return.md", "# Return\n\nreturnneedle body");
        fixture.run_update().unwrap();
        fs::remove_file(fixture.root.join("return.md")).unwrap();

        let removed_counts = fixture.run_update().unwrap();
        let state_while_missing = (
            retrieved_body(&fixture.db, "return.md"),
            bm25_paths(&fixture.db, "returnneedle"),
            path_row_counts(&fixture.db, "return.md"),
        );
        fixture.write("return.md", "# Return\n\nreturnneedle body");
        let returned_counts = fixture.run_update().unwrap();

        assert_eq!(
            (
                removed_counts,
                state_while_missing,
                returned_counts,
                retrieved_body(&fixture.db, "return.md"),
                bm25_paths(&fixture.db, "returnneedle"),
                path_row_counts(&fixture.db, "return.md"),
            ),
            (
                (0, 0, 1),
                (None, vec![], (0, 0)),
                (1, 0, 0),
                Some("# Return\n\nreturnneedle body".to_string()),
                vec!["return.md".to_string()],
                (1, 1),
            )
        );
    }

    #[test]
    fn given_same_content_moved_to_new_path_when_update_runs_then_old_path_is_absent() {
        let fixture = Fixture::new();
        fixture.write("old.md", "# Moved\n\nmovedneedle body");
        fixture.run_update().unwrap();
        fs::rename(fixture.root.join("old.md"), fixture.root.join("new.md")).unwrap();

        let counts = fixture.run_update().unwrap();

        assert_eq!(
            (
                counts,
                retrieved_body(&fixture.db, "old.md"),
                retrieved_body(&fixture.db, "new.md"),
                bm25_paths(&fixture.db, "movedneedle"),
            ),
            (
                (1, 0, 1),
                None,
                Some("# Moved\n\nmovedneedle body".to_string()),
                vec!["new.md".to_string()],
            )
        );
    }

    #[test]
    fn given_inactive_legacy_row_for_present_path_when_update_runs_then_path_self_heals() {
        let fixture = Fixture::new();
        fixture
            .db
            .conn()
            .execute(
                "INSERT INTO content (hash, doc, created_at)
                 VALUES ('legacy-hash', '# Legacy\n\nlegacyphantom', '2024-01-01T00:00:00Z')",
                [],
            )
            .unwrap();
        fixture
            .db
            .conn()
            .execute(
                "INSERT INTO documents
                    (path, title, hash, created_at, modified_at, active)
                 VALUES
                    ('legacy.md', 'Legacy', 'legacy-hash',
                     '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', 0)",
                [],
            )
            .unwrap();
        let stale_precondition: (i64, String) = fixture
            .db
            .conn()
            .query_row(
                "SELECT d.active, f.body
                 FROM documents d
                 JOIN documents_fts f ON f.rowid = d.id
                 WHERE d.path = 'legacy.md'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        fixture.write("legacy.md", "# Returned\n\ncurrentneedle body");

        let counts = fixture.run_update().unwrap();

        assert_eq!(
            (
                stale_precondition,
                counts,
                retrieved_body(&fixture.db, "legacy.md"),
                bm25_paths(&fixture.db, "currentneedle"),
                bm25_paths(&fixture.db, "legacyphantom"),
                path_row_counts(&fixture.db, "legacy.md"),
            ),
            (
                (0, "# Legacy\n\nlegacyphantom".to_string()),
                (1, 0, 0),
                Some("# Returned\n\ncurrentneedle body".to_string()),
                vec!["legacy.md".to_string()],
                vec![],
                (1, 1),
            )
        );
    }

    #[test]
    fn given_missing_collection_root_when_update_fails_then_last_commit_is_preserved() {
        let mut fixture = Fixture::new();
        fixture.write("stable.md", "# Stable\n\ncommitted body");
        fixture.run_update().unwrap();
        let before = committed_index(&fixture.db);
        fixture.collection.path = fixture
            .root
            .join("does-not-exist")
            .to_string_lossy()
            .into_owned();

        let result = fixture.run_update();

        assert_eq!(
            (result.is_err(), committed_index(&fixture.db)),
            (true, before)
        );
    }

    #[test]
    fn given_delete_change_and_add_when_apply_fails_then_whole_batch_rolls_back() {
        let fixture = Fixture::new();
        fixture.write("removed.md", "# Removed\n\nremovedcommitted");
        fixture.write("changed.md", "# Changed\n\nchangedcommitted");
        fixture.run_update().unwrap();
        let before = committed_index(&fixture.db);
        fs::remove_file(fixture.root.join("removed.md")).unwrap();
        fixture.write("changed.md", "# Changed\n\nchangeduncommitted");
        fixture.write("inserted.md", "# Inserted\n\ninserteduncommitted");
        fixture
            .db
            .conn()
            .execute_batch(
                "CREATE TRIGGER reject_changed_test_document
                 BEFORE INSERT ON documents
                 WHEN NEW.path = 'changed.md'
                 BEGIN
                     SELECT RAISE(ABORT, 'test rejection');
                 END;",
            )
            .unwrap();

        let result = fixture.run_update();

        assert_eq!(
            (result.is_err(), committed_index(&fixture.db)),
            (true, before)
        );
    }

    #[test]
    fn given_unchanged_path_when_force_update_runs_then_document_is_reindexed() {
        let fixture = Fixture::new();
        fixture.write("force.md", "# Force\n\nforceneedle body");
        fixture.run_update().unwrap();
        let original_id = document_id(&fixture.db, "force.md");

        let counts = fixture.run_update_with_force(true).unwrap();
        let replacement_id = document_id(&fixture.db, "force.md");

        assert_eq!(
            (
                counts,
                replacement_id > original_id,
                retrieved_body(&fixture.db, "force.md"),
                bm25_paths(&fixture.db, "forceneedle"),
                path_row_counts(&fixture.db, "force.md"),
            ),
            (
                (1, 0, 0),
                true,
                Some("# Force\n\nforceneedle body".to_string()),
                vec!["force.md".to_string()],
                (1, 1),
            )
        );
    }

    #[test]
    fn given_committed_index_and_missing_root_when_force_update_fails_then_public_state_is_unchanged()
     {
        let mut fixture = Fixture::new();
        fixture.write("stable.md", "# Stable\n\nforcerollbackneedle body");
        fixture.run_update().unwrap();
        let before = (
            retrieved_body(&fixture.db, "stable.md"),
            bm25_paths(&fixture.db, "forcerollbackneedle"),
        );
        fixture.collection.path = fixture
            .root
            .join("missing-force-root")
            .to_string_lossy()
            .into_owned();

        let result = fixture.run_update_with_force(true);
        let after = (
            retrieved_body(&fixture.db, "stable.md"),
            bm25_paths(&fixture.db, "forcerollbackneedle"),
        );

        assert_eq!((result.is_err(), after), (true, before));
    }

    #[test]
    fn given_raw_index_when_reopened_with_cat_preprocessor_then_state_change_reindexes_atomically()
    {
        let mut fixture = Fixture::new();
        fixture.write("compat.md", "# Compatible\n\ncompatibilityneedle body");
        fixture.run_update().unwrap();
        let db_path = fixture._temp.path().join("test.sqlite");
        let reopened = CollectionDb::open("test", &db_path, true).unwrap();
        let meta_before_update: String = reopened
            .conn()
            .query_row(
                "SELECT value FROM meta WHERE key = 'has_preprocessor'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        fixture
            .config
            .preprocessors
            .insert("test-cat".to_string(), "cat".to_string());
        fixture.collection.preprocessor = Some(vec!["test-cat".to_string()]);

        let counts = update(
            &reopened,
            &fixture.collection,
            &UpdateOptions { force: false },
            &fixture.config,
        )
        .unwrap();
        let committed_meta_and_fts: (String, String) = reopened
            .conn()
            .query_row(
                "SELECT m.value, f.body
                 FROM meta m
                 JOIN documents d ON d.path = 'compat.md' AND d.active = 1
                 JOIN documents_fts f ON f.rowid = d.id
                 WHERE m.key = 'has_preprocessor'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();

        assert_eq!(
            (
                meta_before_update,
                counts,
                retrieved_body(&reopened, "compat.md"),
                bm25_paths(&reopened, "compatibilityneedle"),
                committed_meta_and_fts,
                path_row_counts(&reopened, "compat.md"),
            ),
            (
                "0".to_string(),
                (1, 0, 0),
                Some("# Compatible\n\ncompatibilityneedle body".to_string()),
                vec!["compat.md".to_string()],
                (
                    "1".to_string(),
                    "# Compatible\n\ncompatibilityneedle body".to_string(),
                ),
                (1, 1),
            )
        );
    }

    #[test]
    fn given_cached_vector_when_identical_document_reappears_then_cache_remains_reusable() {
        let fixture = Fixture::new();
        let content = "# Cached\n\ncacheuseneedle body";
        fixture.write("cached.md", content);
        fixture.run_update().unwrap();
        let hash = hasher::hash_bytes(content.as_bytes());
        let hash_seq = format!("{hash}_0");
        let mut embedding = vec![0.0_f32; 768];
        embedding[0] = 1.0;
        crate::db::vectors::insert(fixture.db.conn(), &hash_seq, &embedding).unwrap();
        crate::db::vectors::mark_embedded(fixture.db.conn(), &hash, 0, 0, "test-model").unwrap();
        let pending_before = crate::index::embed::pending_count(fixture.db.conn(), false).unwrap();
        fs::remove_file(fixture.root.join("cached.md")).unwrap();

        let removed_counts = fixture.run_update().unwrap();
        let pending_while_missing =
            crate::index::embed::pending_count(fixture.db.conn(), false).unwrap();
        let vector_while_missing = crate::db::vectors::knn(fixture.db.conn(), &embedding, 1)
            .unwrap()
            .into_iter()
            .map(|result| result.hash_seq)
            .collect::<Vec<_>>();
        fixture.write("cached.md", content);
        let returned_counts = fixture.run_update().unwrap();
        let pending_after = crate::index::embed::pending_count(fixture.db.conn(), false).unwrap();
        let vector_after = crate::db::vectors::knn(fixture.db.conn(), &embedding, 1)
            .unwrap()
            .into_iter()
            .map(|result| result.hash_seq)
            .collect::<Vec<_>>();
        let mapping_count: i64 = fixture
            .db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM content_vectors WHERE hash = ?1 AND seq = 0",
                [&hash],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(
            (
                pending_before,
                removed_counts,
                pending_while_missing,
                vector_while_missing,
                returned_counts,
                pending_after,
                vector_after,
                mapping_count,
                retrieved_body(&fixture.db, "cached.md"),
            ),
            (
                0,
                (0, 0, 1),
                0,
                vec![hash_seq.clone()],
                (1, 0, 0),
                0,
                vec![hash_seq],
                1,
                Some(content.to_string()),
            )
        );
    }

    #[test]
    fn given_cat_preprocessed_path_when_removed_and_restored_then_manual_fts_state_self_heals() {
        let mut fixture = Fixture::new();
        let db_path = fixture._temp.path().join("test.sqlite");
        let preprocessed_db = CollectionDb::open("test", &db_path, true).unwrap();
        fixture
            .config
            .preprocessors
            .insert("test-cat".to_string(), "cat".to_string());
        fixture.collection.preprocessor = Some(vec!["test-cat".to_string()]);
        let content = "# Preprocessed\n\npreprocessedreturnneedle body";
        fixture.write("preprocessed.md", content);
        update(
            &preprocessed_db,
            &fixture.collection,
            &UpdateOptions { force: false },
            &fixture.config,
        )
        .unwrap();
        fs::remove_file(fixture.root.join("preprocessed.md")).unwrap();

        let removed_counts = update(
            &preprocessed_db,
            &fixture.collection,
            &UpdateOptions { force: false },
            &fixture.config,
        )
        .unwrap();
        let removed_state = (
            retrieved_body(&preprocessed_db, "preprocessed.md"),
            bm25_paths(&preprocessed_db, "preprocessedreturnneedle"),
            path_row_counts(&preprocessed_db, "preprocessed.md"),
        );
        fixture.write("preprocessed.md", content);
        let restored_counts = update(
            &preprocessed_db,
            &fixture.collection,
            &UpdateOptions { force: false },
            &fixture.config,
        )
        .unwrap();

        assert_eq!(
            (
                removed_counts,
                removed_state,
                restored_counts,
                retrieved_body(&preprocessed_db, "preprocessed.md"),
                bm25_paths(&preprocessed_db, "preprocessedreturnneedle"),
                path_row_counts(&preprocessed_db, "preprocessed.md"),
            ),
            (
                (0, 0, 1),
                (None, vec![], (0, 0)),
                (1, 0, 0),
                Some(content.to_string()),
                vec!["preprocessed.md".to_string()],
                (1, 1),
            )
        );
    }
}
