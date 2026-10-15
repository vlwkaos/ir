pub mod chunker;
pub mod diff;
pub mod embed;
pub mod hasher;
pub mod scanner;

use crate::config::Config;
use crate::db::{self, CollectionDb};
use crate::error::{Error, Result};
use crate::preprocess::PreprocessChain;
use crate::types::Collection;
use chrono::Utc;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet};

pub struct UpdateOptions {
    pub force: bool,
}

/// Outcome of one `update` run. `scanned` is the number of files the scan
/// matched, so a 0/0/0 diff can be told apart from a scan that found nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateStats {
    pub scanned: usize,
    pub added: usize,
    pub updated: usize,
    pub deactivated: usize,
    /// Active documents without vectors after this run. `update` never embeds.
    pub pending_embeddings: usize,
    /// Warnings for the user: the CLI prints them to stderr, MCP returns them.
    pub notes: Vec<String>,
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

/// {rel_path → (hash, content_bytes, mtime_rfc3339, birthtime_rfc3339)}
type Scanned = HashMap<String, (String, Vec<u8>, String, String)>;

fn scan_entry<'a>(
    scanned: &'a Scanned,
    rel_path: &str,
) -> Result<&'a (String, Vec<u8>, String, String)> {
    scanned
        .get(rel_path)
        .ok_or_else(|| Error::Other(format!("missing scan entry: {rel_path}")))
}

fn decode(content: &[u8]) -> String {
    String::from_utf8_lossy(content).replace("\r\n", "\n")
}

/// What one run changes, derived from the stored state.
struct Plan {
    /// Rebuild everything: `--force`, or the preprocessor binding changed.
    force: bool,
    preprocessor_changed: bool,
    diff: diff::IndexDiff,
}

fn plan(
    conn: &rusqlite::Connection,
    opts: &UpdateOptions,
    has_preprocessor: bool,
    hash_only: &HashMap<String, String>,
) -> Result<Plan> {
    let stored_has_preprocessor = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'has_preprocessor'",
            [],
            |row| row.get::<_, String>(0),
        )
        .is_ok_and(|v| v == "1");
    let preprocessor_changed = stored_has_preprocessor != has_preprocessor;
    let force = opts.force || preprocessor_changed;
    // Force mode presents an empty stored state.
    let stored = if force {
        HashMap::new()
    } else {
        stored_hashes(conn)?
    };
    Ok(Plan {
        force,
        preprocessor_changed,
        diff: diff::compute(hash_only, &stored),
    })
}

/// Preprocessed search text per content hash, for collections with a preprocessor.
/// The chain is spawned once, on the first document that needs it.
/// ^ A preprocessor that cannot start or dies does not stop the update: the
///   document is indexed with its raw text and listed in `raw`, and the next
///   update stores it again. Queries fall back to raw text the same way.
struct FtsBodies<'a> {
    commands: &'a [String],
    chain: ChainState,
    by_hash: HashMap<String, String>,
    raw: HashSet<String>,
    /// Why the first document fell back to raw text, for the update's notes.
    first_error: Option<String>,
}

impl FtsBodies<'_> {
    /// Preprocess every document `diff` stores that is not prepared yet.
    fn prepare(
        &mut self,
        diff: &diff::IndexDiff,
        scanned: &Scanned,
        pb: &ProgressBar,
    ) -> Result<()> {
        if self.commands.is_empty() {
            return Ok(());
        }
        for rel_path in diff.to_add.iter().chain(&diff.to_update) {
            let (hash, content, _, _) = scan_entry(scanned, rel_path)?;
            if !self.by_hash.contains_key(hash) {
                let text = decode(content);
                let body = match self.preprocess(&text, rel_path) {
                    Some(body) => body,
                    None => {
                        self.raw.insert(hash.clone());
                        text
                    }
                };
                self.by_hash.insert(hash.clone(), body);
            }
            pb.inc(1);
        }
        Ok(())
    }

    /// None when the chain cannot start or fails on this document. A chain that
    /// failed is started again for the next document.
    fn preprocess(&mut self, text: &str, rel_path: &str) -> Option<String> {
        if let ChainState::NotStarted = self.chain {
            self.chain = match PreprocessChain::spawn_all(self.commands) {
                Ok(chain) => ChainState::Running(chain),
                Err(e) => {
                    self.first_error.get_or_insert(e.to_string());
                    ChainState::CannotStart
                }
            };
        }
        let ChainState::Running(chain) = &mut self.chain else {
            return None;
        };
        match chain.process_text(text) {
            Ok(body) => Some(body),
            Err(e) => {
                self.first_error.get_or_insert(format!("{e} on {rel_path}"));
                self.chain = ChainState::NotStarted;
                None
            }
        }
    }
}

enum ChainState {
    NotStarted,
    Running(PreprocessChain),
    /// Spawning failed once in this update; not retried until the next one.
    CannotStart,
}

/// Scan, diff, and update the FTS index for a collection.
pub fn update(
    db: &CollectionDb,
    collection: &Collection,
    opts: &UpdateOptions,
    config: &Config,
) -> Result<UpdateStats> {
    let conn = db.conn();

    // Resolve preprocessor aliases to command strings.
    let pp_aliases = collection.preprocessor.as_deref().unwrap_or(&[]);
    let pp_commands = config.resolve_preprocessor_commands(pp_aliases);
    let has_preprocessor = !pp_commands.is_empty();

    // 1. Scan and hash the filesystem. No lock is held here; this is the slow
    // part. The progress bar runs during hashing so operators can see activity
    // from the first second.
    let scanned_files = scanner::scan(collection)?;
    let pb = new_progress_bar(scanned_files.len() as u64);
    pb.set_message("hashing");
    let mut scanned: Scanned = HashMap::with_capacity(scanned_files.len());
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
    let hash_only: HashMap<String, String> = scanned
        .iter()
        .map(|(path, (hash, _, _, _))| (path.clone(), hash.clone()))
        .collect();

    // 2. Preprocess what this run will most likely store, still without a lock.
    // ^ Preprocessing is slow. Done under the write lock it makes every other
    //   writer of this collection (another update, `ir embed`, the daemon's
    //   rerank cache) fail with "database is locked" after the busy timeout.
    let mut fts_bodies = FtsBodies {
        commands: &pp_commands,
        chain: ChainState::NotStarted,
        by_hash: HashMap::new(),
        raw: HashSet::new(),
        first_error: None,
    };
    if has_preprocessor {
        let draft = plan(conn, opts, has_preprocessor, &hash_only)?.diff;
        pb.set_position(0);
        pb.set_length((draft.to_add.len() + draft.to_update.len()) as u64);
        pb.set_message("preprocessing");
        fts_bodies.prepare(&draft, &scanned, &pb)?;
    }

    // 3–6. Diff against the stored state and apply it under one write lock, so a
    // crash leaves the DB consistent.
    // ^ The stored state is read again inside the transaction, and that diff is
    //   the one applied. Applied from a read taken before it, two updates running
    //   at once both act on the same snapshot and the second fails on a duplicate
    //   path.
    let ((n_add, n_update, n_deactivate), n_raw, rebuilt) = db::write_txn(conn, || {
        let Plan {
            force,
            preprocessor_changed,
            diff: d,
        } = plan(conn, opts, has_preprocessor, &hash_only)?;
        // ^ Reported only when something existed to rebuild: a new collection
        //   with a preprocessor also takes this path on its first update.
        let rebuilt = !opts.force
            && preprocessor_changed
            && conn.query_row(
                "SELECT EXISTS (SELECT 1 FROM documents WHERE active = 1)",
                [],
                |row| row.get::<_, bool>(0),
            )?;
        // Only documents another update changed since step 2 are left to do.
        fts_bodies.prepare(&d, &scanned, &pb)?;

        let counts = (d.to_add.len(), d.to_update.len(), d.to_deactivate.len());
        let (n_add, n_update, n_deactivate) = counts;
        pb.set_position(0);
        pb.set_length((n_add + n_update + n_deactivate) as u64);
        pb.set_message("applying");

        if force {
            conn.execute_batch(
                "DELETE FROM documents_fts;
                 DELETE FROM documents;
                 DELETE FROM content;
                 DELETE FROM content_vectors;
                 DELETE FROM vectors_vec;
                 DELETE FROM llm_cache;
                 DELETE FROM fts_raw;",
            )?;
            db::ann::invalidate(conn)?;
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

        // 4. Delete removed files. Content and vectors remain hash-addressed caches.
        for rel_path in &d.to_deactivate {
            // ! Triggers disabled — must manually remove from FTS.
            delete_document(conn, rel_path, has_preprocessor)?;
            pb.inc(1);
            pb.set_message(format!("remove {rel_path}"));
        }

        // 5. Add new files
        for rel_path in &d.to_add {
            let (hash, content, file_mtime, file_birthtime) = scan_entry(&scanned, rel_path)?;
            store_document(
                conn,
                rel_path,
                hash,
                &decode(content),
                file_birthtime,
                file_mtime,
                has_preprocessor.then_some(&fts_bodies.by_hash),
            )?;
            pb.inc(1);
            pb.set_message(format!("add {rel_path}"));
        }

        // 6. Update changed files
        for rel_path in &d.to_update {
            let (hash, content, file_mtime, _file_birthtime) = scan_entry(&scanned, rel_path)?;
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
                hash,
                &decode(content),
                &created_at,
                file_mtime,
                has_preprocessor.then_some(&fts_bodies.by_hash),
            )?;
            pb.inc(1);
            pb.set_message(format!("update {rel_path}"));
        }

        // ^ Counted here, not from `raw`: step 2 may have prepared documents
        //   that another update stored first.
        let mut n_raw = 0;
        if has_preprocessor {
            for rel_path in d.to_add.iter().chain(&d.to_update) {
                let hash = &scan_entry(&scanned, rel_path)?.0;
                let sql = if fts_bodies.raw.contains(hash) {
                    n_raw += 1;
                    "INSERT OR IGNORE INTO fts_raw (hash) VALUES (?1)"
                } else {
                    "DELETE FROM fts_raw WHERE hash = ?1"
                };
                conn.execute(sql, [hash])?;
            }
        }

        // Record the preprocessing state only after the matching FTS content is ready.
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('has_preprocessor', ?1)",
            [if has_preprocessor { "1" } else { "0" }],
        )?;

        Ok((counts, n_raw, rebuilt))
    })?;

    pb.finish_with_message("done");
    let mut notes = Vec::new();
    if rebuilt {
        notes.push(format!(
            "the preprocessor setting changed ({}), so the collection was re-indexed \
             from scratch and its vectors were dropped",
            if has_preprocessor { "bound" } else { "unbound" }
        ));
    }
    if n_raw > 0 {
        notes.push(format!(
            "{n_raw} documents indexed without the preprocessor: {}; the next update retries them",
            fts_bodies.first_error.as_deref().unwrap_or("unknown error")
        ));
    }
    Ok(UpdateStats {
        scanned: scanned_files.len(),
        added: n_add,
        updated: n_update,
        deactivated: n_deactivate,
        pending_embeddings: embed::pending_count(conn, false)?,
        notes,
    })
}

/// {rel_path → hash} of the active documents.
/// ^ A document with no search-index row, or one indexed with raw text because
///   its preprocessor failed (`fts_raw`), gets an empty hash, so the diff sees
///   it as changed and stores it again. ir ≤ 0.18.1 wrote rows without search
///   text when a preprocessor failed to start; they otherwise report
///   "unchanged" forever.
fn stored_hashes(conn: &rusqlite::Connection) -> Result<HashMap<String, String>> {
    let mut stmt = conn.prepare(
        "SELECT d.path,
                CASE WHEN EXISTS (SELECT 1 FROM documents_fts f WHERE f.rowid = d.id)
                      AND NOT EXISTS (SELECT 1 FROM fts_raw r WHERE r.hash = d.hash)
                     THEN d.hash ELSE '' END
         FROM documents d
         WHERE d.active = 1",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    Ok(rows.collect::<std::result::Result<HashMap<_, _>, _>>()?)
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

/// `fts_bodies` is Some for a collection with a preprocessor: its FTS triggers
/// are dropped, so the preprocessed text is inserted here.
fn store_document(
    conn: &rusqlite::Connection,
    rel_path: &str,
    hash: &str,
    text: &str,
    created_at: &str,
    modified_at: &str,
    fts_bodies: Option<&HashMap<String, String>>,
) -> Result<()> {
    let title = chunker::extract_title(text, rel_path);

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

    if let Some(bodies) = fts_bodies {
        let body = bodies
            .get(hash)
            .ok_or_else(|| Error::Other(format!("no preprocessed text for {rel_path}")))?;
        conn.execute(
            "INSERT INTO documents_fts(rowid, path, title, body) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![doc_id, rel_path, title, body],
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
            .map(|s| (s.added, s.updated, s.deactivated))
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
}
