// End-to-end index scenarios through the compiled `ir` binary.
// Oracle for "indexed" is always a real BM25 search, never a row count.

mod support;

use std::fs;
use std::process::{Child, Stdio};
use std::time::Duration;
use support::{Env, add_preprocessor, line_filter, update_counts, wait_within, write};

fn spawn_update(env: &Env, collection: &str) -> Child {
    env.ir()
        .args(["update", collection])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ir update")
}

/// Run `writers` updates of one collection at the same moment and return the
/// summed (added, updated, deactivated). Every process must succeed.
fn concurrent_updates(env: &Env, collection: &str, writers: usize) -> (usize, usize, usize) {
    let children: Vec<Child> = (0..writers)
        .map(|_| spawn_update(env, collection))
        .collect();
    let mut total = (0, 0, 0);
    for child in children {
        let (success, stdout, stderr) =
            wait_within(child, Duration::from_secs(120)).expect("concurrent update finished");
        assert!(success, "concurrent update failed:\n{stdout}\n{stderr}");
        let (_, added, updated, deactivated) = update_counts(&stdout);
        total = (total.0 + added, total.1 + updated, total.2 + deactivated);
    }
    total
}

#[test]
fn concurrent_updates_apply_every_change_exactly_once() {
    let env = Env::new("e2e-concurrent-update");
    let notes = env.add_collection("notes", &[]);
    for i in 0..400 {
        write(
            &notes,
            &format!("n{i:03}.md"),
            &format!("# Note {i}\n\nfirstpass{i:03} body\n"),
        );
    }

    let first = concurrent_updates(&env, "notes", 6);

    for i in 0..100 {
        write(
            &notes,
            &format!("n{i:03}.md"),
            &format!("# Note {i}\n\nsecondpass{i:03} body\n"),
        );
    }
    for i in 100..150 {
        fs::remove_file(notes.join(format!("n{i:03}.md"))).unwrap();
    }
    let second = concurrent_updates(&env, "notes", 6);
    let settled = update_counts(&env.ok(&["update", "notes"]));

    assert_eq!(
        (
            first,
            second,
            settled,
            env.bm25_paths(&["notes"], "firstpass399"),
            env.bm25_paths(&["notes"], "secondpass042"),
            env.bm25_paths(&["notes"], "firstpass042"),
            env.bm25_paths(&["notes"], "firstpass120"),
        ),
        (
            (400, 0, 0),
            (0, 100, 50),
            (350, 0, 0, 0),
            vec!["n399.md".to_string()],
            vec!["n042.md".to_string()],
            vec![],
            vec![],
        )
    );
}

/// A preprocessor that cannot run must not stop indexing. The update succeeds
/// with a warning and indexes raw text, so every file stays searchable. Once
/// the preprocessor works again, the next update stores those files again
/// with it, and the update after that has nothing left to do.
fn failing_preprocessor_indexes_raw_text_and_retries(tag: &str, broken_command: &str) {
    let env = Env::new(tag);
    let rewrite = rewriting_preprocessor(&env);
    add_preprocessor(&env, "pp", &rewrite);
    let notes = env.add_collection("notes", &["--preprocessor", "pp"]);
    write(&notes, "a.md", "# A\n\nalphaneedle rawtoken\n");
    env.ok(&["update", "notes"]);

    env.ok(&["preprocessor", "add", "pp", broken_command]);
    write(&notes, "a.md", "# A\n\nbetaneedle rawtoken\n");
    write(&notes, "b.md", "# B\n\ngammaneedle rawtoken\n");
    let (broken_success, broken_out, broken_stderr) = env.run(&["update", "notes"]);
    let while_broken = (
        update_counts(&broken_out),
        env.bm25_paths(&["notes"], "betaneedle"),
        env.bm25_paths(&["notes"], "gammaneedle"),
        env.bm25_paths(&["notes"], "alphaneedle"),
    );

    add_preprocessor(&env, "pp", &rewrite);
    let repaired = update_counts(&env.ok(&["update", "notes"]));
    let settled = update_counts(&env.ok(&["update", "notes"]));
    let mut cooked = env.bm25_paths(&["notes"], "cookedtoken");
    cooked.sort();

    assert_eq!(
        (
            broken_success,
            broken_stderr.contains("2 documents indexed without the preprocessor"),
            while_broken,
            repaired,
            settled,
            cooked,
        ),
        (
            true,
            true,
            (
                (2, 1, 1, 0),
                vec!["a.md".to_string()],
                vec!["b.md".to_string()],
                vec![],
            ),
            (2, 0, 2, 0),
            (2, 0, 0, 0),
            vec!["a.md".to_string(), "b.md".to_string()],
        ),
        "stderr of the update with the broken preprocessor: {broken_stderr}"
    );
}

#[test]
fn preprocessor_that_cannot_start_does_not_stop_the_update() {
    failing_preprocessor_indexes_raw_text_and_retries(
        "e2e-pp-missing",
        "/nonexistent/ir-e2e-preprocessor",
    );
}

#[test]
fn preprocessor_that_exits_does_not_stop_the_update() {
    failing_preprocessor_indexes_raw_text_and_retries("e2e-pp-exits", "false");
}

/// An index written by ir <= 0.18.1 while its preprocessor was failing has
/// document rows with no search text. Such rows report "unchanged" forever.
/// The damage is reproduced by removing the search rows from a real index.
#[test]
fn update_repairs_documents_recorded_without_search_text() {
    let env = Env::new("e2e-repair");
    env.ok(&["preprocessor", "add", "pp", "cat"]);
    let notes = env.add_collection("notes", &["--preprocessor", "pp"]);
    write(&notes, "kept.md", "# Kept\n\nkeptneedle body\n");
    write(&notes, "lost.md", "# Lost\n\nlostneedle body\n");
    env.ok(&["update", "notes"]);
    {
        let conn = rusqlite::Connection::open(env.db_path("notes")).unwrap();
        let removed = conn
            .execute(
                "DELETE FROM documents_fts
                 WHERE rowid = (SELECT id FROM documents WHERE path = 'lost.md')",
                [],
            )
            .unwrap();
        assert_eq!(removed, 1, "fixture must remove exactly one search row");
    }
    let damaged = env.bm25_paths(&["notes"], "lostneedle");

    let repair = update_counts(&env.ok(&["update", "notes"]));
    let settled = update_counts(&env.ok(&["update", "notes"]));

    assert_eq!(
        (
            damaged,
            repair,
            settled,
            env.bm25_paths(&["notes"], "lostneedle"),
            env.bm25_paths(&["notes"], "keptneedle"),
        ),
        (
            vec![],
            (2, 0, 1, 0),
            (2, 0, 0, 0),
            vec!["lost.md".to_string()],
            vec!["kept.md".to_string()],
        )
    );
}

/// A streaming filter echoes while ir is still writing. A line larger than the
/// OS pipe buffers must not deadlock the update.
#[test]
fn very_long_line_does_not_deadlock_a_streaming_preprocessor() {
    let env = Env::new("e2e-long-line");
    env.ok(&["preprocessor", "add", "pp", "cat"]);
    let notes = env.add_collection("notes", &["--preprocessor", "pp"]);
    let long_line = format!("{} longlineneedle", "filler ".repeat(100_000));
    write(
        &notes,
        "long.md",
        &format!("# Long\n\n{long_line}\n\nafterneedle\n"),
    );

    let finished = wait_within(spawn_update(&env, "notes"), Duration::from_secs(60));

    let (success, stdout, stderr) = finished.expect("update must not hang on a 700 KB line");
    assert!(success, "update failed:\n{stdout}\n{stderr}");
    assert_eq!(
        (
            update_counts(&stdout),
            env.bm25_paths(&["notes"], "longlineneedle"),
            env.bm25_paths(&["notes"], "afterneedle"),
        ),
        (
            (1, 1, 0, 0),
            vec!["long.md".to_string()],
            vec!["long.md".to_string()],
        )
    );
}

/// `rawtoken` -> `cookedtoken` line filter obeying the preprocessor protocol.
fn rewriting_preprocessor(env: &Env) -> String {
    line_filter(
        env,
        "rewrite.py",
        "    print(s.replace('rawtoken', 'cookedtoken'), flush=True)",
    )
}

#[test]
fn binding_and_unbinding_a_preprocessor_reindexes_the_collection() {
    let env = Env::new("e2e-bind");
    let notes = env.add_collection("notes", &[]);
    write(&notes, "a.md", "# A\n\nrawtoken body\n");
    env.ok(&["update", "notes"]);
    add_preprocessor(&env, "rw", &rewriting_preprocessor(&env));
    let before = env.bm25_paths(&["notes"], "cookedtoken");

    env.ok(&["preprocessor", "bind", "rw", "notes"]);
    let bound = (
        env.bm25_paths(&["notes"], "cookedtoken"),
        update_counts(&env.ok(&["update", "notes"])),
    );
    env.ok(&["preprocessor", "unbind", "rw", "notes"]);
    let unbound = (
        env.bm25_paths(&["notes"], "cookedtoken"),
        env.bm25_paths(&["notes"], "rawtoken"),
    );

    assert_eq!(
        (before, bound, unbound),
        (
            vec![],
            (vec!["a.md".to_string()], (1, 0, 0, 0)),
            (vec![], vec!["a.md".to_string()]),
        )
    );
}

#[test]
fn search_across_all_collections_skips_one_that_was_never_indexed() {
    let env = Env::new("e2e-unindexed");
    let notes = env.add_collection("notes", &[]);
    env.add_collection("fresh", &[]);
    write(&notes, "a.md", "# A\n\nonlyneedle body\n");
    env.ok(&["update", "notes"]);

    assert_eq!(env.bm25_paths(&[], "onlyneedle"), vec!["a.md".to_string()]);
}

/// `retrieval:` in config.yml is the authoritative per-collection switch for
/// what `ir sync` builds. Both collections are marked as already embedded (the
/// rows the embedder writes), so the run needs no model.
#[test]
fn sync_builds_the_doc_graph_only_where_the_collection_config_allows_it() {
    let env = Env::new("e2e-retrieval-config");
    let plain = env.path("plain");
    let tuned = env.path("tuned");
    write(&plain, "a.md", "# A\n\nplain body\n");
    write(&tuned, "a.md", "# A\n\ntuned body\n");
    fs::write(
        env.config_dir.join("config.yml"),
        format!(
            "collections:\n- name: plain\n  path: {}\n- name: tuned\n  path: {}\n  retrieval:\n    t0_graph_expand: false\n",
            plain.display(),
            tuned.display()
        ),
    )
    .unwrap();
    env.ok(&["update"]);
    for name in ["plain", "tuned"] {
        let conn = rusqlite::Connection::open(env.db_path(name)).unwrap();
        conn.execute(
            "INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
             SELECT hash, 0, 0, 'e2e', '2024-01-01' FROM documents",
            [],
        )
        .unwrap();
    }

    let stdout = env.ok(&["sync"]);

    assert_eq!(
        (
            stdout.contains("building doc graph for 'plain'"),
            stdout.contains("building doc graph for 'tuned'"),
        ),
        (true, false),
        "sync output: {stdout}"
    );
}

/// Preprocessing is the slow part of an update. While one update preprocesses,
/// a second writer of the same collection must not run into the 5 s lock wait.
#[test]
fn slow_preprocessing_does_not_lock_out_a_concurrent_update() {
    let env = Env::new("e2e-slow-pp");
    let slow = line_filter(
        &env,
        "slow.py",
        "    if s != 'IRSENTINEL':\n        time.sleep(0.1)\n    print(s, flush=True)",
    );
    add_preprocessor(&env, "slow", &slow);
    let notes = env.add_collection("notes", &["--preprocessor", "slow"]);
    // 35 documents x 2 text lines x 0.1 s = 7 s of preprocessing per update.
    for i in 0..35 {
        write(
            &notes,
            &format!("n{i:02}.md"),
            &format!("# Note {i}\n\nslowneedle{i:02} body\n"),
        );
    }

    let added = concurrent_updates(&env, "notes", 2);

    assert_eq!(
        (added, env.bm25_paths(&["notes"], "slowneedle34")),
        ((35, 0, 0), vec!["n34.md".to_string()])
    );
}

/// The stored document, as `ir get` prints it; None when ir has no such path.
fn stored_body(env: &Env, path: &str) -> Option<String> {
    let (success, stdout, _) = env.run(&["get", path, "-c", "notes"]);
    success.then_some(stdout)
}

fn removed_file_is_gone_and_returns_once_when_recreated(tag: &str, collection_args: &[&str]) {
    let env = Env::new(tag);
    env.ok(&["preprocessor", "add", "pp", "cat"]);
    let notes = env.add_collection("notes", collection_args);
    let content = "# Return\n\nreturnneedle body\n";
    write(&notes, "return.md", content);
    env.ok(&["update", "notes"]);

    fs::remove_file(notes.join("return.md")).unwrap();
    let removed = (
        update_counts(&env.ok(&["update", "notes"])),
        env.bm25_paths(&["notes"], "returnneedle"),
        stored_body(&env, "return.md"),
    );
    write(&notes, "return.md", content);
    let returned = (
        update_counts(&env.ok(&["update", "notes"])),
        env.bm25_paths(&["notes"], "returnneedle"),
        stored_body(&env, "return.md").map(|body| body.contains("returnneedle body")),
    );

    assert_eq!(
        (removed, returned),
        (
            ((0, 0, 0, 1), vec![], None),
            ((1, 1, 0, 0), vec!["return.md".to_string()], Some(true)),
        )
    );
}

#[test]
fn removed_file_is_gone_and_returns_once_when_recreated_plain() {
    removed_file_is_gone_and_returns_once_when_recreated("e2e-return", &[]);
}

#[test]
fn removed_file_is_gone_and_returns_once_when_recreated_with_preprocessor() {
    removed_file_is_gone_and_returns_once_when_recreated(
        "e2e-return-pp",
        &["--preprocessor", "pp"],
    );
}

#[test]
fn moved_file_is_found_only_at_its_new_path() {
    let env = Env::new("e2e-move");
    let notes = env.add_collection("notes", &[]);
    write(&notes, "old.md", "# Moved\n\nmovedneedle body\n");
    env.ok(&["update", "notes"]);

    fs::rename(notes.join("old.md"), notes.join("new.md")).unwrap();
    let counts = update_counts(&env.ok(&["update", "notes"]));

    assert_eq!(
        (
            counts,
            env.bm25_paths(&["notes"], "movedneedle"),
            stored_body(&env, "old.md"),
            stored_body(&env, "new.md").map(|body| body.contains("movedneedle body")),
        ),
        ((1, 1, 0, 1), vec!["new.md".to_string()], None, Some(true))
    );
}

/// A collection directory that is gone (unmounted volume, renamed folder) must
/// fail the update, with or without `--force`, and leave the index as it was.
#[test]
fn missing_collection_directory_fails_the_update_and_keeps_the_index() {
    let env = Env::new("e2e-missing-root");
    let notes = env.add_collection("notes", &[]);
    write(&notes, "stable.md", "# Stable\n\nstableneedle body\n");
    env.ok(&["update", "notes"]);
    fs::remove_dir_all(&notes).unwrap();

    let (update_ok, _, _) = env.run(&["update", "notes"]);
    let after_update = env.bm25_paths(&["notes"], "stableneedle");
    let (force_ok, _, _) = env.run(&["update", "notes", "--force"]);
    let after_force = env.bm25_paths(&["notes"], "stableneedle");

    assert_eq!(
        (update_ok, after_update, force_ok, after_force),
        (
            false,
            vec!["stable.md".to_string()],
            false,
            vec!["stable.md".to_string()],
        )
    );
}

/// A preprocessor may print nothing for a line (a tokenizer that filters a
/// punctuation-only line). The update must not wait for output that never
/// comes, and the lines around the dropped one stay searchable.
#[test]
fn preprocessor_that_drops_a_line_does_not_stall_the_update() {
    let env = Env::new("e2e-pp-drops");
    let dropper = line_filter(
        &env,
        "drop.py",
        "    if s != '.':\n        print(s, flush=True)",
    );
    add_preprocessor(&env, "dropper", &dropper);
    let notes = env.add_collection("notes", &["--preprocessor", "dropper"]);
    write(
        &notes,
        "mixed.md",
        "# Mixed\n\nbeforeneedle\n.\nmiddleneedle\n.\nafterneedle\n",
    );

    let finished = wait_within(spawn_update(&env, "notes"), Duration::from_secs(60));

    let (success, stdout, stderr) = finished.expect("update must not hang on a dropped line");
    assert!(success, "update failed:\n{stdout}\n{stderr}");
    let found = vec!["mixed.md".to_string()];
    assert_eq!(
        (
            env.bm25_paths(&["notes"], "beforeneedle"),
            env.bm25_paths(&["notes"], "middleneedle"),
            env.bm25_paths(&["notes"], "afterneedle"),
        ),
        (found.clone(), found.clone(), found)
    );
}

/// Preprocessors bound to one collection run in order, each on the output of
/// the one before, for documents and for queries alike.
#[test]
fn chained_preprocessors_apply_in_order_to_documents_and_queries() {
    let env = Env::new("e2e-pp-chain");
    let first = line_filter(
        &env,
        "first.py",
        "    print(s.replace('rawtoken', 'midtoken'), flush=True)",
    );
    let second = line_filter(
        &env,
        "second.py",
        "    print(s.replace('midtoken', 'cookedtoken'), flush=True)",
    );
    add_preprocessor(&env, "first", &first);
    add_preprocessor(&env, "second", &second);
    let notes = env.add_collection(
        "notes",
        &["--preprocessor", "first", "--preprocessor", "second"],
    );
    write(&notes, "a.md", "# A\n\nrawtoken body\n");
    env.ok(&["update", "notes"]);

    let a = vec!["a.md".to_string()];
    assert_eq!(
        (
            env.bm25_paths(&["notes"], "cookedtoken"),
            env.bm25_paths(&["notes"], "rawtoken"),
            env.bm25_paths(&["notes"], "midtoken"),
        ),
        (a.clone(), a.clone(), a)
    );
}
