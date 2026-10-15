// End-to-end output contract of `ir update`, `ir sync` and `ir embed`.

mod support;

use std::path::Path;
use support::{Env, Snapshot, write};

fn update_header(dir: &Path) -> String {
    format!("updating 'empty' ({})…\n", dir.display())
}

fn assert_model_failure_after_update(
    dir: &Path,
    output: Snapshot,
    expected_scanned: usize,
    expected_added: usize,
) {
    let (success, stdout, stderr) = output;
    let expected_stdout = format!(
        "{}  {expected_scanned} scanned: {expected_added} added, 0 updated, 0 deactivated\n",
        update_header(dir)
    );
    assert_eq!(
        (
            success,
            stdout,
            stderr.contains("IR_EMBEDDING_MODEL="),
            stderr.contains("is not a file, directory, or known HuggingFace repo ID"),
        ),
        (false, expected_stdout, true, true)
    );
}

#[test]
fn empty_collection_sync_and_embed_update_before_skipping_model_loading() {
    let env = Env::new("sync-cli");
    let dir = env.add_collection("empty", &[]);
    let expected_phases = format!(
        "{}{}",
        update_header(&dir),
        concat!(
            "  0 scanned: 0 added, 0 updated, 0 deactivated\n",
            "embedding 'empty'…\n",
            "  0 documents, 0 chunks embedded\n",
        )
    );

    assert_eq!(
        (env.run(&["sync", "empty"]), env.run(&["embed", "empty"])),
        (
            (true, expected_phases.clone(), String::new()),
            (true, expected_phases, String::new()),
        )
    );
}

#[test]
fn nonempty_sync_and_embed_commit_updates_before_model_loading_failure() {
    let env = Env::new("sync-cli");
    let dir = env.add_collection("empty", &[]);

    write(&dir, "first.md", "# First\n");
    assert_model_failure_after_update(&dir, env.run(&["sync", "empty"]), 1, 1);
    assert_model_failure_after_update(&dir, env.run(&["sync", "empty"]), 1, 0);

    write(&dir, "second.md", "# Second\n");
    assert_model_failure_after_update(&dir, env.run(&["embed", "empty"]), 2, 1);
    assert_model_failure_after_update(&dir, env.run(&["embed", "empty"]), 2, 0);
}

#[test]
fn update_reports_collection_path_scan_count_and_pending_embeddings() {
    let env = Env::new("sync-cli");
    let dir = env.add_collection("empty", &[]);
    let expected = |counts: &str, pending: &str| {
        (
            true,
            format!("{}  {counts}\n  {pending}\n", update_header(&dir)),
            String::new(),
        )
    };
    let hint = "pending embedding (run `ir embed` or `ir sync`)";

    let empty = env.run(&["update", "empty"]);
    write(&dir, "first.md", "# First\n");
    let added = env.run(&["update", "empty"]);
    let unchanged = env.run(&["update", "empty"]);
    let forced = env.run(&["update", "empty", "--force"]);

    assert_eq!(
        (empty, added, unchanged, forced),
        (
            expected(
                "0 scanned: 0 added, 0 updated, 0 deactivated",
                "0 pending embedding"
            ),
            expected(
                "1 scanned: 1 added, 0 updated, 0 deactivated",
                &format!("1 {hint}")
            ),
            expected(
                "1 scanned: 0 added, 0 updated, 0 deactivated",
                &format!("1 {hint}")
            ),
            expected(
                "1 scanned: 1 added, 0 updated, 0 deactivated",
                &format!("1 {hint}")
            ),
        )
    );
}

/// `--force` re-adds an unchanged document; without it nothing is added.
#[test]
fn sync_accepts_force_in_every_position_and_at_most_one_collection() {
    let env = Env::new("sync-cli");
    let dir = env.add_collection("empty", &[]);
    write(&dir, "first.md", "# First\n");
    env.ok(&["update", "empty"]);

    assert_model_failure_after_update(&dir, env.run(&["sync"]), 1, 0);
    assert_model_failure_after_update(&dir, env.run(&["sync", "-f"]), 1, 1);
    assert_model_failure_after_update(&dir, env.run(&["sync", "empty", "--force"]), 1, 1);
    assert_model_failure_after_update(&dir, env.run(&["sync", "--force", "empty"]), 1, 1);
    let (success, _, stderr) = env.run(&["sync", "empty", "second"]);
    assert_eq!(
        (success, stderr.contains("unexpected argument 'second'")),
        (false, true),
        "stderr: {stderr}"
    );
}
