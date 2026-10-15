// End-to-end scenarios against the search daemon and the models it loads.
// They are `#[ignore]`d: all but one start a real daemon with the models of
// whoever runs them, and that one waits out the two-minute reply timeout.
//
//   cargo test --test e2e_daemon -- --ignored

mod support;

use std::fs;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};
use support::{Env, eventually, snapshot, wait_within, write};

const DAEMON_START: Duration = Duration::from_secs(180);

/// Pid of the running daemon, as `ir daemon status` reports it.
fn daemon_pid(env: &Env) -> Option<i32> {
    let status = env.ok(&["daemon", "status"]);
    let pid = status.split("pid=").nth(1)?.split_whitespace().next()?;
    pid.parse().ok()
}

fn wait_for_daemon(env: &Env, other_than: Option<i32>) -> Option<i32> {
    let mut found = None;
    eventually(DAEMON_START, || {
        found = daemon_pid(env).filter(|pid| Some(*pid) != other_than);
        found.is_some()
    });
    found
}

fn process_is_alive(pid: i32) -> bool {
    unsafe { libc::kill(pid, 0) == 0 }
}

/// Put a copy of the built binary at `dest` the way an installer does: write
/// the new file next to it, then rename it over the old one.
fn install(dest: &Path) {
    fs::create_dir_all(dest.parent().unwrap()).unwrap();
    let staged = dest.with_extension("new");
    fs::copy(env!("CARGO_BIN_EXE_ir"), &staged).unwrap();
    fs::rename(&staged, dest).unwrap();
}

/// After ir is upgraded, the daemon started by the old binary keeps running.
/// The next search must replace it without the user doing anything, and every
/// search on the way must still answer.
#[test]
#[ignore = "starts a real daemon: needs the embedding model"]
fn search_replaces_a_daemon_left_over_from_before_an_upgrade() {
    let env = Env::with_real_models("e2e-daemon-upgrade");
    let notes = env.add_collection("notes", &[]);
    write(&notes, "a.md", "# A\n\nupgradeneedle body\n");
    env.ok(&["update", "notes"]);
    let installed = env.path("bin/ir");
    install(&installed);
    let search = || {
        let output = env
            .ir_at(&installed)
            .args([
                "search",
                "--json",
                "--quiet",
                "-c",
                "notes",
                "upgradeneedle",
            ])
            .output()
            .unwrap();
        let (success, stdout, _) = snapshot(output);
        (success, stdout.contains("a.md"))
    };

    let cold = search();
    let first_pid = wait_for_daemon(&env, None).expect("first search starts the daemon");
    let warm = search();
    let kept_for_same_binary = daemon_pid(&env) == Some(first_pid);

    // File systems with one-second timestamps must still see a new file.
    std::thread::sleep(Duration::from_millis(1100));
    install(&installed);
    let status_after_upgrade = env.ok(&["daemon", "status"]);
    let upgraded = search();
    let second_pid = wait_for_daemon(&env, Some(first_pid));
    let old_daemon_gone = eventually(Duration::from_secs(10), || !process_is_alive(first_pid));
    let settled = search();
    let kept_after_replacement = daemon_pid(&env) == second_pid;

    assert_eq!(
        (
            cold,
            warm,
            kept_for_same_binary,
            status_after_upgrade.contains("started before ir was upgraded"),
            upgraded,
            second_pid.is_some(),
            old_daemon_gone,
            settled,
            kept_after_replacement,
        ),
        (
            (true, true),
            (true, true),
            true,
            true,
            (true, true),
            true,
            true,
            (true, true),
            true,
        ),
        "status after upgrade: {status_after_upgrade}"
    );
}

/// Replacement must follow upgrades without restarting the daemon in a loop.
/// Two install dirs hold the same ir, like two Homebrew kegs. The record a
/// daemon writes is edited to stand in for an older ir (its file left in place)
/// and for an executable the daemon could not read at startup.
#[test]
#[ignore = "starts a real daemon: needs the embedding model"]
fn daemon_is_replaced_only_by_a_newer_ir_and_never_in_a_loop() {
    let env = Env::with_real_models("e2e-daemon-versions");
    let notes = env.add_collection("notes", &[]);
    write(&notes, "a.md", "# A\n\nversionneedle body\n");
    env.ok(&["update", "notes"]);
    let keg_a = env.path("keg-a/ir");
    let keg_b = env.path("keg-b/ir");
    install(&keg_a);
    install(&keg_b);
    let search_from = |exe: &Path| {
        let output = env
            .ir_at(exe)
            .args(["search", "--json", "--quiet", "-c", "notes", "absentword"])
            .output()
            .unwrap();
        snapshot(output).0
    };
    let record = env.config_dir.join("daemon.build");
    let set_record_line = |index: usize, value: &str| {
        let text = fs::read_to_string(&record).unwrap();
        let mut lines: Vec<&str> = text.lines().collect();
        lines[index] = value;
        fs::write(&record, lines.join("\n")).unwrap();
    };

    search_from(&keg_a);
    let first = wait_for_daemon(&env, None).expect("search starts the daemon");
    let same_version_other_keg = (search_from(&keg_b), daemon_pid(&env) == Some(first));

    set_record_line(1, "unknown");
    let unknown_stamp = (
        search_from(&keg_a),
        search_from(&keg_a),
        daemon_pid(&env) == Some(first),
    );

    set_record_line(2, "0.0.1");
    let status_old = env.ok(&["daemon", "status"]);
    search_from(&keg_b);
    let second = wait_for_daemon(&env, Some(first));
    let first_gone = eventually(Duration::from_secs(10), || !process_is_alive(first));
    let after = (search_from(&keg_b), search_from(&keg_a));
    let kept = daemon_pid(&env) == second;

    assert_eq!(
        (
            same_version_other_keg,
            unknown_stamp,
            status_old.contains("started before ir was upgraded"),
            second.is_some(),
            first_gone,
            after,
            kept,
        ),
        (
            (true, true),
            (true, true, true),
            true,
            true,
            true,
            (true, true),
            true,
        ),
        "status with the older record: {status_old}"
    );
}

/// The daemon serves one connection at a time. A client that connects and then
/// sends nothing must not block the searches behind it.
#[test]
#[ignore = "starts a real daemon: needs the embedding model"]
fn stalled_client_does_not_block_other_searches() {
    let env = Env::with_real_models("e2e-daemon-stall");
    let notes = env.add_collection("notes", &[]);
    write(&notes, "a.md", "# A\n\nstallneedle body\n");
    env.ok(&["update", "notes"]);
    env.ok(&["search", "--quiet", "-c", "notes", "stallneedle"]);
    wait_for_daemon(&env, None).expect("first search starts the daemon");

    let stalled = UnixStream::connect(env.config_dir.join("daemon.sock")).unwrap();
    // BM25 finds nothing for this query, so the search has to ask the daemon.
    let search = env
        .ir()
        .args(["search", "--json", "-c", "notes", "absentword"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let finished = wait_within(search, Duration::from_secs(60));
    drop(stalled);

    let (success, _, stderr) = finished.expect("search must not wait for the stalled client");
    assert!(success, "search failed: {stderr}");
}

/// `ir update --force` drops every vector. When `ir embed` then runs with ANN
/// off, the vector count can match the old ANN index again. An ANN search must
/// not trust that index: its keys point at the dropped vectors.
#[test]
#[ignore = "embeds with the real model and starts a real daemon"]
fn ann_search_ignores_an_index_built_before_vectors_were_dropped() {
    let env = Env::with_real_models("e2e-daemon-ann");
    let notes = env.add_collection("notes", &[]);
    let rockets = "# Launch\n\nThe rocket engine fires and the launch vehicle climbs to orbit.\n";
    let cats = "# Cats\n\nThe cat purrs on the sofa and naps in the warm sun.\n";
    write(&notes, "a.md", cats);
    write(&notes, "b.md", rockets);
    env.ok(&["sync", "notes"]);
    let ann_file_built = fs::read_dir(env.config_dir.join("collections"))
        .unwrap()
        .any(|e| e.unwrap().file_name().to_string_lossy().contains("usearch"));

    // New text, so the old index keys name content that no longer exists.
    write(&notes, "a.md", &format!("{rockets}Stage two separates.\n"));
    write(&notes, "b.md", &format!("{cats}It chases a toy mouse.\n"));
    env.ok(&["update", "--force", "notes"]);
    let embed = env
        .ir()
        .env("IR_ANN", "off")
        .args(["embed", "notes"])
        .output()
        .unwrap();
    let (embedded, _, embed_err) = snapshot(embed);

    env.ok(&["search", "--quiet", "-c", "notes", "absentword"]);
    wait_for_daemon(&env, None).expect("search starts the daemon");
    let out = env.ok(&[
        "search",
        "--json",
        "--mode",
        "vector",
        "-c",
        "notes",
        "rocket launch into orbit",
    ]);
    let paths = support::result_paths(&serde_json::from_str(&out).unwrap());

    assert_eq!(
        (ann_file_built, embedded, paths.first().map(String::as_str)),
        (true, true, Some("a.md")),
        "embed stderr: {embed_err}\nresults: {paths:?}"
    );
}

/// The daemon caches rerank scores in the collection index. While another
/// process holds that index's write lock (an `ir update` in progress), a
/// reranked search must skip the cache write instead of waiting for the lock.
#[test]
#[ignore = "embeds and reranks with the real models"]
fn reranked_search_does_not_wait_for_an_index_update_in_progress() {
    let env = Env::with_real_models("e2e-daemon-rerank-lock");
    let notes = env.add_collection("notes", &[]);
    for (name, body) in [
        (
            "a.md",
            "The rocket engine fires and the vehicle climbs to orbit.",
        ),
        (
            "b.md",
            "The cat purrs on the sofa and naps in the warm sun.",
        ),
        (
            "c.md",
            "Sourdough bread needs a starter, flour, water and time.",
        ),
        (
            "d.md",
            "The river floods the valley every spring after the thaw.",
        ),
    ] {
        write(&notes, name, &format!("# {name}\n\n{body}\n"));
    }
    env.ok(&["sync", "notes"]);
    // ^ Research switch: no shortcut, so every search reaches the reranker.
    let search = |query: &str| {
        let started = Instant::now();
        let output = env
            .ir()
            .env("IR_DISABLE_SHORTCUTS", "1")
            .args(["search", "--json", "--quiet", "-c", "notes", query])
            .output()
            .unwrap();
        let (success, stdout, _) = snapshot(output);
        (success && stdout.contains(".md"), started.elapsed())
    };
    let db = env.config_dir.join("collections").join("notes.sqlite");
    let cached_scores = || {
        let out = std::process::Command::new("sqlite3")
            .arg(&db)
            .arg("SELECT COUNT(*) FROM llm_cache")
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u32>()
            .unwrap_or(0)
    };

    search("warming up");
    wait_for_daemon(&env, None).expect("search starts the daemon");
    let tier2_ready = eventually(DAEMON_START, || {
        env.ok(&["daemon", "status"]).contains("tier-2 ready")
    });
    let (unlocked_ok, unlocked_took) = search("space travel");
    let reranked = cached_scores() > 0;

    let mut writer = std::process::Command::new("sqlite3")
        .arg(&db)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    use std::io::Write;
    let mut stdin = writer.stdin.take().unwrap();
    stdin
        .write_all(b"BEGIN IMMEDIATE; INSERT INTO meta VALUES ('e2e-lock', '1');\n")
        .unwrap();
    stdin.flush().unwrap();
    std::thread::sleep(Duration::from_millis(500));
    let (locked_ok, locked_took) = search("baking at home");
    drop(stdin);
    let _ = writer.wait();

    assert_eq!(
        (
            tier2_ready,
            unlocked_ok,
            reranked,
            locked_ok,
            locked_took < unlocked_took + Duration::from_secs(3),
        ),
        (true, true, true, true, true),
        "unlocked {unlocked_took:?}, locked {locked_took:?}"
    );
}

/// A daemon that accepts a query and never answers must not hang the search:
/// after the reply timeout the search returns what it has. The hung daemon is
/// a listener on the daemon socket that never reads.
#[test]
#[ignore = "waits out the 120 s daemon reply timeout"]
fn search_gives_up_on_a_daemon_that_never_answers() {
    let env = Env::new("e2e-daemon-hung");
    let notes = env.add_collection("notes", &[]);
    write(&notes, "a.md", "# A\n\nhungneedle body\n");
    env.ok(&["update", "notes"]);
    let _hung = UnixListener::bind(env.config_dir.join("daemon.sock")).unwrap();

    let started = Instant::now();
    let search = env
        .ir()
        .args(["search", "--json", "-c", "notes", "absentword"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let finished = wait_within(search, Duration::from_secs(180));
    let waited = started.elapsed();

    let (success, stdout, stderr) = finished.expect("search must give up on a silent daemon");
    assert_eq!(
        (
            success,
            stdout.trim(),
            stderr.contains("daemon did not answer within 120s"),
            waited >= Duration::from_secs(115),
        ),
        (true, "[]", true, true),
        "waited {waited:?}; stderr: {stderr}"
    );
}
