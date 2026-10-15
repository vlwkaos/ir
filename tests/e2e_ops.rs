// End-to-end scenarios for cross-process state: config.yml, the daemon's
// files, and the per-collection embed lock.

mod support;

use std::fs;
use std::os::unix::io::AsRawFd;
use std::process::{Command, Stdio};
use std::time::Duration;
use support::{Env, add_preprocessor, eventually, wait_within, write};

#[test]
fn concurrent_collection_adds_all_survive() {
    let env = Env::new("e2e-config-race");
    let names: Vec<String> = (0..12).map(|i| format!("col{i:02}")).collect();
    let children: Vec<_> = names
        .iter()
        .map(|name| {
            let dir = env.path(name);
            fs::create_dir_all(&dir).unwrap();
            env.ir()
                .args(["collection", "add", name])
                .arg(&dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("spawn ir collection add")
        })
        .collect();
    for child in children {
        let (success, stdout, stderr) =
            wait_within(child, Duration::from_secs(60)).expect("collection add finished");
        assert!(success, "collection add failed:\n{stdout}\n{stderr}");
    }

    let listing = env.ok(&["collection", "ls"]);
    let mut listed: Vec<String> = listing
        .lines()
        .filter_map(|line| line.split_whitespace().next().map(str::to_string))
        .collect();
    listed.sort();

    assert_eq!(listed, names);
}

/// After a crash or reboot the pid file can name a pid that now belongs to an
/// unrelated process. `ir daemon stop` must not signal it.
#[test]
fn daemon_stop_never_signals_a_process_named_by_a_stale_pid_file() {
    let env = Env::new("e2e-stale-pid");
    let mut bystander = Command::new("sleep")
        .arg("60")
        .spawn()
        .expect("spawn bystander");
    let pid_file = env.config_dir.join("daemon.pid");
    fs::write(&pid_file, bystander.id().to_string()).unwrap();

    let (success, _, stderr) = env.run(&["daemon", "stop"]);
    std::thread::sleep(Duration::from_millis(300));
    let bystander_alive = bystander.try_wait().unwrap().is_none();
    let _ = bystander.kill();
    let _ = bystander.wait();

    assert_eq!(
        (success, bystander_alive, pid_file.exists()),
        (true, true, false),
        "stderr: {stderr}"
    );
}

/// Two `ir sync`/`ir embed` runs on one collection would load the model twice
/// and embed the same documents twice. The second run must wait for the first.
/// The first run is represented by holding the collection's embed lock the way
/// another ir process holds it: an exclusive flock.
#[test]
fn sync_waits_while_another_process_embeds_the_same_collection() {
    let env = Env::new("e2e-embed-lock");
    env.add_collection("notes", &[]);
    env.ok(&["update", "notes"]);
    let lock_path = env.config_dir.join("collections").join("notes.embed.lock");
    let lock = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .unwrap();
    assert_eq!(unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) }, 0);

    let mut sync = env
        .ir()
        .args(["sync", "notes"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ir sync");
    std::thread::sleep(Duration::from_millis(1500));
    let blocked = sync.try_wait().unwrap().is_none();
    drop(lock);
    let (success, stdout, stderr) =
        wait_within(sync, Duration::from_secs(60)).expect("sync finishes once the lock is free");

    assert_eq!(
        (
            blocked,
            success,
            stdout.ends_with("embedding 'notes'…\n  0 documents, 0 chunks embedded\n"),
            stderr.contains("waiting"),
        ),
        (true, true, true, true),
        "stdout: {stdout}\nstderr: {stderr}"
    );
}

#[test]
fn renaming_a_collection_keeps_its_index() {
    let env = Env::new("e2e-rename");
    let notes = env.add_collection("notes", &[]);
    write(&notes, "a.md", "# A\n\nrenameneedle body\n");
    env.ok(&["update", "notes"]);

    env.ok(&["collection", "rename", "notes", "journal"]);

    assert_eq!(
        (
            env.bm25_paths(&["journal"], "renameneedle"),
            support::update_counts(&env.ok(&["update", "journal"])),
        ),
        (vec!["a.md".to_string()], (1, 0, 0, 0))
    );
}

#[test]
fn purging_a_collection_removes_every_index_file() {
    let env = Env::new("e2e-purge");
    let notes = env.add_collection("notes", &[]);
    env.ok(&["sync", "notes"]);
    write(&notes, "a.md", "# A\n\npurgeneedle body\n");
    env.ok(&["update", "notes"]);

    env.ok(&["collection", "rm", "notes", "--purge"]);
    let leftovers: Vec<String> = fs::read_dir(env.config_dir.join("collections"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("notes."))
        .collect();
    env.ok(&["collection", "add", "notes", notes.to_str().unwrap()]);

    assert_eq!(
        (
            leftovers,
            support::update_counts(&env.ok(&["update", "notes"])),
        ),
        (Vec::<String>::new(), (1, 1, 0, 0))
    );
}

/// BM25 needs no model, so a BM25 search must leave the daemon alone. A hybrid
/// search starts it in the background and answers this query from BM25.
#[test]
fn only_a_search_that_needs_a_model_starts_the_daemon() {
    let mut env = Env::new("e2e-daemon-start");
    let notes = env.add_collection("notes", &[]);
    write(&notes, "a.md", "# A\n\nstartneedle one\n");
    write(&notes, "b.md", "# B\n\nstartneedle two and more words\n");
    env.ok(&["update", "notes"]);
    env.allow_daemon_start();
    let daemon_log = env.config_dir.join("daemon.log");

    let mut bm25 = env.bm25_paths(&["notes"], "startneedle");
    let started_by_bm25 = daemon_log.exists();
    let hybrid_out = env.ok(&["search", "--json", "--quiet", "-c", "notes", "startneedle"]);
    let mut hybrid = support::result_paths(&serde_json::from_str(&hybrid_out).unwrap());
    // The started daemon cannot load a model in this fixture; it says so and exits.
    let daemon_ran = eventually(Duration::from_secs(30), || {
        fs::read_to_string(&daemon_log).is_ok_and(|log| log.contains("IR_EMBEDDING_MODEL"))
    });
    bm25.sort();
    hybrid.sort();

    assert_eq!(
        (started_by_bm25, daemon_ran, hybrid),
        (false, true, bm25),
        "daemon log: {:?}",
        fs::read_to_string(&daemon_log)
    );
}

/// A long-running `ir mcp` starts daemons; each one exits sooner or later (idle
/// timeout, replacement after an upgrade, or here: no model to load). The MCP
/// server must reap it, not keep it as a zombie for the rest of its life.
#[test]
fn mcp_server_reaps_the_daemons_it_started() {
    let mut env = Env::new("e2e-mcp-reap");
    let notes = env.add_collection("notes", &[]);
    write(&notes, "a.md", "# A\n\nreapneedle one\n");
    env.ok(&["update", "notes"]);
    env.allow_daemon_start();
    let daemon_log = env.config_dir.join("daemon.log");
    let mut mcp = support::Mcp::start(&env);

    mcp.tool(
        "search",
        serde_json::json!({ "query": "absentword", "mode": "hybrid", "collections": ["notes"] }),
    )
    .expect("search tool");
    let daemon_exited = eventually(Duration::from_secs(30), || {
        fs::read_to_string(&daemon_log).is_ok_and(|log| log.contains("IR_EMBEDDING_MODEL"))
    });
    // ^ States of the MCP server's child processes; "Z" is a zombie.
    let children = || {
        let out = Command::new("ps")
            .args(["-A", "-o", "ppid=,stat="])
            .output()
            .expect("ps");
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| {
                let mut f = l.split_whitespace();
                (f.next()? == mcp.pid().to_string()).then(|| f.next().unwrap_or("").to_string())
            })
            .collect::<Vec<_>>()
    };
    let reaped = eventually(Duration::from_secs(5), || {
        !children().iter().any(|s| s.starts_with('Z'))
    });

    assert_eq!(
        (daemon_exited, reaped),
        (true, true),
        "children of ir mcp: {:?}",
        children()
    );
}

/// A collection name becomes a file name under the data dir.
#[test]
fn collection_names_that_are_not_plain_file_names_are_rejected() {
    let env = Env::new("e2e-names");
    let dir = env.path("docs");
    fs::create_dir_all(&dir).unwrap();
    let add = |name: &str| {
        env.run(&["collection", "add", name, dir.to_str().unwrap()])
            .0
    };

    let rejected: Vec<bool> = ["", "a/b", "..", "a..b"].iter().map(|n| add(n)).collect();
    let accepted = (add("knowledge"), add("my-notes_2024"));
    let (bad_rename, _, _) = env.run(&["collection", "rename", "knowledge", "a/b"]);
    let (good_rename, _, _) = env.run(&["collection", "rename", "knowledge", "ok-name"]);
    let mut listed: Vec<String> = env
        .ok(&["collection", "ls"])
        .lines()
        .filter_map(|line| line.split_whitespace().next().map(str::to_string))
        .collect();
    listed.sort();

    assert_eq!(
        (rejected, accepted, bad_rename, good_rename, listed),
        (
            vec![false; 4],
            (true, true),
            false,
            true,
            vec!["my-notes_2024".to_string(), "ok-name".to_string()],
        )
    );
}

/// Binding the Korean preprocessor also applies its routing default to the
/// collection; unbinding removes that default again. Other aliases set none.
#[test]
fn binding_ko_sets_the_korean_routing_default_and_unbinding_clears_it() {
    let env = Env::new("e2e-ko-routing");
    add_preprocessor(&env, "ko", "cat");
    add_preprocessor(&env, "ja", "cat");
    env.add_collection("korean", &[]);
    env.add_collection("japanese", &[]);
    let config = || fs::read_to_string(env.config_dir.join("config.yml")).unwrap();
    let default_routing = "fused_strong_product: 0.05";

    env.ok(&["preprocessor", "bind", "ja", "japanese"]);
    let after_ja = config().contains(default_routing);
    env.ok(&["preprocessor", "bind", "ko", "korean"]);
    let after_ko = config().contains(default_routing);
    env.ok(&["preprocessor", "unbind", "ko", "korean"]);
    let after_unbind = config().contains(default_routing);

    assert_eq!((after_ja, after_ko, after_unbind), (false, true, false));
}

/// A script installs the Korean preprocessor while a collection exists. With no
/// terminal for the bind prompt the install still succeeds, and binding it
/// lets BM25 match a verb by its stem (끓이는 and 끓이다 share 끓이).
#[test]
#[ignore = "downloads lindera and its dictionary from GitHub"]
fn preprocessor_install_without_a_terminal_succeeds_and_tokenizes_korean() {
    let env = Env::new("e2e-pp-install");
    let docs = env.add_collection("kor", &[]);
    write(&docs, "a.md", "# 날씨\n\n오늘은 서울에 비가 내렸습니다.\n");
    write(
        &docs,
        "b.md",
        "# 요리\n\n김치찌개를 끓이는 방법을 배웠다.\n",
    );
    env.ok(&["update", "kor"]);
    let before = env.bm25_paths(&["kor"], "끓이다");

    let (installed, stdout, stderr) = env.run(&["preprocessor", "install", "ko"]);
    env.ok(&["preprocessor", "bind", "ko", "kor"]);

    assert_eq!(
        (
            before,
            installed,
            stdout.contains("ir preprocessor bind ko <collection>"),
            env.bm25_paths(&["kor"], "끓이다"),
        ),
        (Vec::<String>::new(), true, true, vec!["b.md".to_string()]),
        "stdout: {stdout}\nstderr: {stderr}"
    );
}

/// `XDG_CONFIG_HOME` still locates the data dir when `IR_CONFIG_DIR` is unset,
/// with a deprecation warning.
#[test]
fn xdg_config_home_is_a_deprecated_fallback_for_the_data_dir() {
    let env = Env::new("e2e-xdg");
    let dir = env.path("docs");
    fs::create_dir_all(&dir).unwrap();
    let xdg = env.path("xdg");

    let output = env
        .ir()
        .env_remove("IR_CONFIG_DIR")
        .env("XDG_CONFIG_HOME", &xdg)
        .args(["collection", "add", "notes"])
        .arg(&dir)
        .output()
        .unwrap();
    let (success, _, stderr) = support::snapshot(output);

    assert_eq!(
        (
            success,
            xdg.join("ir").join("config.yml").exists(),
            env.config_dir.join("config.yml").exists(),
            stderr.contains("XDG_CONFIG_HOME is deprecated"),
        ),
        (true, true, false, true),
        "stderr: {stderr}"
    );
}
