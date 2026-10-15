// End-to-end scenarios against a long-lived `ir mcp` stdio server (issue #20):
// the same server process must see files created, edited and deleted after it
// started, and must report enough to explain an empty result.

mod support;

use serde_json::{Value, json};
use std::fs;
use support::{Env, Mcp, add_preprocessor, result_paths, write};

fn search(server: &mut Mcp, query: &str) -> Vec<String> {
    let results = server
        .tool(
            "search",
            json!({ "query": query, "mode": "bm25", "collections": ["notes"] }),
        )
        .expect("search tool");
    result_paths(&results)
}

fn update(server: &mut Mcp, arguments: Value) -> Value {
    let results = server.tool("update", arguments).expect("update tool");
    results[0].clone()
}

fn update_row(notes: &std::path::Path, counts: [usize; 4], pending: usize) -> Value {
    json!({
        "collection": "notes",
        "path": notes.to_str().unwrap(),
        "scanned": counts[0],
        "added": counts[1],
        "updated": counts[2],
        "deactivated": counts[3],
        "pending_embeddings": pending,
        "notes": [],
    })
}

#[test]
fn long_lived_server_tracks_files_created_edited_and_deleted_after_start() {
    let env = Env::new("e2e-mcp-lifecycle");
    let notes = env.add_collection("notes", &[]);
    let mut server = Mcp::start(&env);

    let empty = update(&mut server, json!({ "collection": "notes" }));

    write(&notes, "a.md", "# A\n\ncreatedneedle body\n");
    let created = (
        update(&mut server, json!({ "collection": "notes" })),
        search(&mut server, "createdneedle"),
    );

    write(&notes, "a.md", "# A\n\neditedneedle body\n");
    let edited = (
        update(&mut server, json!({})),
        search(&mut server, "editedneedle"),
        search(&mut server, "createdneedle"),
    );

    let unchanged = update(&mut server, json!({ "collection": "notes" }));
    let forced = update(&mut server, json!({ "collection": "notes", "force": true }));

    fs::remove_file(notes.join("a.md")).unwrap();
    let deleted = (
        update(&mut server, json!({ "collection": "notes" })),
        search(&mut server, "editedneedle"),
    );

    let a = vec!["a.md".to_string()];
    assert_eq!(
        (empty, created, edited, unchanged, forced, deleted),
        (
            update_row(&notes, [0, 0, 0, 0], 0),
            (update_row(&notes, [1, 1, 0, 0], 1), a.clone()),
            (update_row(&notes, [1, 0, 1, 0], 1), a, vec![]),
            update_row(&notes, [1, 0, 0, 0], 1),
            update_row(&notes, [1, 1, 0, 0], 1),
            (update_row(&notes, [0, 0, 0, 1], 0), vec![]),
        )
    );
}

#[test]
fn overlapping_update_calls_in_one_server_apply_each_file_once() {
    let env = Env::new("e2e-mcp-overlap");
    let notes = env.add_collection("notes", &[]);
    for i in 0..300 {
        write(
            &notes,
            &format!("n{i:03}.md"),
            &format!("# Note {i}\n\noverlap{i:03} body\n"),
        );
    }
    let mut server = Mcp::start(&env);

    let ids: Vec<u64> = (0..4)
        .map(|_| server.send_tool("update", json!({ "collection": "notes" })))
        .collect();
    let added: Vec<u64> = ids
        .into_iter()
        .map(|id| {
            let rows = server.tool_reply(id).expect("overlapping update");
            rows[0]["added"].as_u64().unwrap()
        })
        .collect();

    assert_eq!(
        (added.iter().sum::<u64>(), search(&mut server, "overlap299")),
        (300, vec!["n299.md".to_string()])
    );
}

#[test]
fn update_tool_tells_the_caller_it_is_text_only_and_what_force_drops() {
    let env = Env::new("e2e-mcp-describe");
    env.add_collection("notes", &[]);
    let mut server = Mcp::start(&env);

    let tools = server.request("tools/list", json!({}));
    let update = tools["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "update")
        .expect("update tool is listed");
    let description = update["description"].as_str().unwrap();
    let unknown = server.tool("update", json!({ "collection": "missing" }));

    assert_eq!(
        (
            description.contains("does not compute embeddings"),
            description.contains("drops all existing vectors"),
            description.contains("`ir embed` or `ir sync`"),
            unknown.is_err(),
        ),
        (true, true, true, true),
        "description: {description}"
    );
}

/// Lines of `stderr` that start with `prefix`, without it.
fn prefixed(stderr: &str, prefix: &str) -> Vec<String> {
    stderr
        .lines()
        .filter_map(|line| line.strip_prefix(prefix))
        .map(str::to_string)
        .collect()
}

/// An MCP caller learns what a terminal user sees on stderr. With a
/// preprocessor that cannot start and a collection that was never indexed,
/// the `update` result carries the same warnings `ir update` prints, and
/// `search` adds the same notes `ir search` prints as a block after the
/// unchanged results.
#[test]
fn mcp_reports_the_warnings_the_cli_prints() {
    let env = Env::new("e2e-mcp-notes");
    add_preprocessor(&env, "pp", "cat");
    let notes = env.add_collection("notes", &["--preprocessor", "pp"]);
    env.add_collection("later", &[]);
    write(&notes, "a.md", "# A\n\nalphaneedle\n");
    let (_, _, first_stderr) = env.run(&["update", "notes"]);
    let mut server = Mcp::start(&env);
    let healthy = update(&mut server, json!({ "collection": "notes" }));

    env.ok(&["preprocessor", "add", "pp", "/nonexistent/pp"]);
    write(&notes, "b.md", "# B\n\nbetaneedle\n");
    let broken = update(&mut server, json!({ "collection": "notes" }));
    // ^ b.md is still listed as indexed without the preprocessor, so the CLI
    //   retries it and must report the same warning.
    let (_, _, cli_update_stderr) = env.run(&["update", "notes"]);

    let mcp_search = server.tool_texts("search", json!({ "query": "betaneedle", "mode": "bm25" }));
    let (_, cli_search_out, cli_search_stderr) =
        env.run(&["search", "--mode", "bm25", "--json", "betaneedle"]);
    let cli_notes = prefixed(&cli_search_stderr, "note: ");

    let broken_notes: Vec<String> = serde_json::from_value(broken["notes"].clone()).unwrap();
    assert_eq!(
        (
            prefixed(&first_stderr, "warning: "),
            healthy["notes"].clone(),
            (broken["added"].clone(), broken_notes.len()),
            prefixed(&cli_update_stderr, "warning: "),
            mcp_search.len(),
            result_paths(&serde_json::from_str(&mcp_search[0]).unwrap()),
            result_paths(&serde_json::from_str(&cli_search_out).unwrap()),
            mcp_search[1].clone(),
            cli_notes.len(),
        ),
        (
            Vec::<String>::new(),
            json!([]),
            (json!(1), 1),
            broken_notes.clone(),
            2,
            vec!["b.md".to_string()],
            vec!["b.md".to_string()],
            format!("notes:\n{}", cli_notes.join("\n")),
            2,
        )
    );
    assert!(
        broken_notes[0].starts_with("1 documents indexed without the preprocessor: ")
            && broken_notes[0].contains("/nonexistent/pp"),
        "update note names the count and the failing command: {broken_notes:?}"
    );
    assert!(
        cli_notes[0].contains("'later' is not indexed")
            && cli_notes[1].contains("preprocessor of collection 'notes' could not start"),
        "search notes name the skipped collection and the preprocessor: {cli_notes:?}"
    );
}
