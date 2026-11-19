// End-to-end scenarios for the tier-2 reranker eval kit (research/reranker-eval.md):
// IR_RERANK_DUMP gate lines from `ir search`, `ir rerank-eval --from-dump` over a
// frozen window, and `scripts/beir-eval.py rerank` on top of both. No model loads.

mod support;

use serde_json::{Value, json};
use std::fs;
use std::path::Path;
use std::process::Command;
use support::{Env, snapshot, write};

fn doc(
    path: &str,
    fused: f64,
    text: Option<&str>,
    score: Option<f64>,
    ms: f64,
    tokens: u64,
) -> Value {
    json!({"collection": "c", "path": path, "hash": format!("h-{path}"), "fused": fused,
           "text": text, "score": score, "ms": ms, "tokens": tokens})
}

/// One tier-2 dump line. Window a, b, c (c has no text, so no score) and tail d.
/// Blend: a 0.4·0.9 + 0.6·0.1 = 0.42, b 0.4·0.8 + 0.6·0.9 = 0.86, c stays 0.7.
fn t2_line(id: &str, keep_window: bool) -> Value {
    json!({
        "gate": "t2", "id": id, "query": id, "model": "m", "limit": 3, "keep_window": keep_window,
        "window": [
            doc("a.txt", 0.9, Some("A"), Some(0.1), 10.0, 5),
            doc("b.txt", 0.8, Some("B"), Some(0.9), 20.0, 7),
            doc("c.txt", 0.7, None, None, 1.0, 0),
        ],
        "rest": [{"collection": "c", "path": "d.txt", "hash": "h-d", "fused": 0.95}],
    })
}

fn jsonl(lines: &[Value]) -> String {
    lines.iter().map(|l| format!("{l}\n")).collect()
}

fn rerank_eval(env: &Env, input: &Path, window: &str) -> (bool, Vec<Value>, String) {
    let output = env.path("out.jsonl");
    let _ = fs::remove_file(&output);
    let (ok, _, stderr) = env.run(&[
        "rerank-eval",
        "--from-dump",
        "--window",
        window,
        "--input",
        input.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
    ]);
    let rows = fs::read_to_string(&output)
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    (ok, rows, stderr)
}

/// A frozen window replays with search's own blend: judged documents first when
/// keep-window is on, the whole list by score when it is off, `limit` kept, and
/// the costs summed over the first `window` documents only. Gate lines are skipped.
#[test]
fn rerank_eval_replays_the_dump_with_search_blend() {
    let env = Env::new("e2e-rerank-eval");
    let input = env.path("dump.jsonl");
    let gate = json!({"gate": "bm25-strong", "query": "gated"});
    fs::write(
        &input,
        jsonl(&[gate, t2_line("keep", true), t2_line("flat", false)]),
    )
    .unwrap();

    let (ok, rows, stderr) = rerank_eval(&env, &input, "3");
    assert!(ok, "{stderr}");
    let summary: Vec<Value> = rows
        .iter()
        .map(|r| json!([r["id"], r["blend"], r["ms"], r["tokens"], r["window"]]))
        .collect();
    assert_eq!(
        summary,
        vec![
            json!(["keep", ["b.txt", "c.txt", "a.txt"], 31.0, 12, 3]),
            json!(["flat", ["d.txt", "b.txt", "c.txt"], 31.0, 12, 3]),
        ]
    );
    assert_eq!(rows[0]["scores"], json!([0.1, 0.9, null]));

    let (ok, rows, stderr) = rerank_eval(&env, &input, "1");
    assert!(ok, "{stderr}");
    assert_eq!(
        json!([
            rows[0]["blend"],
            rows[0]["ms"],
            rows[0]["tokens"],
            rows[0]["scores"]
        ]),
        json!([["a.txt", "b.txt", "c.txt"], 10.0, 5, [0.1]])
    );
}

#[test]
fn rerank_eval_names_the_bad_line_and_refuses_bad_windows() {
    let env = Env::new("e2e-rerank-eval-bad");
    let input = env.path("dump.jsonl");
    fs::write(
        &input,
        format!(
            "{}{{\"gate\": \"t2\", \"query\": 1}}\n",
            jsonl(&[t2_line("q", true)])
        ),
    )
    .unwrap();

    let (ok, _, stderr) = rerank_eval(&env, &input, "3");
    assert!(!ok && stderr.contains("dump.jsonl:2:"), "{stderr}");
    let (ok, _, stderr) = rerank_eval(&env, &input, "0");
    assert!(!ok && stderr.contains("window"), "{stderr}");

    // ^ A listwise dump ranked its whole window: a smaller replay window would misreport it.
    let mut listwise = t2_line("q", true);
    listwise["model"] = json!("listwise:m.gguf:n20:t160:s1:st10");
    fs::write(&input, jsonl(&[listwise])).unwrap();
    let (ok, _, stderr) = rerank_eval(&env, &input, "2");
    assert!(!ok && stderr.contains("--window >= 3"), "{stderr}");
    let (ok, rows, stderr) = rerank_eval(&env, &input, "3");
    assert!(ok && rows.len() == 1, "{stderr}");
}

/// The BM25 shortcut records a gate line; a query that a fallback answers
/// (here: no daemon, so BM25 stands in for hybrid) leaves no line at all.
#[test]
fn search_records_a_gate_line_only_for_a_shortcut() {
    let env = Env::new("e2e-rerank-gate");
    let notes = env.add_collection("notes", &[]);
    write(
        &notes,
        "rocket.md",
        "# Rocket\n\nzymurgy zymurgy zymurgy brewing science of fermentation.\n",
    );
    // ^ Enough other notes that the rare word's BM25 score passes the shortcut floor.
    for i in 0..30 {
        write(
            &notes,
            &format!("n{i}.md"),
            &format!("# n{i}\n\nThe cat naps. The weather is mild today {i}.\n"),
        );
    }
    env.ok(&["update", "notes"]);
    let dump = env.path("dump.jsonl");
    let search = |query: &str| {
        let output = env
            .ir()
            .env("IR_RERANK_DUMP", &dump)
            .args([
                "search", "--mode", "hybrid", "--json", "--quiet", "-c", "notes", query,
            ])
            .output()
            .unwrap();
        let (ok, stdout, stderr) = snapshot(output);
        assert!(ok, "{stdout}\n{stderr}");
    };

    search("zymurgy");
    search("weather today");
    let lines: Vec<Value> = fs::read_to_string(&dump)
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(
        lines,
        vec![json!({"gate": "bm25-strong", "query": "zymurgy"})]
    );
}

/// `beir-eval.py rerank` refuses a dump run in which a query has no dump line or
/// in which the dump's own scores do not give the run's ranking; otherwise the
/// replay of the whole window equals the run, and the oracle bounds every row.
#[test]
fn beir_eval_rerank_refuses_fallback_queries_and_bounds_the_rows() {
    let env = Env::new("e2e-rerank-beir");
    let data = env.path("data");
    write(
        &data,
        "queries.jsonl",
        &jsonl(&[
            json!({"_id": "q0", "text": "q0"}),
            json!({"_id": "keep", "text": "keep"}),
            json!({"_id": "q2", "text": "q2"}),
        ]),
    );
    write(
        &data,
        "qrels/test.tsv",
        "query-id\tcorpus-id\tscore\nq0\tx\t1\nkeep\ta\t1\nq2\ty\t1\n",
    );
    // ^ As run.py lays it out: the dump run's own hybrid partial next to the variants.
    let out = env.path("res/r.json");
    let base = env.path("res/r.json.partial/hybrid.jsonl");
    let rec =
        |id: &str, ranked: &[&str]| json!({"query_id": id, "ranked": ranked, "elapsed_ms": 1.0});
    let dump = env.path("dump.jsonl");
    let run = |dump_lines: &[Value], keep_ranked: &[&str]| {
        fs::write(&dump, jsonl(dump_lines)).unwrap();
        let _ = fs::remove_dir_all(env.path("res"));
        fs::create_dir_all(base.parent().unwrap()).unwrap();
        fs::write(
            &base,
            jsonl(&[
                rec("q0", &["x"]),
                rec("keep", keep_ranked),
                rec("q2", &["z", "y"]),
            ]),
        )
        .unwrap();
        let output = Command::new("python3")
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/beir-eval.py"))
            .args(["rerank", "--collection", "c", "--ir-bin", env!("CARGO_BIN_EXE_ir")])
            .arg("--data")
            .arg(&data)
            .arg("--dump")
            .arg(&dump)
            .arg("--base")
            .arg(&base)
            .args(["--configs", r#"[{"name":"pw3","window":3,"from_dump":true},{"name":"pw1","window":1,"from_dump":true}]"#])
            .arg("--output")
            .arg(&out)
            .env("IR_CONFIG_DIR", &env.config_dir)
            .output()
            .unwrap();
        let (ok, stdout, stderr) = snapshot(output);
        let results: Value = fs::read_to_string(&out)
            .map(|s| serde_json::from_str(&s).unwrap())
            .unwrap_or(Value::Null);
        (ok, format!("{stdout}\n{stderr}"), results)
    };
    let gate = |q: &str| json!({"gate": "fused-strong", "query": q});
    let blended = ["b", "c", "a"];

    let (ok, log, _) = run(&[gate("q0"), t2_line("keep", true)], &blended);
    assert!(
        !ok && log.contains("1 of 3 queries have no dump line") && log.contains("q2"),
        "{log}"
    );

    let (ok, log, _) = run(
        &[gate("q0"), t2_line("keep", true), gate("q2")],
        &["a", "b", "c"],
    );
    assert!(
        !ok && log.contains("differs from the dump run's ranking on 1 of 1"),
        "{log}"
    );

    let (ok, log, results) = run(&[gate("q0"), t2_line("keep", true), gate("q2")], &blended);
    assert!(ok, "{log}");
    let ndcg: std::collections::BTreeMap<String, f64> = results["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r["mode"].as_str().unwrap().to_string(),
                r["metrics"]["ndcg_10"].as_f64().unwrap(),
            )
        })
        .collect();
    let modes: Vec<&str> = ndcg.keys().map(String::as_str).collect();
    let all_modes = [
        "fused",
        "hybrid",
        "oracle1",
        "oracle3",
        "pw1.blend",
        "pw1.rerank",
        "pw3.blend",
        "pw3.rerank",
    ];
    assert_eq!(modes, all_modes);
    assert_eq!(ndcg["pw3.blend"], ndcg["hybrid"]);
    assert!(ndcg.values().all(|&v| v <= ndcg["oracle3"]), "{ndcg:?}");

    // ^ A shard where no query reached tier 2 still gives every row: the merge needs each shard.
    let (ok, log, results) = run(&[gate("q0"), gate("keep"), gate("q2")], &blended);
    let modes: Vec<&str> = results["results"]
        .as_array()
        .map(|a| a.iter().map(|r| r["mode"].as_str().unwrap()).collect())
        .unwrap_or_default();
    assert!(ok && modes == all_modes, "{modes:?}\n{log}");
}
