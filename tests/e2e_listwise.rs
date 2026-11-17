// End-to-end scenarios for the listwise tier-2 scorer (src/llm/listwise.rs,
// research/reranker-eval.md), driven through `ir rerank-eval` over a frozen window.
// They load a real model, so they are `#[ignore]`d and need IR_LISTWISE_MODEL (the daemon
// scenario also loads the real embedding model):
//
//   IR_LISTWISE_MODEL=<instruct.gguf> cargo test --test e2e_listwise -- --ignored

mod support;

use serde_json::{Value, json};
use std::fs;
use std::time::Duration;
use support::{Env, eventually, write};

const QUERY: &str = "how does a rocket reach orbit";
const ANSWER: &str = "e.md";

fn model() -> String {
    std::env::var("IR_LISTWISE_MODEL")
        .expect("set IR_LISTWISE_MODEL to an instruct GGUF (for example Qwen3-0.6B-Q8_0.gguf)")
}

/// The answer is last in fused order, so a scorer that keeps the input order fails.
fn window() -> Value {
    let docs = [
        (
            "a.md",
            "The cat purrs on the sofa and naps in the warm afternoon sun.",
        ),
        (
            "b.md",
            "Sourdough bread needs a starter, flour, water, salt and a long rise.",
        ),
        (
            "c.md",
            "The river floods the valley every spring after the mountain snow thaws.",
        ),
        (
            "d.md",
            "Tomatoes grow best in full sun with regular watering and stakes.",
        ),
        (
            ANSWER,
            "A rocket reaches orbit by burning its engines to climb above the atmosphere and then \
             accelerating sideways to about 7.8 km/s, fast enough that it keeps falling around Earth.",
        ),
    ];
    let window: Vec<Value> = docs
        .iter()
        .enumerate()
        .map(|(i, (path, text))| {
            json!({"collection": "c", "path": path, "hash": path, "fused": 0.9 - 0.1 * i as f64,
                   "text": text, "score": null, "ms": null, "tokens": null})
        })
        .collect();
    json!({"gate": "t2", "id": "q", "query": QUERY, "model": "m", "limit": 5,
           "keep_window": true, "window": window, "rest": []})
}

/// Runs `ir rerank-eval` with the listwise scorer; returns (ok, output row, stderr).
fn rerank(env: &Env, vars: &[(&str, &str)]) -> (bool, Value, String) {
    let input = env.path("in.jsonl");
    let output = env.path("out.jsonl");
    fs::write(&input, format!("{}\n", window())).unwrap();
    let _ = fs::remove_file(&output);
    let mut command = env.ir();
    command.env("IR_LISTWISE_MODEL", model()).args([
        "rerank-eval",
        "--window",
        "5",
        "--input",
        input.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
    ]);
    for (k, v) in vars {
        command.env(k, v);
    }
    let (ok, _, stderr) = support::snapshot(command.output().unwrap());
    let row = fs::read_to_string(&output)
        .ok()
        .and_then(|s| serde_json::from_str(s.trim()).ok())
        .unwrap_or(Value::Null);
    (ok, row, stderr)
}

/// The document with the highest listwise score (the blend also weighs the fused score).
fn top(row: &Value) -> String {
    let scores = row["scores"].as_array().cloned().unwrap_or_default();
    let best = (0..scores.len()).max_by(|&a, &b| {
        scores[a]
            .as_f64()
            .unwrap_or(-1.0)
            .total_cmp(&scores[b].as_f64().unwrap_or(-1.0))
    });
    best.map(|i| window()["window"][i]["path"].as_str().unwrap().to_string())
        .unwrap_or_default()
}

/// One prompt holds the whole window: the answer ranks first, every document
/// gets a score, and the prompt tokens are counted.
#[test]
#[ignore = "loads IR_LISTWISE_MODEL"]
fn listwise_ranks_the_answer_first_in_one_prompt() {
    let env = Env::new("e2e-listwise");
    let (ok, row, stderr) = rerank(&env, &[("IR_LISTWISE_N", "5")]);
    assert!(ok, "{stderr}");
    assert_eq!(row["scores"].as_array().map(Vec::len), Some(5), "{row}");
    assert!(row["tokens"].as_u64().unwrap_or(0) > 0, "{row}");
    assert_eq!(top(&row), ANSWER, "{row}");
}

/// A window longer than N slides over the list from the tail: one prompt per window
/// and shuffled order. The answer is the last of five documents, outside the first
/// window of N = 3 (a, b, c), so it can rank above c.md only if a slide carried it
/// there. (It need not win the head window: Qwen3-0.6B favors label A strongly.)
#[test]
#[ignore = "loads IR_LISTWISE_MODEL"]
fn listwise_slides_over_a_window_longer_than_n() {
    let env = Env::new("e2e-listwise-slide");
    let (ok, row, stderr) = rerank(
        &env,
        &[
            ("IR_LISTWISE_N", "3"),
            ("IR_LISTWISE_STRIDE", "2"),
            ("IR_LISTWISE_SHUFFLES", "2"),
        ],
    );
    assert!(ok, "{stderr}");
    let scores: Vec<f64> = row["scores"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default();
    assert_eq!(scores.len(), 5, "{row}");
    // ^ Windows 2..5 and 0..3, two orders each, one prompt per order.
    let one_prompt = rerank(&env, &[("IR_LISTWISE_N", "5")]).1["tokens"]
        .as_u64()
        .unwrap();
    assert!(row["tokens"].as_u64().unwrap() > 2 * one_prompt, "{row}");
    assert!(scores[4] > scores[2], "{row}");
}

#[test]
#[ignore = "loads IR_LISTWISE_MODEL"]
fn listwise_refuses_a_bad_configuration() {
    let env = Env::new("e2e-listwise-bad");
    let (ok, _, stderr) = rerank(&env, &[("IR_LISTWISE_N", "27")]);
    assert!(
        !ok && stderr.contains("IR_LISTWISE_N=27 is out of range"),
        "{stderr}"
    );
    let missing = env.path("missing.gguf");
    let mut command = env.ir();
    command.env("IR_LISTWISE_MODEL", &missing).args([
        "rerank-eval",
        "--window",
        "5",
        "--input",
        "in.jsonl",
        "--output",
        "out.jsonl",
    ]);
    let (ok, _, stderr) = support::snapshot(command.output().unwrap());
    assert!(
        !ok && stderr.contains("IR_LISTWISE_MODEL: no file"),
        "{stderr}"
    );
}

/// A search's daemon reranks with the listwise scorer that IR_LISTWISE_MODEL names, also
/// when IR_COMBINED_MODEL is set: a research run must never get another scorer's row.
#[test]
#[ignore = "loads IR_LISTWISE_MODEL and the embedding model"]
fn search_daemon_reranks_with_the_listwise_scorer_over_combined_mode() {
    let env = Env::with_real_models("e2e-listwise-daemon");
    let notes = env.add_collection("notes", &[]);
    for (name, body) in [
        (
            "a.md",
            "The cat purrs on the sofa and naps in the warm sun.",
        ),
        (
            "b.md",
            "Sourdough bread needs a starter, flour, water and time.",
        ),
        (
            "e.md",
            "A rocket climbs above the atmosphere and accelerates sideways into orbit.",
        ),
    ] {
        write(&notes, name, &format!("# {name}\n\n{body}\n"));
    }
    env.ok(&["sync", "notes"]);
    let dump = env.path("dump.jsonl");
    // ^ The listwise GGUF also loads as a combined model, so without the precedence the
    //   daemon would rerank with it in combined mode.
    let search = |query: &str| {
        let output = env
            .ir()
            .env("IR_LISTWISE_MODEL", model())
            .env("IR_COMBINED_MODEL", model())
            .env("IR_DISABLE_SHORTCUTS", "1")
            .env("IR_RERANK_DUMP", &dump)
            .args(["search", "--json", "--quiet", "-c", "notes", query])
            .output()
            .unwrap();
        let (ok, stdout, stderr) = support::snapshot(output);
        assert!(ok, "{stdout}\n{stderr}");
    };
    search("warming up");
    assert!(
        eventually(Duration::from_secs(180), || {
            let status = env.ok(&["daemon", "status"]);
            status.contains("tier-2 ready") || status.contains("tier-2 unavailable")
        }),
        "the daemon never finished loading tier 2"
    );
    search(QUERY);
    let line: Value = fs::read_to_string(&dump)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find(|l| l["gate"] == "t2" && l["query"] == QUERY)
        .expect("the search dumped its tier-2 window");
    let scorer = line["model"].as_str().unwrap_or_default();
    assert!(scorer.starts_with("listwise:"), "tier 2 used {scorer}");
}
