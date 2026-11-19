// End-to-end scenarios for the listwise tier-2 scorer (src/llm/listwise.rs,
// research/reranker-eval.md), driven through `ir rerank-eval` over a frozen window.
// They load a real model, so they are `#[ignore]`d and need IR_LISTWISE_MODEL (the daemon
// scenario also loads the real embedding model):
//
//   IR_LISTWISE_MODEL=<instruct.gguf> cargo test --test e2e_listwise -- --ignored
//
// The Laya scenarios need IR_LAYA_MODEL (ggml-org/Laya-GGUF, llama.cpp decision type `laya`):
//
//   IR_LAYA_MODEL=<Laya-Q8_0.gguf> cargo test --test e2e_listwise laya -- --ignored

mod support;

use serde_json::{Value, json};
use std::fs;
use std::time::Duration;
use support::{Env, eventually, write};

const QUERY: &str = "how does a rocket reach orbit";
const ANSWER: &str = "e.md";

fn model_from(var: &str, hint: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| panic!("set {var} to {hint}"))
}

fn model() -> String {
    model_from(
        "IR_LISTWISE_MODEL",
        "an instruct GGUF (for example Qwen3-0.6B-Q8_0.gguf)",
    )
}

fn laya_model() -> String {
    model_from("IR_LAYA_MODEL", "Laya-Q8_0.gguf (ggml-org/Laya-GGUF)")
}

/// The answer is last in fused order, so a scorer that keeps the input order fails.
fn window() -> Value {
    window_of(&DOCS)
}

const DOCS: [(&str, &str); 5] = [
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

fn window_of(docs: &[(&str, &str)]) -> Value {
    let window: Vec<Value> = docs
        .iter()
        .enumerate()
        .map(|(i, (path, text))| {
            json!({"collection": "c", "path": path, "hash": path, "fused": 0.9 - 0.1 * i as f64,
                   "text": text, "score": null, "ms": null, "tokens": null})
        })
        .collect();
    json!({"gate": "t2", "id": "q", "query": QUERY, "model": "m", "limit": docs.len(),
           "keep_window": true, "window": window, "rest": []})
}

/// Runs `ir rerank-eval` with the listwise scorer; returns (ok, output row, stderr).
fn rerank(env: &Env, vars: &[(&str, &str)]) -> (bool, Value, String) {
    rerank_window(env, &model(), &window(), vars)
}

fn rerank_window(
    env: &Env,
    model: &str,
    window: &Value,
    vars: &[(&str, &str)],
) -> (bool, Value, String) {
    let input = env.path("in.jsonl");
    let output = env.path("out.jsonl");
    fs::write(&input, format!("{window}\n")).unwrap();
    let _ = fs::remove_file(&output);
    let size = window["window"].as_array().map_or(0, Vec::len).to_string();
    let mut command = env.ir();
    command.env("IR_LISTWISE_MODEL", model).args([
        "rerank-eval",
        "--window",
        &size,
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

/// The prompts IR_LISTWISE_PROMPTS printed: (marker positions, text with special tokens).
fn prompts(stderr: &str) -> Vec<(Vec<usize>, String)> {
    stderr
        .split("[listwise prompt] markers=")
        .skip(1)
        .map(|p| {
            let (markers, rest) = p.split_once('\n').unwrap();
            let text = rest.split("\n[/listwise prompt]").next().unwrap();
            let markers = serde_json::from_str(markers).unwrap();
            (markers, text.to_string())
        })
        .collect()
}

/// Laya reads llama.cpp's systemone layout (GGUF `tokenizer.chat_template.systemone`, d812350):
/// `[CLS]choice question: Q[SEP]`, `[MASK] passage` per option, `[SEP]state[SEP]`, and one score
/// per `[MASK]` from the decision head. The answer, last in fused order, scores highest. Five
/// short passages fit the head whole, though the default N (20) would not.
#[test]
#[ignore = "loads IR_LAYA_MODEL"]
fn laya_scores_each_passage_at_its_mask_marker() {
    let env = Env::new("e2e-listwise-laya");
    let vars = [("IR_LISTWISE_PROMPTS", "1")];
    let (ok, row, stderr) = rerank_window(&env, &laya_model(), &window(), &vars);
    assert!(ok, "{stderr}");
    let options: String = DOCS
        .iter()
        .map(|(_, text)| format!("[MASK] {text}"))
        .collect();
    let expected = format!(
        "[CLS]choice question: Which passage best answers the search query?[SEP]{options}\
         [SEP]Search query: {QUERY}[SEP]"
    );
    let prompts = prompts(&stderr);
    assert_eq!(prompts.len(), 1, "{stderr}");
    assert_eq!(prompts[0].1, expected);
    assert_eq!(prompts[0].0.len(), 5, "{stderr}");
    assert_eq!(top(&row), ANSWER, "{row}");
}

/// Options that overflow Laya's head budget (GGUF `decision.max_head_tokens`, 192 for English
/// Laya) are cut evenly, as llama.cpp's `fill_task_laya` does: 20 long passages keep
/// (192 - 16) / 20 = 8 tokens each, the marker included.
#[test]
#[ignore = "loads IR_LAYA_MODEL"]
fn laya_cuts_options_to_its_head_budget() {
    let env = Env::new("e2e-listwise-laya-cut");
    let long =
        "Rockets burn fuel to climb and then accelerate sideways to orbital speed. ".repeat(8);
    let docs: Vec<(String, &str)> = (0..20)
        .map(|i| (format!("{i}.md"), long.as_str()))
        .collect();
    let docs: Vec<(&str, &str)> = docs.iter().map(|(p, t)| (p.as_str(), *t)).collect();
    let vars = [("IR_LISTWISE_PROMPTS", "1")];
    let (ok, row, stderr) = rerank_window(&env, &laya_model(), &window_of(&docs), &vars);
    assert!(ok, "{stderr}");
    assert_eq!(row["scores"].as_array().map(Vec::len), Some(20), "{row}");
    let prompts = prompts(&stderr);
    assert_eq!(prompts.len(), 1, "{stderr}");
    let markers = &prompts[0].0;
    assert_eq!(markers.len(), 20, "{stderr}");
    assert!(markers.windows(2).all(|w| w[1] - w[0] == 8), "{markers:?}");
}
