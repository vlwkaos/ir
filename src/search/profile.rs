//! Resolved retrieval-pipeline configuration — the single presence source of
//! truth for the default-pipeline knobs, replacing scattered per-call env reads
//! across the index-build, daemon, and query layers.
//!
//! Resolution precedence is **`config > env > default`** (identical to
//! `RoutingConfig`, so the whole binary shares one rule): a value set in
//! `config.yml` is authoritative and can never be silently masked by a stale
//! environment variable; env is a deprecated convenience layer that only fills in
//! where config is silent; the built-in default applies when neither is set.
//!
//! Query-time knobs are resolved across the searched collections with the
//! "all must agree, else fall back" rule (`agreed`). `expander` is resolved
//! globally (top-level config block) because the model loads once per process.

use crate::types::RetrievalConfig;

/// Fully resolved pipeline configuration — no `Option`, every knob decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetrievalProfile {
    pub ann: bool,
    pub graph_build_from_ann: bool,
    pub t0_graph_expand: bool,
    pub rerank_window: usize,
    pub rerank_keep_window: bool,
    pub expander: bool,
}

/// 0.17 defaults — the pre-0.18 behavior, kept as a reference point (opt back in
/// per-collection via config to restore 0.17 retrieval). No longer the active
/// default as of 0.18.0.
#[allow(dead_code)]
pub const DEFAULT_V017: RetrievalProfile = RetrievalProfile {
    ann: false,
    graph_build_from_ann: false,
    t0_graph_expand: false,
    rerank_window: 20,
    rerank_keep_window: false,
    expander: true,
};

/// 0.18.0 shipping defaults (ADR-0001). ANN on, tier-0 graph expansion on, wide
/// rerank window + keep-window on, LLM expander off. `graph_build_from_ann` stays
/// **false** for 0.18.0 — the doc graph builds via the (correct, tested) matmul
/// path; the O(N·log N) graph-from-ANN build is a 0.18.x follow-up.
pub const DEFAULT_V018: RetrievalProfile = RetrievalProfile {
    ann: true,
    graph_build_from_ann: false,
    t0_graph_expand: true,
    rerank_window: 100,
    rerank_keep_window: true,
    expander: false,
};

/// The active built-in default. 0.17 shipped `DEFAULT_V017`; 0.18.0 flips to
/// `DEFAULT_V018`. Every knob remains overridable via config.yml / env.
pub const DEFAULT: RetrievalProfile = DEFAULT_V018;

// --- env layer (deprecated convenience; each returns None when unset) ----------

/// Parse a boolean env flag, distinguishing "unset" (None) from set-and-false.
/// Truthy set matches `config::env_flag` exactly.
fn env_flag_opt(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|raw| {
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// `IR_ANN` — canonical value `hnsw`, plus the shared boolean spellings.
fn env_ann() -> Option<bool> {
    std::env::var("IR_ANN").ok().map(|raw| {
        let t = raw.trim();
        t.eq_ignore_ascii_case("hnsw")
            || matches!(t.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    })
}

/// `IR_RERANK_WINDOW_OVERRIDE` — positive usize only.
fn env_rerank_window() -> Option<usize> {
    std::env::var("IR_RERANK_WINDOW_OVERRIDE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
}

/// `IR_DISABLE_EXPANDER` is the (inverted) legacy knob: present-and-true disables
/// the expander, so `expander = !disabled`. Unset → None (default decides).
fn env_expander() -> Option<bool> {
    env_flag_opt("IR_DISABLE_EXPANDER").map(|disabled| !disabled)
}

// --- resolution ---------------------------------------------------------------

/// Config layer for a query-time knob: the value only applies if every searched
/// collection that sets it agrees; a conflict falls back to the lower layers.
fn agreed<T: PartialEq + Copy>(mut values: impl Iterator<Item = Option<T>>) -> Option<T> {
    let first = values.next().flatten()?;
    if values.all(|v| v == Some(first)) {
        Some(first)
    } else {
        None
    }
}

/// Combine the three layers for one knob: `config > env > default`.
fn pick<T>(config: Option<T>, env: Option<T>, default: T) -> T {
    config.or(env).unwrap_or(default)
}

/// Resolve the query-time knobs (`ann`, `t0_graph_expand`, `rerank_window`,
/// `rerank_keep_window`) from the searched collections' `retrieval:` blocks.
/// `expander` is not a query decision here — it comes from the daemon resolution.
pub fn resolve_for_query<'a>(
    configs: impl IntoIterator<Item = Option<&'a RetrievalConfig>>,
) -> RetrievalProfile {
    let configs: Vec<Option<&RetrievalConfig>> = configs.into_iter().collect();
    let d = DEFAULT;
    RetrievalProfile {
        ann: pick(
            agreed(configs.iter().map(|c| c.and_then(|c| c.ann))),
            env_ann(),
            d.ann,
        ),
        t0_graph_expand: pick(
            agreed(configs.iter().map(|c| c.and_then(|c| c.t0_graph_expand))),
            env_flag_opt("IR_GRAPH_T0_EXPAND"),
            d.t0_graph_expand,
        ),
        rerank_window: pick(
            agreed(configs.iter().map(|c| c.and_then(|c| c.rerank_window))),
            env_rerank_window(),
            d.rerank_window,
        ),
        rerank_keep_window: pick(
            agreed(configs.iter().map(|c| c.and_then(|c| c.rerank_keep_window))),
            env_flag_opt("IR_RERANK_KEEP_WINDOW"),
            d.rerank_keep_window,
        ),
        graph_build_from_ann: d.graph_build_from_ann,
        expander: d.expander,
    }
}

/// Resolve the daemon-global `expander` decision (which models to load) from the
/// top-level `retrieval:` block. Other fields are left at default — the daemon
/// only consults `expander` at model-load time.
pub fn resolve_for_daemon(global: Option<&RetrievalConfig>) -> RetrievalProfile {
    RetrievalProfile {
        expander: pick(
            global.and_then(|c| c.expander),
            env_expander(),
            DEFAULT.expander,
        ),
        ..DEFAULT
    }
}

/// Resolve the per-collection build-time knobs at `ir embed`: `ann` (build the
/// HNSW sidecar), `graph_build_from_ann` (build strategy), and `t0_graph_expand`
/// (whether the doc graph is needed, so `ir embed` knows to build it).
pub fn resolve_for_build(col: Option<&RetrievalConfig>) -> RetrievalProfile {
    RetrievalProfile {
        ann: pick(col.and_then(|c| c.ann), env_ann(), DEFAULT.ann),
        graph_build_from_ann: pick(
            col.and_then(|c| c.graph_build_from_ann),
            None,
            DEFAULT.graph_build_from_ann,
        ),
        t0_graph_expand: pick(
            col.and_then(|c| c.t0_graph_expand),
            env_flag_opt("IR_GRAPH_T0_EXPAND"),
            DEFAULT.t0_graph_expand,
        ),
        ..DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Env-mutating tests share the process env; serialize them.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // Deliberately does NOT touch IR_ANN: the db::ann tests set/assert on it under
    // their own module lock, and IR_ANN with a separate lock here would race on the
    // process-global var. These tests never set or assert IR_ANN.
    fn clear_env() {
        for k in [
            "IR_GRAPH_T0_EXPAND",
            "IR_RERANK_WINDOW_OVERRIDE",
            "IR_RERANK_KEEP_WINDOW",
            "IR_DISABLE_EXPANDER",
        ] {
            unsafe { std::env::remove_var(k) };
        }
    }

    #[test]
    fn active_default_is_v018() {
        // 0.18.0 ships DEFAULT_V018 as the active default. V017/V018 both pinned
        // to their literal values so an accidental edit trips this test.
        // (Copy into locals so the checks are runtime, not const-folded.)
        let active = DEFAULT;
        assert_eq!(active, DEFAULT_V018);
        assert_eq!(
            DEFAULT_V018,
            RetrievalProfile {
                ann: true,
                graph_build_from_ann: false, // matmul ships in 0.18.0
                t0_graph_expand: true,
                rerank_window: 100,
                rerank_keep_window: true,
                expander: false,
            }
        );
        assert_eq!(
            DEFAULT_V017,
            RetrievalProfile {
                ann: false,
                graph_build_from_ann: false,
                t0_graph_expand: false,
                rerank_window: 20,
                rerank_keep_window: false,
                expander: true,
            }
        );
    }

    #[test]
    fn no_config_no_env_yields_default() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let p = resolve_for_query([None]);
        assert_eq!(p.rerank_window, DEFAULT.rerank_window);
        assert_eq!(p.t0_graph_expand, DEFAULT.t0_graph_expand);
        assert_eq!(resolve_for_daemon(None).expander, DEFAULT.expander);
        clear_env();
    }

    #[test]
    fn config_wins_over_env() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        // Both differ from the default (100) so the assert can only pass if the
        // config layer actually won over env.
        unsafe { std::env::set_var("IR_RERANK_WINDOW_OVERRIDE", "20") };
        let cfg = RetrievalConfig {
            rerank_window: Some(33),
            ..Default::default()
        };
        let p = resolve_for_query([Some(&cfg)]);
        assert_eq!(p.rerank_window, 33);
        clear_env();
    }

    #[test]
    fn env_fills_when_config_silent() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        // env sets a NON-default value (window 50 ≠ default 100; t0 off ≠ default
        // on) so the assert proves the env layer filled in where config was silent.
        unsafe { std::env::set_var("IR_RERANK_WINDOW_OVERRIDE", "50") };
        unsafe { std::env::set_var("IR_GRAPH_T0_EXPAND", "0") };
        let p = resolve_for_query([None]);
        assert_eq!(p.rerank_window, 50);
        assert!(!p.t0_graph_expand);
        clear_env();
    }

    #[test]
    fn query_config_applies_only_on_agreement() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let a = RetrievalConfig {
            rerank_window: Some(33),
            ..Default::default()
        };
        let b = RetrievalConfig {
            rerank_window: Some(44),
            ..Default::default()
        };
        // Disagreement → fall back through env (unset) to default (100).
        let p = resolve_for_query([Some(&a), Some(&b)]);
        assert_eq!(p.rerank_window, DEFAULT.rerank_window);
        // Agreement → the shared value (≠ default) applies.
        let p = resolve_for_query([Some(&a), Some(&a)]);
        assert_eq!(p.rerank_window, 33);
        clear_env();
    }

    #[test]
    fn expander_env_and_config_layers() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        // Default is off (0.18). env "0" turns it ON (≠ default) → env layer fired.
        unsafe { std::env::set_var("IR_DISABLE_EXPANDER", "0") };
        assert!(resolve_for_daemon(None).expander);
        // Global config wins over the env.
        let cfg = RetrievalConfig {
            expander: Some(false),
            ..Default::default()
        };
        assert!(!resolve_for_daemon(Some(&cfg)).expander);
        clear_env();
    }
}
