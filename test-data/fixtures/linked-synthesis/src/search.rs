// [routing-shortcut]
// Keep this branch cheap; see [routing note](docs/search-routing.md).
pub fn route_query(top_bm25: f64, gap: f64, daemon_ready: bool) -> &'static str {
    if top_bm25 >= 0.75 && gap >= 0.10 {
        return "return_bm25_while_daemon_warms";
    }
    if daemon_ready {
        "hybrid"
    } else {
        "wait"
    }
}
