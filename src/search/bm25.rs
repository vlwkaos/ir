// BM25 search via FTS5. Re-exports db::fts for the search command.
// Cross-collection fan-out is handled in fan_out.rs.

pub use crate::db::fts::{build_query, search, BM25Query};
