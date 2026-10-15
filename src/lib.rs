// Library crate root — re-exports internal modules for use by src/bin/* targets.
pub mod config;
pub mod db;
pub mod error;
pub(crate) mod frontmatter;
pub mod get;
pub mod index;
pub mod llm;
pub mod lock;
pub mod preprocess;
pub mod search;
pub mod types;
