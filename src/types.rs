use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub globs: Vec<String>,
    #[serde(default)]
    pub excludes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: i64,
    pub collection: String,
    pub path: String,
    pub title: String,
    pub hash: String,
    pub created_at: String,
    pub modified_at: String,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub hash: String,
    pub seq: i64,
    pub pos: i64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub collection: String,
    pub path: String,
    pub title: String,
    pub score: f64,
    pub snippet: Option<String>,
    pub hash: String,
    pub doc_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    Bm25,
    Vector,
    #[default]
    Hybrid,
}

impl std::str::FromStr for SearchMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bm25" => Ok(Self::Bm25),
            "vector" | "vec" => Ok(Self::Vector),
            "hybrid" => Ok(Self::Hybrid),
            _ => Err(format!("unknown mode '{s}'. Use: bm25, vector, hybrid")),
        }
    }
}
