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
pub struct SearchResult {
    pub collection: String,
    pub path: String,
    pub title: String,
    pub score: f64,
    pub snippet: Option<String>,
    pub hash: String,
    pub doc_id: String,
}

impl SearchResult {
    pub fn sort_desc(results: &mut [Self]) {
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
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
