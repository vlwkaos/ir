// In-memory LRU for query expansion (per process lifetime).
// Reranking cache is stored in per-collection llm_cache table via db/mod.rs.

use std::collections::HashMap;

pub struct ExpansionCache {
    inner: HashMap<String, Vec<String>>,
    capacity: usize,
}

impl ExpansionCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    pub fn get(&self, query: &str) -> Option<&Vec<String>> {
        self.inner.get(query)
    }

    pub fn insert(&mut self, query: String, expanded: Vec<String>) {
        if self.inner.len() >= self.capacity {
            // Evict an arbitrary entry (simple strategy, not true LRU)
            if let Some(key) = self.inner.keys().next().cloned() {
                self.inner.remove(&key);
            }
        }
        self.inner.insert(query, expanded);
    }
}
