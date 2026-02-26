// Compare scanned filesystem state vs database state to produce a diff.

use std::collections::HashMap;

#[derive(Debug)]
pub struct IndexDiff {
    pub to_add: Vec<String>,        // rel_paths new to the DB
    pub to_update: Vec<String>,     // rel_paths whose content hash changed
    pub to_deactivate: Vec<String>, // rel_paths no longer on disk
}

/// Given a map of {rel_path → content_hash} from the filesystem scan and
/// a map of {rel_path → stored_hash} from the DB, produce the diff.
pub fn compute(scanned: &HashMap<String, String>, stored: &HashMap<String, String>) -> IndexDiff {
    let mut to_add = Vec::new();
    let mut to_update = Vec::new();
    let mut to_deactivate = Vec::new();

    for (path, scan_hash) in scanned {
        match stored.get(path) {
            None => to_add.push(path.clone()),
            Some(db_hash) if db_hash != scan_hash => to_update.push(path.clone()),
            _ => {} // unchanged
        }
    }

    for path in stored.keys() {
        if !scanned.contains_key(path) {
            to_deactivate.push(path.clone());
        }
    }

    IndexDiff {
        to_add,
        to_update,
        to_deactivate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn h(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn new_file_added() {
        let d = compute(&h(&[("a.md", "h1")]), &h(&[]));
        assert_eq!(d.to_add, vec!["a.md"]);
        assert!(d.to_update.is_empty());
        assert!(d.to_deactivate.is_empty());
    }

    #[test]
    fn removed_file_deactivated() {
        let d = compute(&h(&[]), &h(&[("a.md", "h1")]));
        assert!(d.to_add.is_empty());
        assert!(d.to_update.is_empty());
        assert_eq!(d.to_deactivate, vec!["a.md"]);
    }

    #[test]
    fn changed_hash_updated() {
        let d = compute(&h(&[("a.md", "h2")]), &h(&[("a.md", "h1")]));
        assert!(d.to_add.is_empty());
        assert_eq!(d.to_update, vec!["a.md"]);
        assert!(d.to_deactivate.is_empty());
    }

    #[test]
    fn unchanged_file_skipped() {
        let d = compute(&h(&[("a.md", "h1")]), &h(&[("a.md", "h1")]));
        assert!(d.to_add.is_empty());
        assert!(d.to_update.is_empty());
        assert!(d.to_deactivate.is_empty());
    }

    #[test]
    fn mixed_diff() {
        let scanned = h(&[("add.md", "h1"), ("update.md", "h_new"), ("keep.md", "hk")]);
        let stored = h(&[
            ("update.md", "h_old"),
            ("keep.md", "hk"),
            ("remove.md", "hr"),
        ]);
        let d = compute(&scanned, &stored);
        assert_eq!(d.to_add, vec!["add.md"]);
        assert_eq!(d.to_update, vec!["update.md"]);
        assert_eq!(d.to_deactivate, vec!["remove.md"]);
    }
}
