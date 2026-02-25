// Compare scanned filesystem state vs database state to produce a diff.

use std::collections::HashMap;

#[derive(Debug)]
pub struct IndexDiff {
    pub to_add: Vec<String>,    // rel_paths new to the DB
    pub to_update: Vec<String>, // rel_paths whose content hash changed
    pub to_deactivate: Vec<String>, // rel_paths no longer on disk
}

/// Given a map of {rel_path → content_hash} from the filesystem scan and
/// a map of {rel_path → stored_hash} from the DB, produce the diff.
pub fn compute(
    scanned: &HashMap<String, String>,
    stored: &HashMap<String, String>,
) -> IndexDiff {
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
