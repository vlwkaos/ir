// Auto-detect active collection via longest-prefix match of CWD against collection paths.

use crate::types::Collection;
use std::path::Path;

pub fn detect_collection<'a>(collections: &'a [Collection], cwd: &Path) -> Option<&'a Collection> {
    collections
        .iter()
        .filter_map(|c| {
            let col_path = Path::new(&c.path);
            // Canonicalize both sides; fall back to raw comparison on error.
            let cwd_canon = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
            let col_canon = col_path
                .canonicalize()
                .unwrap_or_else(|_| col_path.to_path_buf());
            if cwd_canon.starts_with(&col_canon) {
                Some((c, col_canon.as_os_str().len()))
            } else {
                None
            }
        })
        .max_by_key(|(_, len)| *len)
        .map(|(c, _)| c)
}
