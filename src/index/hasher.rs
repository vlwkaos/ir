// SHA-256 content hashing for content-addressed storage.

use sha2::{Digest, Sha256};

pub fn hash_bytes(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}

pub fn hash_str(s: &str) -> String {
    hash_bytes(s.as_bytes())
}
