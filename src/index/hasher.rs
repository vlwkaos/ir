// SHA-256 content hashing for content-addressed storage.

use sha2::{Digest, Sha256};

pub fn hash_bytes(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_str(s: &str) -> String {
        hash_bytes(s.as_bytes())
    }

    #[test]
    fn deterministic() {
        assert_eq!(hash_bytes(b"hello"), hash_bytes(b"hello"));
    }

    #[test]
    fn differs_on_different_input() {
        assert_ne!(hash_bytes(b"hello"), hash_bytes(b"world"));
    }

    #[test]
    fn hash_str_matches_hash_bytes() {
        assert_eq!(hash_str("hello"), hash_bytes(b"hello"));
    }

    #[test]
    fn output_is_64_char_hex() {
        let h = hash_str("test");
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
