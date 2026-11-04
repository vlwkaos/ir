// Document chunker: 512-token chunks with 15% overlap and break-point scoring.
// Ported from qmd's chunkDocument() in store.ts.
//
// Token approximation: 4 chars ≈ 1 token (same as qmd).
// Break-point scoring (higher = preferred split location):
//   h1=100, h2=90, h3=80, h4=70, h5=60, h6=50
//   code fence boundary = 80
//   blank line = 20
//   list item  = 5
//   newline    = 1
//
// Scoring uses quadratic distance decay toward the target position:
//   final = break_score * (1 - (norm_dist^2) * 0.7)

use std::sync::atomic::{AtomicUsize, Ordering};

const CHARS_PER_TOKEN: usize = 4;
const DEFAULT_CHUNK_SIZE_TOKENS: usize = 512;
const CHUNK_OVERLAP_PERCENT: usize = 15;
/// Minimum chunk size: chunks shorter than this are merged with their predecessor.
const MIN_CHUNK_SIZE_TOKENS: usize = 100;
/// Window before the target end position in which to search for a break point.
const BREAK_WINDOW_CHARS: usize = 800;
static CHUNK_SIZE_OVERRIDE_TOKENS: AtomicUsize = AtomicUsize::new(0);

#[allow(dead_code)] // used by eval binary
pub fn set_chunk_size_tokens_override(tokens: Option<usize>) {
    CHUNK_SIZE_OVERRIDE_TOKENS.store(tokens.unwrap_or(0), Ordering::Relaxed);
}

pub fn chunk_size_tokens() -> usize {
    let v = CHUNK_SIZE_OVERRIDE_TOKENS.load(Ordering::Relaxed);
    if v > 0 { v } else { DEFAULT_CHUNK_SIZE_TOKENS }
}

fn chunk_overlap_tokens(chunk_size_tokens: usize) -> usize {
    ((chunk_size_tokens * CHUNK_OVERLAP_PERCENT) + 50) / 100
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub seq: usize,
    /// Byte offset of this chunk's start in the original document.
    pub pos: usize,
    pub text: String,
}

pub fn chunk_document(doc: &str) -> Vec<Chunk> {
    let chunk_size_chars = chunk_size_tokens() * CHARS_PER_TOKEN;
    let chunk_overlap_chars = chunk_overlap_tokens(chunk_size_tokens()) * CHARS_PER_TOKEN;
    let min_chunk_chars = MIN_CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN;

    if doc.len() <= chunk_size_chars {
        return vec![Chunk {
            seq: 0,
            pos: 0,
            text: doc.to_string(),
        }];
    }

    let break_points = precompute_break_points(doc);
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut start = 0usize;
    let mut prev_end = 0usize;

    while start < doc.len() {
        let doc_tail = doc.len() - start;
        // Don't pick a break that produces a sub-floor chunk or re-uses the previous boundary
        // (which would happen when a semantic break falls inside the overlap region).
        let min_break_pos = (start + min_chunk_chars).max(prev_end + 1);

        let end = if doc_tail <= chunk_size_chars {
            doc.len()
        } else {
            let naive_target = start + chunk_size_chars;
            let remaining_after = doc.len() - naive_target;
            if remaining_after >= min_chunk_chars {
                // Healthy tail room — normal split.
                best_break(doc, &break_points, start, naive_target, min_break_pos)
            } else if doc_tail >= 2 * min_chunk_chars {
                // Would leave a sub-min tail but room for two min-sized chunks:
                // rebalance so the tail hits the floor, keeping this chunk ≤ chunk_size.
                let rebalanced_target = doc.len() - min_chunk_chars;
                best_break(doc, &break_points, start, rebalanced_target, min_break_pos)
            } else {
                // Can't fit two min-sized chunks — absorb the rest.
                doc.len()
            }
        };

        chunks.push(Chunk {
            seq: chunks.len(),
            pos: start,
            text: doc[start..end].to_string(),
        });
        prev_end = end;

        if end == doc.len() {
            break;
        }

        let prev_start = start;
        start = end.saturating_sub(chunk_overlap_chars);
        while start < end && !doc.is_char_boundary(start) {
            start += 1;
        }
        // Defense-in-depth: guarantee forward progress for incoherent configs
        // (chunk_size < min_chars + overlap) that would otherwise loop forever.
        if start <= prev_start {
            start = end;
        }
    }

    chunks
}

/// Precompute all break point positions and their scores.
fn precompute_break_points(doc: &str) -> Vec<(usize, f64)> {
    let mut points: Vec<(usize, f64)> = Vec::new();
    let mut in_code_fence = false;
    let mut pos = 0usize;

    for line in doc.lines() {
        let line_start = pos;
        let line_end = pos + line.len();

        if line.starts_with("```") || line.starts_with("~~~") {
            in_code_fence = !in_code_fence;
            // The fence boundary itself is a good split point.
            points.push((line_start, 80.0));
        } else if !in_code_fence {
            let score = line_break_score(line);
            if score > 0.0 {
                points.push((line_start, score));
            }
        }

        // +1 for the newline char (lines() strips it)
        pos = line_end + 1;
    }

    points
}

fn line_break_score(line: &str) -> f64 {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return 20.0;
    }
    if trimmed.starts_with("# ") || trimmed == "#" {
        return 100.0;
    }
    if trimmed.starts_with("## ") || trimmed == "##" {
        return 90.0;
    }
    if trimmed.starts_with("### ") || trimmed == "###" {
        return 80.0;
    }
    if trimmed.starts_with("#### ") || trimmed == "####" {
        return 70.0;
    }
    if trimmed.starts_with("##### ") || trimmed == "#####" {
        return 60.0;
    }
    if trimmed.starts_with("###### ") || trimmed == "######" {
        return 50.0;
    }
    if trimmed.starts_with("- ")
        || trimmed.starts_with("* ")
        || trimmed.starts_with("+ ")
        || trimmed
            .split_once(". ")
            .map(|(n, _)| n.chars().all(|c| c.is_ascii_digit()))
            .unwrap_or(false)
    {
        return 5.0;
    }
    1.0 // bare newline between non-empty lines
}

/// Find the best split position within BREAK_WINDOW_CHARS before target_end.
/// Only considers break points at or after min_break_pos (to prevent sub-floor chunks
/// and to avoid re-selecting the previous chunk's boundary when it falls in the overlap).
/// Falls back to a char boundary at target_end if no qualifying break is found.
fn best_break(
    doc: &str,
    break_points: &[(usize, f64)],
    start: usize,
    target_end: usize,
    min_break_pos: usize,
) -> usize {
    let window_start = target_end.saturating_sub(BREAK_WINDOW_CHARS).max(start);
    let window_size = (target_end - window_start) as f64;

    let best = break_points
        .iter()
        .filter(|(pos, _)| *pos > window_start && *pos <= target_end && *pos >= min_break_pos)
        .map(|(pos, score)| {
            // Distance from target_end, normalized to [0, 1] (0 = at target).
            let dist = (target_end - pos) as f64;
            let norm_dist = if window_size > 0.0 {
                dist / window_size
            } else {
                0.0
            };
            let adjusted = score * (1.0 - norm_dist.powi(2) * 0.7);
            (*pos, adjusted)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((pos, _)) = best {
        let mut p = pos;
        while p < doc.len() && !doc.is_char_boundary(p) {
            p += 1;
        }
        return p;
    }

    // No qualifying break found — fall back to target_end.
    let mut p = target_end;
    while p < doc.len() && !doc.is_char_boundary(p) {
        p += 1;
    }
    p
}

/// Find the YAML frontmatter block in `doc`.
/// Returns `(yaml_content, end_byte_offset)` where `end_byte_offset` is the position
/// in `doc` immediately after the closing frontmatter delimiter line (including its newline).
/// Returns `None` if no valid frontmatter is found.
pub(crate) use crate::frontmatter::extract as extract_frontmatter;
pub(crate) use crate::frontmatter::flatten as flatten_frontmatter;

/// Extract a title from the document.
/// Priority: YAML frontmatter `title` or `name` field → first `# Heading` → first non-empty line → filename.
pub fn extract_title(doc: &str, path_hint: &str) -> String {
    let mut body_start = 0;

    if let Some((yaml_block, end)) = crate::frontmatter::find_block(doc) {
        body_start = end;
        if let Ok(yaml_serde::Value::Mapping(m)) =
            yaml_serde::from_str::<yaml_serde::Value>(yaml_block)
        {
            for key in ["title", "name"] {
                let k = yaml_serde::Value::String(key.to_string());
                if let Some(yaml_serde::Value::String(s)) = m.get(&k)
                    && !s.is_empty()
                {
                    return s.clone();
                }
            }
        }
        // Fall through to scan body for headings.
    }

    for line in doc[body_start..].lines() {
        let trimmed = line.trim();
        if let Some(heading) = trimmed.strip_prefix("# ") {
            return heading.trim().to_string();
        }
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    // Filename without extension as final fallback.
    std::path::Path::new(path_hint)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path_hint.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize tests that mutate the global CHUNK_SIZE_OVERRIDE_TOKENS.
    static CHUNK_OVERRIDE_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn short_doc_is_single_chunk() {
        let doc = "Hello world";
        let chunks = chunk_document(doc);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, doc);
        assert_eq!(chunks[0].pos, 0);
    }

    #[test]
    fn long_doc_splits_into_multiple_chunks() {
        // Build a doc > CHUNK_SIZE_CHARS
        let line = "word ".repeat(100); // 500 chars
        let doc = (line + "\n").repeat(10); // 5010 chars > 3600 (chunk_size)
        let chunks = chunk_document(&doc);
        assert!(
            chunks.len() > 1,
            "expected multiple chunks, got {}",
            chunks.len()
        );
        // Each chunk should be non-empty
        for c in &chunks {
            assert!(!c.text.is_empty());
        }
    }

    #[test]
    fn chunks_cover_full_document() {
        let line = "The quick brown fox jumps over the lazy dog. ".repeat(20);
        let doc = (line + "\n\n## Section\n\n").repeat(10);
        let chunks = chunk_document(&doc);
        // Last chunk should end at doc end
        let last = chunks.last().unwrap();
        assert_eq!(last.pos + last.text.len(), doc.len());
    }

    #[test]
    fn doc_at_chunk_size_boundary_is_single_chunk() {
        let _lock = CHUNK_OVERRIDE_LOCK.lock().unwrap();
        // chunk_size=200 tokens=800 chars. Doc of 500 chars ≤ 800 → early-return single chunk.
        set_chunk_size_tokens_override(Some(200));
        let doc = "word ".repeat(100); // 500 chars
        let chunks = chunk_document(&doc);
        assert_eq!(chunks.len(), 1, "doc ≤ chunk_size should be a single chunk");
        assert_eq!(chunks[0].text.len(), doc.len());
        set_chunk_size_tokens_override(None);
    }

    #[test]
    fn rebalances_when_tail_would_be_sub_min() {
        let _lock = CHUNK_OVERRIDE_LOCK.lock().unwrap();
        // chunk_size=200 tokens=800 chars, min=100 tokens=400 chars.
        // Doc of 1000 chars: naive split → remaining_after=200 < min(400), sub-min tail.
        // doc_tail=1000 ≥ 2*min=800 → rebalance: target pulled to 1000-400=600.
        // Result: ≥2 chunks, all ≥ min_chars.
        set_chunk_size_tokens_override(Some(200));
        let doc = "word ".repeat(200); // 1000 chars
        let chunks = chunk_document(&doc);
        assert!(chunks.len() >= 2, "rebalanced doc should produce ≥2 chunks");
        let min_chars = MIN_CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN;
        for c in &chunks {
            assert!(
                c.text.len() >= min_chars,
                "chunk {} below floor: {} chars",
                c.seq,
                c.text.len()
            );
        }
        set_chunk_size_tokens_override(None);
    }

    #[test]
    fn normal_split_when_tail_is_healthy() {
        let _lock = CHUNK_OVERRIDE_LOCK.lock().unwrap();
        // chunk_size=200 tokens=800 chars, min=400 chars.
        // Doc ~1414 chars: heading break at ~702 (section boundary).
        // remaining_after naive split ≈ 614 ≥ 400 → normal split, rebalance not triggered.
        // min_break_pos prevents the overlapping chunk from re-selecting the heading.
        // All chunks must be ≥ min.
        set_chunk_size_tokens_override(Some(200));
        let section = "word ".repeat(140); // 700 chars
        let doc = format!("{section}\n\n## Section\n\n{section}"); // ~1414 chars
        let chunks = chunk_document(&doc);
        assert!(
            chunks.len() >= 2,
            "healthy-tail doc should produce ≥2 chunks"
        );
        let min_chars = MIN_CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN;
        for c in &chunks {
            assert!(
                c.text.len() >= min_chars,
                "chunk {} below floor: {} chars",
                c.seq,
                c.text.len()
            );
        }
        set_chunk_size_tokens_override(None);
    }

    // ── chunker edge cases ───────────────────────────────────────────────────

    #[test]
    fn doc_exactly_chunk_size_is_single_chunk() {
        let _lock = CHUNK_OVERRIDE_LOCK.lock().unwrap();
        // doc.len() == chunk_size_chars triggers the early-return branch (<=).
        set_chunk_size_tokens_override(Some(100)); // 400 chars
        let doc = "x".repeat(400);
        let chunks = chunk_document(&doc);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text.len(), 400);
        set_chunk_size_tokens_override(None);
    }

    #[test]
    fn doc_one_byte_over_chunk_size_splits() {
        let _lock = CHUNK_OVERRIDE_LOCK.lock().unwrap();
        // doc.len() == chunk_size_chars + 1 triggers the split path.
        set_chunk_size_tokens_override(Some(100)); // 400 chars
        // Must be > 2*min_chunk_chars (800) to avoid absorb, so we need more room.
        // chunk_size=200 → 800 chars. doc=801 chars.
        set_chunk_size_tokens_override(Some(200)); // 800 chars
        let min_chars = MIN_CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN; // 400
        // 800+1=801 chars. remaining_after=1 < min(400) → rebalance if doc_tail ≥ 2*min=800.
        // doc_tail=801 ≥ 800 → rebalance: target = 801-400=401.
        let doc = "x".repeat(801);
        let chunks = chunk_document(&doc);
        assert!(
            chunks.len() >= 2,
            "801-char doc with chunk_size=800 should split"
        );
        for c in &chunks {
            assert!(
                c.text.len() >= min_chars,
                "chunk {} too small: {}",
                c.seq,
                c.text.len()
            );
        }
        set_chunk_size_tokens_override(None);
    }

    #[test]
    fn multibyte_utf8_near_boundary_stays_valid() {
        let _lock = CHUNK_OVERRIDE_LOCK.lock().unwrap();
        // 3-byte CJK chars; chunk boundary must land on a char boundary.
        set_chunk_size_tokens_override(Some(200)); // 800 chars
        // Each "あ" is 3 bytes but counts as 1 char in Rust str.
        // We need enough bytes to exceed chunk_size=800 chars.
        // Use ASCII + a CJK char right at the boundary to force alignment.
        let ascii_part = "a".repeat(798); // 798 chars
        let cjk = "あ"; // 3 bytes, 1 char — sits at char 799 (byte 799..802)
        let rest = "b".repeat(400);
        let doc = format!("{ascii_part}{cjk}{rest}");
        // doc.len() in bytes > 800 (chars ~1199, bytes ~1201)
        let chunks = chunk_document(&doc);
        // All chunk texts must be valid UTF-8 (would panic on bad slice otherwise).
        for c in &chunks {
            assert!(std::str::from_utf8(c.text.as_bytes()).is_ok());
        }
        set_chunk_size_tokens_override(None);
    }

    #[test]
    fn best_break_fallback_when_no_qualifying_break() {
        let _lock = CHUNK_OVERRIDE_LOCK.lock().unwrap();
        // If no break point falls in [min_break_pos, target_end], best_break falls back
        // to a char boundary at target_end. Verify: doc with no semantic breaks at all
        // (continuous ASCII) still produces valid chunk boundaries.
        set_chunk_size_tokens_override(Some(200)); // 800 chars
        let doc = "a".repeat(2400); // no break points — all ASCII, no newlines
        let chunks = chunk_document(&doc);
        assert!(chunks.len() > 1, "featureless doc should still chunk");
        for c in &chunks {
            assert!(!c.text.is_empty());
        }
        set_chunk_size_tokens_override(None);
    }

    #[test]
    fn overlap_larger_than_min_chunk_forward_progress_guard() {
        // If overlap ≥ min_chunk_chars, the overlap-adjusted start can equal or precede
        // prev_start, triggering the forward-progress guard (start = end).
        // This config: chunk_size=50 tokens (200 chars), overlap ~15%=7 tokens=28 chars,
        // min=100 tokens=400 chars. With min > chunk_size the absorb path fires for
        // everything, but let's use chunk_size=300 (1200 chars), overlap=45 tokens=180 chars,
        // min=100 tokens=400 chars — overlap(180) < min(400) so guard won't fire normally.
        // To force the guard: chunk_size=120 tokens (480 chars), overlap=18 tokens=72 chars.
        // min=100 tokens=400 chars. overlap(72) < min(400). Still safe.
        // Real trigger: we need overlap ≥ chunk_size - min so second start ≤ first start.
        // chunk_size=110 tokens=440 chars, overlap=16 tokens=64 chars, min=100 tokens=400 chars.
        // After chunk1 ends at ~440, start2 = 440-64=376 < prev_start=0? No, prev_start=0.
        // For guard: start2 ≤ prev_start means 440-64=376 ≤ 0 → false.
        // The guard fires when rebalance target coincides exactly with the start of the last chunk.
        // Let's just verify the chunker never loops forever on any plausible config.
        let _lock = CHUNK_OVERRIDE_LOCK.lock().unwrap();
        set_chunk_size_tokens_override(Some(120)); // 480 chars
        let doc = "word ".repeat(400); // 2000 chars
        let chunks = chunk_document(&doc);
        assert!(chunks.len() >= 2);
        // Verify sequential, non-overlapping positions
        for w in chunks.windows(2) {
            assert!(w[1].pos > w[0].pos, "positions must strictly increase");
        }
        set_chunk_size_tokens_override(None);
    }

    #[test]
    fn consecutive_headings_exactly_at_chunk_boundary() {
        let _lock = CHUNK_OVERRIDE_LOCK.lock().unwrap();
        // Two headings back-to-back right at the target_end. The min_break_pos guard
        // (prev_end + 1) must prevent re-selecting the first heading on the overlapping chunk.
        set_chunk_size_tokens_override(Some(200)); // 800 chars
        let pad = "word ".repeat(160); // 800 chars
        // Place two headings right at the 800-char mark.
        let doc = format!("{pad}## SectionA\n## SectionB\n{pad}");
        let chunks = chunk_document(&doc);
        // All chunks must have distinct start positions.
        let positions: Vec<usize> = chunks.iter().map(|c| c.pos).collect();
        let unique: std::collections::HashSet<usize> = positions.iter().copied().collect();
        assert_eq!(
            positions.len(),
            unique.len(),
            "duplicate start positions detected"
        );
        set_chunk_size_tokens_override(None);
    }

    #[test]
    fn absorb_when_tail_less_than_two_min_chunks() {
        // When doc_tail < 2*min_chunk_chars, the rest is absorbed into the current chunk.
        // chunk_size=200 tokens=800 chars, min=400 chars, 2*min=800 chars.
        // Doc=1100 chars: after start=0, doc_tail=1100, naive_target=800,
        // remaining_after=300 < min(400), doc_tail=1100 ≥ 2*min=800 → rebalance (not absorb).
        // For absorb: need doc_tail < 800. Use doc=750 chars → single chunk (≤800).
        // Use chunk_size=150 tokens=600 chars, min=100 tokens=400 chars, 2*min=800.
        // doc=700 chars: naive_target=600, remaining_after=100 < min(400),
        // doc_tail=700 < 2*min=800 → absorb to doc.len() → single chunk despite > chunk_size.
        let _lock = CHUNK_OVERRIDE_LOCK.lock().unwrap();
        set_chunk_size_tokens_override(Some(150)); // 600 chars
        let doc = "x".repeat(700);
        let chunks = chunk_document(&doc);
        assert_eq!(chunks.len(), 1, "absorb path should yield single chunk");
        assert_eq!(chunks[0].text.len(), 700);
        set_chunk_size_tokens_override(None);
    }

    #[test]
    fn empty_doc_produces_single_chunk() {
        let chunks = chunk_document("");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "");
    }

    #[test]
    fn chunk_positions_cover_document() {
        let _lock = CHUNK_OVERRIDE_LOCK.lock().unwrap();
        // Every byte in the doc must be covered by at least the last chunk's range.
        set_chunk_size_tokens_override(Some(200));
        let line = "The quick brown fox. ".repeat(20); // 420 chars
        let doc = (line + "\n").repeat(8); // ~3368 chars
        let chunks = chunk_document(&doc);
        let last = chunks.last().unwrap();
        assert_eq!(
            last.pos + last.text.len(),
            doc.len(),
            "last chunk must reach end of doc"
        );
        set_chunk_size_tokens_override(None);
    }

    #[test]
    fn extract_title_from_heading() {
        let doc = "# My Title\n\nContent here.";
        assert_eq!(extract_title(doc, "file.md"), "My Title");
    }

    #[test]
    fn extract_title_fallback_to_filename() {
        let doc = "";
        assert_eq!(extract_title(doc, "my-note.md"), "my-note");
    }
}
