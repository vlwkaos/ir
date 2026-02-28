// Document chunker: 900-token chunks with 15% overlap and break-point scoring.
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
const DEFAULT_CHUNK_SIZE_TOKENS: usize = 900;
const CHUNK_OVERLAP_PERCENT: usize = 15;
/// Window before the target end position in which to search for a break point.
const BREAK_WINDOW_CHARS: usize = 800;
static CHUNK_SIZE_OVERRIDE_TOKENS: AtomicUsize = AtomicUsize::new(0);

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

    if doc.len() <= chunk_size_chars {
        return vec![Chunk {
            seq: 0,
            pos: 0,
            text: doc.to_string(),
        }];
    }

    let break_points = precompute_break_points(doc);
    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < doc.len() {
        let target_end = (start + chunk_size_chars).min(doc.len());

        let end = if target_end == doc.len() {
            doc.len()
        } else {
            best_break(doc, &break_points, start, target_end)
        };

        let text = doc[start..end].to_string();
        chunks.push(Chunk {
            seq: chunks.len(),
            pos: start,
            text,
        });

        if end == doc.len() {
            break;
        }

        // Next chunk starts with overlap before where this one ended.
        start = end.saturating_sub(chunk_overlap_chars);
        // Snap to a valid char boundary.
        while start < end && !doc.is_char_boundary(start) {
            start += 1;
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
/// Falls back to a char boundary at target_end if no break points found.
fn best_break(doc: &str, break_points: &[(usize, f64)], start: usize, target_end: usize) -> usize {
    let window_start = target_end.saturating_sub(BREAK_WINDOW_CHARS).max(start);
    let window_size = (target_end - window_start) as f64;

    let best = break_points
        .iter()
        .filter(|(pos, _)| *pos > window_start && *pos <= target_end)
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
        // Snap to char boundary.
        let mut p = pos;
        while p < doc.len() && !doc.is_char_boundary(p) {
            p += 1;
        }
        return p;
    }

    // No break point found — split at target_end on a char boundary.
    let mut p = target_end;
    while p < doc.len() && !doc.is_char_boundary(p) {
        p += 1;
    }
    p
}

/// Extract a title from the document.
/// Priority: YAML frontmatter `title` or `name` field → first `# Heading` → first non-empty line → filename.
pub fn extract_title(doc: &str, path_hint: &str) -> String {
    let mut lines = doc.lines().peekable();

    // Parse YAML frontmatter
    if lines.peek() == Some(&"---") {
        lines.next(); // consume opening ---
        let mut fm_title: Option<String> = None;
        for line in lines.by_ref() {
            if line == "---" || line == "..." {
                break;
            }
            // Match `title: value` or `name: value`
            if let Some(rest) = line
                .strip_prefix("title:")
                .or_else(|| line.strip_prefix("name:"))
            {
                let val = rest.trim().trim_matches('"').trim_matches('\'');
                if !val.is_empty() {
                    fm_title = Some(val.to_string());
                    // Keep consuming until end of frontmatter
                }
            }
        }
        if let Some(t) = fm_title {
            return t;
        }
        // Fall through to scan the rest of the document for headings.
    }

    for line in lines {
        let trimmed = line.trim();
        if trimmed.starts_with("# ") {
            return trimmed[2..].trim().to_string();
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
