// Output formatters for search results.
// docs: https://docs.rs/colored/latest/colored/

use crate::types::SearchResult;
use colored::Colorize;

#[derive(Debug, Clone, Copy, Default)]
pub enum Format {
    #[default]
    Pretty,
    Json,
    Csv,
    Markdown,
    Files,
}

pub fn print_results(results: &[SearchResult], fmt: Format, full: bool) {
    match fmt {
        Format::Pretty => pretty(results, full),
        Format::Json => json(results),
        Format::Csv => csv_fmt(results),
        Format::Markdown => markdown(results),
        Format::Files => files(results),
    }
}

fn pretty(results: &[SearchResult], full: bool) {
    if results.is_empty() {
        println!("{}", "no results".dimmed());
        return;
    }
    for r in results {
        // Header: score + docid + path
        println!(
            "{} {} {}",
            format!("{:.3}", r.score).dimmed(),
            r.doc_id.cyan(),
            r.path.bold(),
        );
        // Title (if different from path)
        let filename = std::path::Path::new(&r.path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if r.title != filename {
            println!("    {}", r.title.italic());
        }
        // Content
        if full {
            // Full content not yet available (would need db lookup)
            println!("    {}", "[full content: use ir get]".dimmed());
        } else if let Some(snippet) = &r.snippet {
            let cleaned = snippet.replace("<b>", "").replace("</b>", "");
            println!("    {}", cleaned.dimmed());
        }
        println!();
    }
}

fn json(results: &[SearchResult]) {
    println!("{}", serde_json::to_string_pretty(results).unwrap_or_default());
}

fn csv_fmt(results: &[SearchResult]) {
    println!("collection,path,title,score,doc_id");
    for r in results {
        println!(
            "{},{},{},{:.4},{}",
            escape_csv(&r.collection),
            escape_csv(&r.path),
            escape_csv(&r.title),
            r.score,
            r.doc_id,
        );
    }
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn markdown(results: &[SearchResult]) {
    for r in results {
        println!("- [{title}]({path}) `{score:.3}` {doc_id}",
            title = r.title,
            path = r.path,
            score = r.score,
            doc_id = r.doc_id,
        );
        if let Some(snippet) = &r.snippet {
            let cleaned = snippet.replace("<b>", "**").replace("</b>", "**");
            println!("  > {cleaned}");
        }
    }
}

fn files(results: &[SearchResult]) {
    for r in results {
        println!("{}", r.path);
    }
}
