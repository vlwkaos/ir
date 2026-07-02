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

pub fn print_results(results: &[SearchResult], fmt: Format) {
    match fmt {
        Format::Pretty => pretty(results),
        Format::Json => json(results),
        Format::Csv => csv_fmt(results),
        Format::Markdown => markdown(results),
        Format::Files => files(results),
    }
}

fn pretty(results: &[SearchResult]) {
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
        // Content: full text if fetched, else snippet
        if let Some(content) = &r.content {
            println!("{content}");
        } else if let Some(snippet) = &r.snippet {
            let cleaned = snippet.replace("<b>", "").replace("</b>", "");
            println!("    {}", cleaned.dimmed());
        }
        for item in &r.related {
            let loc = match (item.start_line, item.end_line) {
                (Some(start), Some(end)) if start != end => format!(":{}-{}", start, end),
                (Some(start), _) => format!(":{start}"),
                _ => String::new(),
            };
            let label = item
                .symbol
                .as_deref()
                .or(item.title.as_deref())
                .unwrap_or(item.target.as_str());
            println!(
                "    {} {}{} {}",
                "related".dimmed(),
                item.path.cyan(),
                loc.dimmed(),
                label.dimmed()
            );
        }
        println!();
    }
}

fn json(results: &[SearchResult]) {
    println!(
        "{}",
        serde_json::to_string_pretty(results).unwrap_or_default()
    );
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
        println!(
            "- [{title}]({path}) `{score:.3}` {doc_id}",
            title = r.title,
            path = r.path,
            score = r.score,
            doc_id = r.doc_id,
        );
        if let Some(snippet) = &r.snippet {
            let cleaned = snippet.replace("<b>", "**").replace("</b>", "**");
            println!("  > {cleaned}");
        }
        for item in &r.related {
            let loc = item
                .start_line
                .map(|line| format!(":{line}"))
                .unwrap_or_default();
            let label = item
                .symbol
                .as_deref()
                .or(item.title.as_deref())
                .unwrap_or(item.target.as_str());
            println!("  - related: [{}]({}{})", label, item.path, loc);
        }
    }
}

fn files(results: &[SearchResult]) {
    for r in results {
        println!("{}", r.path);
    }
}
