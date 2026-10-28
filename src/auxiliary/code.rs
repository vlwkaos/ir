// Code-domain extraction: one artifact per file plus one per named symbol
// (function, method, type, trait, constant, module), with its doc comment.
// Bodies are not indexed: full code competes with prose as noise.
// docs: https://docs.rs/tree-sitter/0.25.10/tree_sitter/
//       knowledge: bigpicture/Multiscript and Auxiliary Retrieval Roadmap.md (Branch 2)

use tree_sitter::{Language, Node, Parser};

/// Bump when the extracted text or kinds change: stored sources from another
/// version are extracted again on the next update.
pub const EXTRACTOR_VERSION: &str = "code-ts-3";

/// ^ Larger files are mostly generated or vendored code.
const MAX_FILE_BYTES: usize = 1 << 20;
const MAX_SYMBOLS_PER_FILE: usize = 2000;
const MAX_DOC_CHARS: usize = 1500;
const MAX_SIGNATURE_CHARS: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Tsx,
    Go,
}

impl Lang {
    pub fn from_path(path: &str) -> Option<Self> {
        let ext = path.rsplit_once('.')?.1.to_ascii_lowercase();
        Some(match ext.as_str() {
            "rs" => Self::Rust,
            "py" | "pyi" => Self::Python,
            "js" | "jsx" | "mjs" | "cjs" => Self::JavaScript,
            "ts" | "mts" | "cts" => Self::TypeScript,
            "tsx" => Self::Tsx,
            "go" => Self::Go,
            _ => return None,
        })
    }

    fn language(self) -> Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Self::Go => tree_sitter_go::LANGUAGE.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Artifact {
    pub kind: &'static str,
    pub name: String,
    pub start_line: usize,
    pub end_line: usize,
    /// Search text: kind, name, name words, signature line, doc comment.
    pub text: String,
}

/// Why a file produced no symbols. The file artifact is still stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Skip {
    TooLarge,
    ParseFailed,
}

/// Extract the file artifact (first) and its symbol artifacts.
pub fn extract(rel_path: &str, lang: Lang, src: &str) -> (Vec<Artifact>, Option<Skip>) {
    let total_lines = src.lines().count().max(1);
    let mut symbols = Vec::new();
    let mut module_doc = String::new();
    let skip = if src.len() > MAX_FILE_BYTES {
        Some(Skip::TooLarge)
    } else {
        match parse(lang, src) {
            Some(tree) => {
                let root = tree.root_node();
                module_doc = module_doc_of(lang, root, src);
                collect(lang, root, src, false, &mut symbols);
                None
            }
            None => Some(Skip::ParseFailed),
        }
    };
    symbols.truncate(MAX_SYMBOLS_PER_FILE);

    let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
    let file = Artifact {
        kind: "file",
        name: rel_path.to_string(),
        start_line: 1,
        end_line: total_lines,
        text: join_text(&[
            "file",
            rel_path,
            &split_words(rel_path),
            &truncate(&module_doc, MAX_DOC_CHARS),
            &names.join(" "),
        ]),
    };
    let mut out = Vec::with_capacity(symbols.len() + 1);
    out.push(file);
    out.extend(symbols);
    (out, skip)
}

fn parse(lang: Lang, src: &str) -> Option<tree_sitter::Tree> {
    let mut parser = Parser::new();
    parser.set_language(&lang.language()).ok()?;
    parser.parse(src, None)
}

/// Walk named nodes; `in_container` marks functions inside impl/class/trait
/// bodies, which become methods.
fn collect(lang: Lang, node: Node, src: &str, in_container: bool, out: &mut Vec<Artifact>) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if out.len() >= MAX_SYMBOLS_PER_FILE {
            return;
        }
        let (symbol, container) = classify(lang, child, in_container);
        if let Some(kind) = symbol {
            for name_node in names_of(lang, child) {
                let name = text_of(name_node, src).trim().to_string();
                if name.is_empty() {
                    continue;
                }
                out.push(Artifact {
                    kind,
                    start_line: child.start_position().row + 1,
                    end_line: child.end_position().row + 1,
                    text: join_text(&[
                        kind,
                        &name,
                        &split_words(&name),
                        &truncate(signature(child, src), MAX_SIGNATURE_CHARS),
                        &truncate(&doc_of(lang, child, src), MAX_DOC_CHARS),
                    ]),
                    name,
                });
            }
        }
        collect(lang, child, src, container || in_container, out);
    }
}

/// (symbol kind if `node` declares one, whether its children are members).
fn classify(lang: Lang, node: Node, in_container: bool) -> (Option<&'static str>, bool) {
    let func = if in_container { "method" } else { "function" };
    let kind = node.kind();
    match lang {
        Lang::Rust => match kind {
            "function_item" | "function_signature_item" => (Some(func), false),
            "struct_item" | "enum_item" | "union_item" | "type_item" => (Some("type"), false),
            "trait_item" => (Some("trait"), true),
            "impl_item" => (None, true),
            "const_item" | "static_item" => (Some("constant"), false),
            "mod_item" => (Some("module"), false),
            "macro_definition" => (Some("macro"), false),
            _ => (None, false),
        },
        Lang::Python => match kind {
            "function_definition" => (Some(func), false),
            "class_definition" => (Some("type"), true),
            _ => (None, false),
        },
        Lang::JavaScript | Lang::TypeScript | Lang::Tsx => match kind {
            "function_declaration" | "generator_function_declaration" => (Some(func), false),
            "method_definition" | "method_signature" | "abstract_method_signature" => {
                (Some("method"), false)
            }
            "class_declaration" | "abstract_class_declaration" => (Some("type"), true),
            "interface_declaration" => (Some("trait"), true),
            "type_alias_declaration" | "enum_declaration" => (Some("type"), false),
            "variable_declarator" if is_function_value(node) => (Some(func), false),
            _ => (None, false),
        },
        Lang::Go => match kind {
            "function_declaration" => (Some("function"), false),
            "method_declaration" => (Some("method"), false),
            "type_spec" => (Some("type"), false),
            "const_spec" => (Some("constant"), false),
            _ => (None, false),
        },
    }
}

fn is_function_value(declarator: Node) -> bool {
    declarator.child_by_field_name("value").is_some_and(|v| {
        matches!(
            v.kind(),
            "arrow_function" | "function_expression" | "function"
        )
    })
}

/// Name nodes of a declaration. Go `const_spec` can declare several names.
fn names_of<'a>(lang: Lang, node: Node<'a>) -> Vec<Node<'a>> {
    if lang == Lang::Go && node.kind() == "const_spec" {
        let mut cursor = node.walk();
        return node.children_by_field_name("name", &mut cursor).collect();
    }
    node.child_by_field_name("name").into_iter().collect()
}

fn text_of<'a>(node: Node, src: &'a str) -> &'a str {
    src.get(node.byte_range()).unwrap_or("")
}

/// First line of the declaration, up to the body.
fn signature<'a>(node: Node, src: &'a str) -> &'a str {
    let text = text_of(node, src);
    let head = text.split(['{', '\n']).next().unwrap_or("");
    head.trim().trim_end_matches(':').trim()
}

/// Doc comment of a declaration: Python docstring, else the comment lines
/// directly above it (across attributes, decorators and `export`).
fn doc_of(lang: Lang, node: Node, src: &str) -> String {
    if lang == Lang::Python {
        return node
            .child_by_field_name("body")
            .and_then(|body| docstring(body, src))
            .unwrap_or_default();
    }
    // ^ The comment sits above the wrapping export statement, not the declaration.
    let mut anchor = node;
    while let Some(parent) = anchor.parent() {
        if matches!(
            parent.kind(),
            "export_statement"
                | "lexical_declaration"
                | "variable_declaration"
                | "type_declaration"
                | "const_declaration"
        ) {
            anchor = parent;
        } else {
            break;
        }
    }
    let mut lines = Vec::new();
    let mut next_row = anchor.start_position().row;
    let mut prev = anchor.prev_sibling();
    while let Some(p) = prev {
        // ^ Only a contiguous block directly above counts.
        if p.end_position().row + 1 < next_row {
            break;
        }
        match p.kind() {
            "line_comment" | "block_comment" | "comment" => {
                let text = text_of(p, src);
                if lang == Lang::Rust && !is_rust_doc(text) {
                    break;
                }
                lines.push(clean_comment(text));
            }
            "attribute_item" | "decorator" => {}
            _ => break,
        }
        next_row = p.start_position().row;
        prev = p.prev_sibling();
    }
    lines.reverse();
    lines.join(" ")
}

fn is_rust_doc(text: &str) -> bool {
    text.starts_with("///") || text.starts_with("/**") || text.starts_with("//!")
}

fn docstring(body: Node, src: &str) -> Option<String> {
    let first = body.named_child(0)?;
    if first.kind() != "expression_statement" {
        return None;
    }
    let string = first.named_child(0)?;
    (string.kind() == "string").then(|| clean_docstring(text_of(string, src)))
}

/// Module doc: Rust `//!` lines, Python module docstring, Go package comment.
fn module_doc_of(lang: Lang, root: Node, src: &str) -> String {
    match lang {
        Lang::Python => docstring(root, src).unwrap_or_default(),
        Lang::Rust => {
            let mut cursor = root.walk();
            root.named_children(&mut cursor)
                .take_while(|n| matches!(n.kind(), "line_comment" | "block_comment"))
                .map(|n| text_of(n, src))
                .filter(|t| t.starts_with("//!") || t.starts_with("/*!"))
                .map(clean_comment)
                .collect::<Vec<_>>()
                .join(" ")
        }
        Lang::Go => {
            let mut cursor = root.walk();
            root.named_children(&mut cursor)
                .find(|n| n.kind() == "package_clause")
                .map(|pkg| doc_of(Lang::Go, pkg, src))
                .unwrap_or_default()
        }
        Lang::JavaScript | Lang::TypeScript | Lang::Tsx => String::new(),
    }
}

fn clean_comment(text: &str) -> String {
    let t = text.trim();
    let t = t
        .strip_prefix("/**")
        .or_else(|| t.strip_prefix("/*!"))
        .or_else(|| t.strip_prefix("/*"))
        .map(|s| s.strip_suffix("*/").unwrap_or(s))
        .unwrap_or(t);
    t.lines()
        .map(|l| {
            l.trim()
                .trim_start_matches("///")
                .trim_start_matches("//!")
                .trim_start_matches("//")
                .trim_start_matches('*')
                .trim()
        })
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn clean_docstring(text: &str) -> String {
    let t = text.trim_start_matches(['r', 'u', 'b', 'R', 'U', 'B', 'f', 'F']);
    let t = ["\"\"\"", "'''", "\"", "'"]
        .iter()
        .find_map(|q| t.strip_prefix(q).and_then(|s| s.strip_suffix(q)))
        .unwrap_or(t);
    t.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Documents a source file points to from its comments: `knowledge: <path>`
/// (the convention for linking code to project knowledge, any comment style)
/// and `[[Wiki Title]]`. URLs are not documents and are skipped.
/// `// knowledge: domain/x.md (section)` → `domain/x.md`.
pub fn anchors(src: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in src.lines() {
        let t = line.trim_start();
        let is_comment = ["//", "#", "/*", "*", "--", "<!--"]
            .iter()
            .any(|p| t.starts_with(p));
        if !is_comment {
            continue;
        }
        // ^ `backticked` text is an example of the syntax, not a link.
        let unquoted: String = t.split('`').step_by(2).collect::<Vec<_>>().join(" ");
        let t = unquoted.as_str();
        if let Some(i) = t.find("knowledge:") {
            let rest = t[i + "knowledge:".len()..].trim();
            let target = rest
                .split(" (")
                .next()
                .unwrap_or("")
                .trim_end_matches(['.', ',', ';', ')', '*', '/'])
                .trim();
            if !target.is_empty() && !target.contains("://") {
                out.push(target.to_string());
            }
        }
        let mut rest = t;
        while let Some(start) = rest.find("[[") {
            let after = &rest[start + 2..];
            let Some(end) = after.find("]]") else { break };
            let title = after[..end].split(['|', '#']).next().unwrap_or("").trim();
            if !title.is_empty() {
                out.push(title.to_string());
            }
            rest = &after[end + 2..];
        }
    }
    out.sort();
    out.dedup();
    out
}

/// `extract_frontmatter` → "extract frontmatter"; `CollectionDb` → "collection db";
/// `src/db/schema.rs` → "src db schema rs".
pub fn split_words(ident: &str) -> String {
    let mut words: Vec<String> = Vec::new();
    for part in ident.split(|c: char| !c.is_alphanumeric()) {
        let mut cur = String::new();
        let mut prev_lower = false;
        for c in part.chars() {
            if c.is_uppercase() && prev_lower && !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
            prev_lower = c.is_lowercase() || c.is_ascii_digit();
            cur.extend(c.to_lowercase());
        }
        if !cur.is_empty() {
            words.push(cur);
        }
    }
    words.join(" ")
}

fn truncate(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

fn join_text(parts: &[&str]) -> String {
    parts
        .iter()
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}
