//! Tools receive their execution backend through their constructor; none of
//! them reaches for the process-wide installed backend.

use std::path::{Path, PathBuf};

/// Production source of a file: `#[cfg(test)]` items (modules, functions,
/// imports) are removed by brace matching.
fn production_source(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut kept = String::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim() != "#[cfg(test)]" {
            kept.push_str(lines[i]);
            kept.push('\n');
            i += 1;
            continue;
        }
        // Skip further attributes, then the item itself.
        i += 1;
        while i < lines.len() && lines[i].trim_start().starts_with("#[") {
            i += 1;
        }
        let mut depth = 0i64;
        let mut opened = false;
        while i < lines.len() {
            let line = lines[i];
            depth += line.matches('{').count() as i64;
            depth -= line.matches('}').count() as i64;
            opened |= line.contains('{');
            i += 1;
            if (opened && depth <= 0) || (!opened && line.trim_end().ends_with(';')) {
                break;
            }
        }
    }
    kept
}

/// Whether `path`'s module is declared under a `#[cfg(...test...)]` attribute
/// in its parent module file (a test-only module file such as a smoke test).
fn declared_test_only(path: &Path) -> bool {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return false;
    };
    let dir = path.parent().expect("file has a parent");
    let parent = if stem == "mod" {
        return false;
    } else if dir.join("mod.rs").exists() {
        dir.join("mod.rs")
    } else {
        dir.with_extension("rs")
    };
    let Ok(text) = std::fs::read_to_string(parent) else {
        return false;
    };
    let lines: Vec<&str> = text.lines().collect();
    lines.iter().enumerate().any(|(i, line)| {
        let declaration = line.trim();
        (declaration == format!("mod {stem};") || declaration == format!("pub mod {stem};"))
            && i > 0
            && lines[i - 1].trim().starts_with("#[cfg(")
            && lines[i - 1].contains("test")
    })
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read src/tools") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// Whether `text` names `ident` as a whole identifier: a call, a path
/// segment, or a bare name (an alias import such as
/// `use crate::execution::active_execution_backend as backend;`).
fn names_identifier(text: &str, ident: &str) -> bool {
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    text.match_indices(ident).any(|(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + ident.len()..].chars().next();
        !before.is_some_and(is_ident) && !after.is_some_and(is_ident)
    })
}

#[test]
fn names_identifier_matches_calls_aliases_and_paths_only() {
    let ident = "active_execution_backend";
    assert!(names_identifier(
        "let b = active_execution_backend();",
        ident
    ));
    assert!(names_identifier(
        "use crate::execution::active_execution_backend as backend;",
        ident
    ));
    assert!(names_identifier(
        "use crate::execution::{active_execution_backend};",
        ident
    ));
    assert!(!names_identifier("set_active_execution_backend(b);", ident));
    assert!(!names_identifier("active_execution_backend_slot", ident));
}

#[test]
fn no_tool_reads_global_execution_backend() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tools");
    let needle = ["active_execution", "_backend"].concat();
    let mut files = Vec::new();
    rust_files(&root, &mut files);
    files.sort();
    let offenders: Vec<String> = files
        .iter()
        .filter(|path| {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            !name.ends_with("tests.rs") && !declared_test_only(path)
        })
        .filter(|path| {
            let text = std::fs::read_to_string(path).expect("read tool source");
            names_identifier(&production_source(&text), &needle)
        })
        .map(|path| path.strip_prefix(&root).unwrap().display().to_string())
        .collect();
    assert!(
        offenders.is_empty(),
        "tools must take their ExecutionBackend through the constructor, \
         not the installed global; offenders: {offenders:?}"
    );
}
