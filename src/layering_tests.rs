//! The runtime's top-level modules form layers: a module may use the ones
//! below it, never the ones above. A cycle between modules makes the compiler
//! re-check all of them on any edit, and blocks splitting them into crates.
//!
//! These tests read `aidaemon-runtime-core`'s source (not test code) for
//! `crate::<module>` paths and fail when the module graph has a cycle or a
//! layered module uses one above it. To share something between layers, move
//! the shared item down, or define a trait below and implement it above.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const CORE_SRC: &str = "crates/aidaemon-runtime-core/src";

/// The layered modules, top (uses the others) to bottom.
const LAYERS: &[&str] = &[
    "engine",
    "permissions",
    "specialists",
    "approvals",
    "spawn_tool",
    "turn",
    "host",
    "spawn",
    "background_run",
    "helper_run",
    "run_state",
    "turn_runtime",
    "authority",
    "coordination",
    "untrusted",
    "session_model",
    "llm",
    "media",
    "eval",
    "prompt",
    "known_resources",
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(CORE_SRC)
}

/// Source text with comments, string and char literals blanked out, so that
/// paths and braces are only ever code.
fn code_only(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    let blank = |out: &mut Vec<u8>, c: u8| out.push(if c == b'\n' { b'\n' } else { b' ' });
    while i < b.len() {
        match b[i] {
            b'/' if b.get(i + 1) == Some(&b'/') => {
                while i < b.len() && b[i] != b'\n' {
                    blank(&mut out, b[i]);
                    i += 1;
                }
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                let mut depth = 0;
                while i < b.len() {
                    if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                        depth += 1;
                        out.extend_from_slice(b"  ");
                        i += 2;
                    } else if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                        depth -= 1;
                        out.extend_from_slice(b"  ");
                        i += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        blank(&mut out, b[i]);
                        i += 1;
                    }
                }
            }
            b'r' if matches!(b.get(i + 1), Some(b'"' | b'#'))
                && (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_')) =>
            {
                let mut j = i + 1;
                let mut hashes = 0;
                while b.get(j) == Some(&b'#') {
                    hashes += 1;
                    j += 1;
                }
                if b.get(j) != Some(&b'"') {
                    out.push(b[i]);
                    i += 1;
                    continue;
                }
                j += 1;
                loop {
                    if j >= b.len() {
                        break;
                    }
                    if b[j] == b'"'
                        && b[j + 1..]
                            .iter()
                            .take(hashes)
                            .filter(|&&c| c == b'#')
                            .count()
                            == hashes
                    {
                        j += 1 + hashes;
                        break;
                    }
                    j += 1;
                }
                for &c in &b[i..j.min(b.len())] {
                    blank(&mut out, c);
                }
                i = j;
            }
            b'"' => {
                blank(&mut out, b[i]);
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' {
                        blank(&mut out, b[i]);
                        i += 1;
                    }
                    if i < b.len() {
                        blank(&mut out, b[i]);
                        i += 1;
                    }
                }
                if i < b.len() {
                    blank(&mut out, b[i]);
                    i += 1;
                }
            }
            // A char literal ('x', '\n', '\u{..}'), not a lifetime ('a).
            b'\'' => {
                let end = if b.get(i + 1) == Some(&b'\\') {
                    b[i + 2..]
                        .iter()
                        .position(|&c| c == b'\'')
                        .map(|p| i + 2 + p)
                } else {
                    let ch_len = std::str::from_utf8(&b[i + 1..])
                        .ok()
                        .and_then(|s| s.chars().next())
                        .map_or(1, char::len_utf8);
                    (b.get(i + 1 + ch_len) == Some(&b'\'')).then_some(i + 1 + ch_len)
                };
                match end {
                    Some(end) => {
                        for &c in &b[i..=end] {
                            blank(&mut out, c);
                        }
                        i = end + 1;
                    }
                    None => {
                        out.push(b[i]);
                        i += 1;
                    }
                }
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Index just past the item starting at `from`: its `;`, or its matched `{}`.
fn item_end(code: &[u8], from: usize) -> usize {
    let mut i = from;
    let mut depth = 0usize;
    while i < code.len() {
        match code[i] {
            b';' if depth == 0 => return i + 1,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    code.len()
}

fn ident_at(code: &str, at: usize) -> &str {
    let rest = &code[at..];
    let len = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(rest.len());
    &rest[..len]
}

/// The file's code outside `#[cfg(test)]` items, and the files that its
/// `#[cfg(test)] mod x;` declarations pull in (to skip as tests).
fn non_test_code(path: &Path, code: &str) -> (String, Vec<PathBuf>) {
    let bytes = code.as_bytes();
    let mut kept = String::with_capacity(code.len());
    let mut test_files = Vec::new();
    let mut cursor = 0;
    while let Some(found) = code[cursor..].find("#[cfg(test)]") {
        let start = cursor + found;
        kept.push_str(&code[cursor..start]);
        let end = item_end(bytes, start + "#[cfg(test)]".len());
        let item = &code[start..end];
        if item.trim_end().ends_with(';') {
            let dir = module_dir(path);
            if let Some(p) = item.find("#[path = ") {
                // The literal is blanked; read it from the original source.
                let raw = std::fs::read_to_string(path).unwrap();
                let lit = &raw[start + p + "#[path = \"".len()..];
                let lit = &lit[..lit.find('"').unwrap()];
                test_files.push(path.parent().unwrap().join(lit));
            } else if let Some(m) = item.rfind("mod ") {
                let name = ident_at(item, m + 4);
                test_files.push(dir.join(format!("{name}.rs")));
                test_files.push(dir.join(name));
            }
        }
        cursor = end;
    }
    kept.push_str(&code[cursor..]);
    (kept, test_files)
}

/// Where a file's child modules live.
fn module_dir(path: &Path) -> PathBuf {
    let name = path.file_name().unwrap().to_string_lossy();
    if name == "mod.rs" || name == "lib.rs" {
        path.parent().unwrap().to_path_buf()
    } else {
        path.with_extension("")
    }
}

/// Top-level modules named by `crate::` paths in `code` (including those in
/// `crate::{a, b::c}` groups, and `super::` in top-level files).
fn referenced_modules(code: &str, top_level_file: bool) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut prefixes = vec!["crate::"];
    if top_level_file {
        prefixes.push("super::");
    }
    for prefix in prefixes {
        let mut from = 0;
        while let Some(found) = code[from..].find(prefix) {
            let at = from + found;
            from = at + prefix.len();
            let before = code[..at].chars().next_back();
            if before.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$') {
                continue;
            }
            if code[from..].starts_with('{') {
                let end = from + item_end(&code.as_bytes()[from..], 0);
                let group = &code[from + 1..end - 1];
                let mut depth = 0;
                let mut start = true;
                for (i, c) in group.char_indices() {
                    match c {
                        '{' => depth += 1,
                        '}' => depth -= 1,
                        ',' if depth == 0 => start = true,
                        c if start && !c.is_whitespace() => {
                            start = false;
                            if depth == 0 {
                                out.insert(ident_at(group, i).to_string());
                            }
                        }
                        _ => {}
                    }
                }
            } else {
                out.insert(ident_at(code, from).to_string());
            }
        }
    }
    out
}

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut pending = vec![dir.to_path_buf()];
    let mut files = Vec::new();
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// `(module, referenced module)` -> the files making that reference.
fn module_graph() -> BTreeMap<(String, String), BTreeSet<String>> {
    let root = root();
    let mut modules = BTreeMap::new();
    for entry in std::fs::read_dir(&root).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        if name != "lib" && (path.is_dir() || path.extension().is_some_and(|e| e == "rs")) {
            modules.entry(name).or_insert_with(Vec::new).push(path);
        }
    }
    let mut sources = Vec::new();
    for (module, paths) in &modules {
        for path in paths {
            let files = if path.is_dir() {
                rust_files(path)
            } else {
                vec![path.clone()]
            };
            sources.extend(files.into_iter().map(|f| (module.clone(), f)));
        }
    }
    let mut test_files = Vec::new();
    let mut code_of = Vec::new();
    for (module, file) in sources {
        let code = code_only(&std::fs::read_to_string(&file).unwrap());
        let (code, tests) = non_test_code(&file, &code);
        test_files.extend(tests);
        code_of.push((module, file, code));
    }
    let mut graph: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
    for (module, file, code) in code_of {
        if test_files.iter().any(|t| file.starts_with(t)) {
            continue;
        }
        let top_level = file.parent() == Some(root.as_path());
        for target in referenced_modules(&code, top_level) {
            if target != module && modules.contains_key(&target) {
                let rel = file
                    .strip_prefix(&root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                graph
                    .entry((module.clone(), target))
                    .or_default()
                    .insert(rel);
            }
        }
    }
    graph
}

fn layer(module: &str) -> Option<usize> {
    LAYERS.iter().position(|m| *m == module)
}

/// A cycle in `edges`, if any, as the modules along it.
fn find_cycle(edges: &BTreeSet<(String, String)>) -> Option<Vec<String>> {
    let mut next: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (a, b) in edges {
        next.entry(a).or_default().push(b);
    }
    // 0 unvisited, 1 on the stack, 2 done.
    let mut state: BTreeMap<&str, u8> = BTreeMap::new();
    fn visit<'a>(
        node: &'a str,
        next: &BTreeMap<&'a str, Vec<&'a str>>,
        state: &mut BTreeMap<&'a str, u8>,
        stack: &mut Vec<&'a str>,
    ) -> Option<Vec<String>> {
        state.insert(node, 1);
        stack.push(node);
        for &to in next.get(node).into_iter().flatten() {
            match state.get(to).copied().unwrap_or(0) {
                1 => {
                    let from = stack.iter().position(|n| *n == to).unwrap();
                    let mut cycle: Vec<String> =
                        stack[from..].iter().map(|s| s.to_string()).collect();
                    cycle.push(to.to_string());
                    return Some(cycle);
                }
                0 => {
                    if let Some(c) = visit(to, next, state, stack) {
                        return Some(c);
                    }
                }
                _ => {}
            }
        }
        stack.pop();
        state.insert(node, 2);
        None
    }
    let nodes: Vec<&str> = next.keys().copied().collect();
    for node in nodes {
        if state.get(node).copied().unwrap_or(0) == 0
            && let Some(c) = visit(node, &next, &mut state, &mut Vec::new())
        {
            return Some(c);
        }
    }
    None
}

#[test]
fn layered_modules_only_use_lower_ones() {
    let graph = module_graph();
    let up: Vec<String> = graph
        .iter()
        .filter(|((a, b), _)| matches!((layer(a), layer(b)), (Some(la), Some(lb)) if la > lb))
        .map(|((a, b), files)| format!("{a} -> {b} in {files:?}"))
        .collect();
    assert!(
        up.is_empty(),
        "a module uses one above it in layering_tests::LAYERS; move the shared \
         item down or pass it in through a trait: {up:#?}"
    );
}

#[test]
fn runtime_modules_have_no_cycles() {
    let graph = module_graph();
    let edges: BTreeSet<(String, String)> = graph.keys().cloned().collect();
    if let Some(cycle) = find_cycle(&edges) {
        let detail: Vec<String> = cycle
            .windows(2)
            .map(|w| {
                format!(
                    "{} -> {} in {:?}",
                    w[0],
                    w[1],
                    graph[&(w[0].clone(), w[1].clone())]
                )
            })
            .collect();
        panic!("runtime-core modules form a cycle: {detail:#?}");
    }
}

/// The scan reads real code: the engine's use of the turn module is seen.
#[test]
fn the_scan_sees_engine_using_turn() {
    assert!(module_graph().contains_key(&("engine".to_string(), "turn".to_string())));
}

/// Prints every reference that points up the layers:
/// `cargo test --lib layering_tests::print_back_edges -- --ignored --nocapture`.
#[test]
#[ignore = "a report, not a check"]
fn print_back_edges() {
    for ((a, b), files) in module_graph() {
        if matches!((layer(&a), layer(&b)), (Some(la), Some(lb)) if la > lb) {
            println!("{a} -> {b}: {files:?}");
        }
    }
}

/// Prints every module and the modules it uses:
/// `cargo test --lib layering_tests::print_module_graph -- --ignored --nocapture`.
#[test]
#[ignore = "a report, not a check"]
fn print_module_graph() {
    let mut uses: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (a, b) in module_graph().into_keys() {
        uses.entry(a).or_default().push(b);
    }
    for (module, used) in uses {
        println!("{module}: {}", used.join(" "));
    }
}
