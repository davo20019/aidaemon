//! Every write transaction takes SQLite's write lock up front.
//!
//! Under WAL, a deferred transaction (`pool.begin()`, `BEGIN`, `BEGIN
//! DEFERRED`) that reads before it writes keeps a snapshot; when another
//! connection commits in between, SQLite fails the upgrade with
//! `SQLITE_BUSY_SNAPSHOT` at once, and the busy timeout cannot help. With
//! several connections writing (heartbeat, goals, channels, memory jobs)
//! that turns ordinary contention into "database is locked" errors: in
//! `crates/aidaemon-store-sqlite/tests/write_contention_bench.rs`, eight
//! writers lose about 85% of deferred read-then-write transactions and none
//! with `BEGIN IMMEDIATE`. A transaction that only reads needs no
//! transaction in WAL; one that writes says so when it begins.
//!
//! This test keeps that true for production code in the daemon and every
//! workspace crate (test modules are free to do what they test).

use std::path::{Path, PathBuf};

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "tests") {
                continue;
            }
            rust_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// The part of a file that is not test code: test files are skipped and a
/// file's test module, by convention at the end, is cut off.
fn production_source(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    if name == "tests.rs" || name.ends_with("_tests.rs") {
        return None;
    }
    if path.components().any(|part| {
        part.as_os_str()
            .to_str()
            .is_some_and(|p| p.ends_with("_tests"))
    }) {
        return None;
    }
    let text = std::fs::read_to_string(path).ok()?;
    Some(match text.find("#[cfg(test)]") {
        Some(cut) => text[..cut].to_string(),
        None => text,
    })
}

#[test]
fn production_transactions_begin_immediate() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    if let Ok(crates) = std::fs::read_dir(root.join("crates")) {
        for krate in crates.flatten() {
            rust_files(&krate.path().join("src"), &mut files);
        }
    }
    assert!(
        files.len() > 100,
        "scan found too few files: {}",
        files.len()
    );

    let deferred = [".begin()", "begin_with(\"BEGIN\")", "BEGIN DEFERRED"];
    let mut offenders = Vec::new();
    for path in files {
        let Some(source) = production_source(&path) else {
            continue;
        };
        for (number, line) in source.lines().enumerate() {
            let code = line.split("//").next().unwrap_or_default();
            if deferred.iter().any(|pattern| code.contains(pattern)) {
                offenders.push(format!(
                    "{}:{}: {}",
                    path.strip_prefix(root).unwrap_or(&path).display(),
                    number + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "use `begin_with(\"BEGIN IMMEDIATE\")` for write transactions \
         (see src/state/write_transaction_tests.rs):\n{}",
        offenders.join("\n")
    );
}
