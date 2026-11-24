//! Change receipts: what a run's write roots hold before and after it.

use super::*;

pub(super) async fn list_backend_files(
    backend: SharedExecutionBackend,
    root: BackendPath,
    skip: &HashSet<&str>,
    max_depth: usize,
    cap: usize,
) -> Vec<String> {
    let mut files = Vec::new();
    let mut queue = std::collections::VecDeque::from([(root.clone(), String::new(), 0usize)]);
    while let Some((directory, prefix, depth)) = queue.pop_front() {
        let Ok(entries) = backend.read_dir(&directory).await else {
            continue;
        };
        for entry in entries {
            if files.len() >= cap {
                return files;
            }
            let name = entry.path.file_name().unwrap_or_default();
            if skip.contains(name) {
                continue;
            }
            let relative = if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}/{name}")
            };
            match entry.metadata.file_type {
                BackendFileType::File => files.push(relative),
                BackendFileType::Directory if depth + 1 < max_depth => {
                    queue.push_back((entry.path, relative, depth + 1));
                }
                _ => {}
            }
        }
    }
    files
}

/// Most entries a run's write roots are read for its change receipt.
pub(super) const CHANGE_RECEIPT_CAP: usize = 100_000;

/// A receipt of what a run's write roots hold: how many entries, their
/// total size and the newest change time. Two that differ show the run
/// changed something there, whatever program it ran and whatever its reply
/// says ([`crate::helper_run::ProgramEnd`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct TreeReceipt {
    pub(super) entries: usize,
    pub(super) bytes: u64,
    pub(super) newest: Option<std::time::SystemTime>,
}

/// A file's size and change time; `None` while it does not exist.
pub(super) type FileMark = (String, Option<(u64, Option<std::time::SystemTime>)>);

/// Most files under a run's write roots marked one by one; past it the
/// tree receipt alone says whether something changed.
const FILE_MARK_CAP: usize = 20_000;

/// The daemon's own database files: a run in the folder that holds them
/// (the daemon started from a checkout) did not change them, though the
/// daemon writes them all the time.
fn daemon_file(path: &str) -> bool {
    crate::state::is_daemon_file(std::path::Path::new(path))
}

/// Folders never marked or counted in receipts: a repository's own records,
/// and dependency and build trees a tool rewrites wholesale (folder names, a
/// machine format).
const UNMARKED: [&str; 10] = [
    ".git",
    "node_modules",
    "target",
    ".venv",
    "__pycache__",
    ".nsh",
    ".next",
    ".turbo",
    ".parcel-cache",
    ".svelte-kit",
];

/// Single-pass filesystem snapshot of a run's write roots: aggregate tree
/// receipt plus per-file marks (up to `FILE_MARK_CAP`), skipping `UNMARKED`
/// subdirectories consistently so `changed_something` and `changed_files` agree.
#[derive(Debug, Clone, Default)]
pub(super) struct RootSnapshot {
    pub(super) receipt: TreeReceipt,
    pub(super) marks: Vec<FileMark>,
}

/// Capture both the aggregate [`TreeReceipt`] and per-file [`FileMark`] list
/// in a single traversal of `roots`.
pub(super) async fn snapshot_roots(
    backend: &SharedExecutionBackend,
    roots: &[String],
) -> RootSnapshot {
    let mut receipt = TreeReceipt::default();
    let mut marks = Vec::new();
    let mut queue: VecDeque<BackendPath> = VecDeque::new();
    for root in roots {
        let path = BackendPath::new(root);
        match backend.metadata(&path).await {
            Ok(meta) if meta.file_type == BackendFileType::Directory => queue.push_back(path),
            Ok(_) if daemon_file(root) => {}
            Ok(meta) => {
                receipt.entries += 1;
                receipt.bytes = receipt.bytes.saturating_add(meta.len);
                receipt.newest = receipt.newest.max(meta.modified);
                if marks.len() < FILE_MARK_CAP {
                    marks.push((root.clone(), Some((meta.len, meta.modified))));
                }
            }
            Err(_) => {
                if marks.len() < FILE_MARK_CAP {
                    marks.push((root.clone(), None));
                }
            }
        }
    }
    while let Some(directory) = queue.pop_front() {
        let Ok(entries) = backend.read_dir(&directory).await else {
            continue;
        };
        for entry in entries {
            if receipt.entries >= CHANGE_RECEIPT_CAP {
                return RootSnapshot { receipt, marks };
            }
            if entry.metadata.file_type == BackendFileType::Directory
                && UNMARKED.contains(&entry.path.file_name().unwrap_or_default())
            {
                continue;
            }
            if entry.metadata.file_type != BackendFileType::Directory
                && daemon_file(&entry.path.to_string())
            {
                continue;
            }
            receipt.entries += 1;
            receipt.bytes = receipt.bytes.saturating_add(entry.metadata.len);
            receipt.newest = receipt.newest.max(entry.metadata.modified);
            match entry.metadata.file_type {
                BackendFileType::Directory => {
                    queue.push_back(entry.path);
                }
                BackendFileType::File if marks.len() < FILE_MARK_CAP => {
                    marks.push((
                        entry.path.to_string(),
                        Some((entry.metadata.len, entry.metadata.modified)),
                    ));
                }
                _ => {}
            }
        }
    }
    RootSnapshot { receipt, marks }
}

/// The files under `roots` that changed, were created, or were removed,
/// computed from two [`FileMark`] snapshots without re-walking the filesystem.
pub(super) fn changed_files_from(before: &[FileMark], after: &[FileMark]) -> Vec<String> {
    let was: std::collections::HashMap<&str, &Option<(u64, Option<std::time::SystemTime>)>> =
        before
            .iter()
            .map(|(path, mark)| (path.as_str(), mark))
            .collect();
    let now: std::collections::HashSet<&str> =
        after.iter().map(|(path, _)| path.as_str()).collect();
    let mut changed: Vec<String> = after
        .iter()
        .filter(|(path, mark)| mark.is_some() && was.get(path.as_str()) != Some(&mark))
        .map(|(path, _)| path.clone())
        .collect();
    // Removed: there before, gone now.
    changed.extend(
        before
            .iter()
            .filter(|(path, mark)| mark.is_some() && !now.contains(path.as_str()))
            .map(|(path, _)| path.clone()),
    );
    changed
}

/// Every file under the run's write roots, each with its size and change time.
#[cfg(test)]
pub(super) async fn file_marks(
    backend: &SharedExecutionBackend,
    roots: &[String],
) -> Vec<FileMark> {
    snapshot_roots(backend, roots).await.marks
}

/// The files under `roots` the run changed, created or removed.
#[cfg(test)]
pub(super) async fn changed_files(
    backend: &SharedExecutionBackend,
    before: &[FileMark],
    roots: &[String],
) -> Vec<String> {
    let after = snapshot_roots(backend, roots).await.marks;
    changed_files_from(before, &after)
}

/// What the run's write roots hold: every file and folder under each root
/// that is a folder, and each root that is a file itself.
#[cfg(test)]
pub(super) async fn tree_receipt(
    backend: &SharedExecutionBackend,
    roots: &[String],
) -> TreeReceipt {
    snapshot_roots(backend, roots).await.receipt
}
