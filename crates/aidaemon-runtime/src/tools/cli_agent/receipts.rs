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

/// What the run's write roots hold: every file and folder under each root
/// that is a folder, and each root that is a file itself (live 2026-11-18:
/// a run given the exact file `notes.txt` to write created it, and its
/// receipt read only folders, so the run read as "did not do its work").
/// A root that does not exist yet counts nothing, so creating it is a
/// change.
pub(super) async fn tree_receipt(
    backend: &SharedExecutionBackend,
    roots: &[String],
) -> TreeReceipt {
    let mut receipt = TreeReceipt::default();
    let mut queue: VecDeque<BackendPath> = VecDeque::new();
    for root in roots {
        let path = BackendPath::new(root);
        match backend.metadata(&path).await {
            Ok(meta) if meta.file_type == BackendFileType::Directory => queue.push_back(path),
            Ok(meta) => {
                receipt.entries += 1;
                receipt.bytes = receipt.bytes.saturating_add(meta.len);
                receipt.newest = receipt.newest.max(meta.modified);
            }
            Err(_) => {}
        }
    }
    while let Some(directory) = queue.pop_front() {
        let Ok(entries) = backend.read_dir(&directory).await else {
            continue;
        };
        for entry in entries {
            if receipt.entries >= CHANGE_RECEIPT_CAP {
                return receipt;
            }
            receipt.entries += 1;
            receipt.bytes = receipt.bytes.saturating_add(entry.metadata.len);
            receipt.newest = receipt.newest.max(entry.metadata.modified);
            if entry.metadata.file_type == BackendFileType::Directory {
                queue.push_back(entry.path);
            }
        }
    }
    receipt
}
