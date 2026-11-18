//! The one path resolver every scope check uses ([`crate::ScopeGuard`],
//! [`crate::ApprovalPolicy`]'s rules and `aidaemon_tools::Workspace`).
//!
//! It answers "which file would the operating system touch?": it walks the
//! path one component at a time from the root, follows each symbolic link
//! it meets (with `symlink_metadata` and `read_link`) and applies `..` to
//! the directory actually reached, never to the text of the path. A link
//! whose target does not exist is followed to that target, as `open` with
//! `O_CREAT` would follow it, so the target is what gets checked.
//!
//! # What a path check cannot see
//!
//! A **hard link** is not a path: a file inside an allowed directory that
//! shares its inode with a file outside resolves inside, and writing it
//! changes the outside file. A process that swaps a directory for a link
//! between the check and the I/O is not seen either. Neither is a boundary
//! a path check can hold; run tools that write a directory shared with
//! untrusted processes in a sandbox (`aidaemon_tools::Sandbox`).

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io;
use std::path::{Component, Path, PathBuf};

/// How many symbolic links one resolution follows before giving up (the
/// limit Linux applies, `ELOOP`).
const MAX_LINKS: usize = 40;

/// Resolve an absolute `path` to the file the operating system would touch:
/// every symbolic link followed, `.` and `..` applied to real directories.
/// The path need not exist; names past the last existing directory are
/// appended as given (a `..` among them removes the name before it, which
/// was never a link).
///
/// Fails for a relative path, a link loop, or a component that cannot be
/// inspected (permission denied, a file used as a directory).
pub fn resolve_path(path: &Path) -> io::Result<PathBuf> {
    if !path.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("`{}` is not an absolute path", path.display()),
        ));
    }
    let mut pending: VecDeque<OsString> = VecDeque::new();
    let mut resolved = PathBuf::new();
    push_components(path, &mut resolved, &mut pending);
    // Names appended past the last existing directory.
    let mut missing = 0usize;
    let mut links = 0usize;
    while let Some(part) = pending.pop_front() {
        if part == ".." {
            missing = missing.saturating_sub(1);
            resolved.pop();
            continue;
        }
        let candidate = resolved.join(&part);
        if missing > 0 {
            resolved = candidate;
            missing += 1;
            continue;
        }
        match candidate.symlink_metadata() {
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                resolved = candidate;
                missing = 1;
            }
            Err(error) => return Err(error),
            Ok(meta) if meta.file_type().is_symlink() => {
                links += 1;
                if links > MAX_LINKS {
                    return Err(io::Error::other(format!(
                        "too many levels of symbolic links in `{}`",
                        path.display()
                    )));
                }
                let target = std::fs::read_link(&candidate)?;
                // The link's own components go before what is left.
                let mut ahead = VecDeque::new();
                if target.is_absolute() {
                    resolved = PathBuf::new();
                    push_components(&target, &mut resolved, &mut ahead);
                } else {
                    let mut ignored = PathBuf::new();
                    push_components(&target, &mut ignored, &mut ahead);
                }
                ahead.extend(pending);
                pending = ahead;
            }
            Ok(_) => resolved = candidate,
        }
    }
    Ok(resolved)
}

/// [`resolve_path`] for a path that may be relative to `base` (itself
/// absolute).
pub fn resolve_path_from(path: &Path, base: &Path) -> io::Result<PathBuf> {
    if path.is_absolute() {
        resolve_path(path)
    } else {
        resolve_path(&base.join(path))
    }
}

/// Split `path` into the walk: a root or prefix starts `resolved`, `.` is
/// dropped, everything else queues (`..` as the literal `..`).
fn push_components(path: &Path, resolved: &mut PathBuf, queue: &mut VecDeque<OsString>) {
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => {
                *resolved = PathBuf::from(prefix.as_os_str());
            }
            Component::RootDir => resolved.push(Component::RootDir.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => queue.push_back(OsString::from("..")),
            Component::Normal(name) => queue.push_back(name.to_os_string()),
        }
    }
}

/// Whether `path` resolves inside `root` (both resolved).
pub fn resolves_within(path: &Path, root: &Path) -> bool {
    match (resolve_path(path), resolve_path(root)) {
        (Ok(path), Ok(root)) => path.starts_with(root),
        _ => false,
    }
}
