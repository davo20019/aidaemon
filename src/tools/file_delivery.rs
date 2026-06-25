use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub struct ReadyMedia {
    pub canonical_path: PathBuf,
    pub filename: String,
    pub size_bytes: u64,
    pub recovered_into_inbox: bool,
}

/// Error variants carry the originally-requested path (or recovery context)
/// so callers can construct user-facing messages that include the relevant detail.
#[derive(Debug)]
#[allow(dead_code)]
pub enum DeliveryError {
    FileNotFound(String),
    NotRegularFile(String),
    Blocked(String),
    OutsideAllowedDirs(String),
    /// Recovery copy into the inbox failed. Carries the canonical source path
    /// (so the caller can render a correct `cp` example) and the error message.
    RecoveryFailed {
        path: PathBuf,
        error: String,
    },
    Ambiguous(Vec<PathBuf>),
}

/// Validate + recover a path for delivery. No transport — callers send `ReadyMedia` through their own sink.
pub fn prepare_delivery(
    requested_path: &str,
    cwd: &Path,
    inbox_dir: &Path,
    outbox_dirs: &[PathBuf],
) -> Result<ReadyMedia, DeliveryError> {
    // 1. Expand ~ in the path
    let expanded = shellexpand::tilde(requested_path).to_string();
    let mut path = PathBuf::from(&expanded);

    // 2. If path doesn't exist, try to find by filename in known roots
    if !path.exists() {
        match resolve_missing_path_by_filename(&path, cwd, inbox_dir, outbox_dirs) {
            Some(ResolveResult::Found(found)) => {
                path = found;
            }
            Some(ResolveResult::Ambiguous(candidates)) => {
                return Err(DeliveryError::Ambiguous(candidates));
            }
            None => {
                return Err(DeliveryError::FileNotFound(requested_path.to_string()));
            }
        }
    }

    // 3. Must be a regular file
    let metadata = std::fs::metadata(&path)
        .map_err(|_| DeliveryError::FileNotFound(requested_path.to_string()))?;
    if !metadata.is_file() {
        return Err(DeliveryError::NotRegularFile(requested_path.to_string()));
    }

    // 4. Canonicalize to resolve symlinks and prevent traversal
    let mut canonical = path
        .canonicalize()
        .map_err(|_| DeliveryError::FileNotFound(requested_path.to_string()))?;

    // 5. Block sensitive files BEFORE any recovery
    if is_path_blocked(&canonical) {
        return Err(DeliveryError::Blocked(requested_path.to_string()));
    }

    // 6. If outside allowed dirs, try auto-recovery from temp roots only
    let mut recovered_into_inbox = false;
    if !is_path_allowed(&canonical, inbox_dir, outbox_dirs) {
        if !is_recoverable_source(&canonical) {
            return Err(DeliveryError::OutsideAllowedDirs(
                requested_path.to_string(),
            ));
        }
        match recover_into_inbox(&canonical, inbox_dir) {
            Ok(copied) => {
                canonical = copied;
                recovered_into_inbox = true;
            }
            Err(e) => {
                return Err(DeliveryError::RecoveryFailed {
                    path: canonical,
                    error: e.to_string(),
                });
            }
        }
    }

    // 7. Compute filename and size
    let filename = canonical
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "file".to_string());

    // Re-read size from the canonical (possibly recovered) path
    let size_bytes = std::fs::metadata(&canonical).map(|m| m.len()).unwrap_or(0);

    Ok(ReadyMedia {
        canonical_path: canonical,
        filename,
        size_bytes,
        recovered_into_inbox,
    })
}

// ── Blocked path patterns ──────────────────────────────────────────────────

pub(crate) const BLOCKED_PATTERNS: &[&str] = &[
    ".ssh",
    ".gnupg",
    ".env",
    "credentials",
    ".key",
    ".pem",
    ".aws/credentials",
    ".netrc",
    ".docker/config.json",
    "config.toml",
];

pub(crate) fn is_path_blocked(path: &Path) -> bool {
    let path_str = path.to_string_lossy();
    for pattern in BLOCKED_PATTERNS {
        if pattern.starts_with('.') || pattern.starts_with('/') {
            // Component-based check: .ssh, .gnupg, .env, .aws/credentials, etc.
            if path_str.contains(&format!("/{}", pattern))
                || path_str.contains(&format!("/{}/", pattern))
            {
                return true;
            }
        } else if pattern.starts_with("*.") {
            // Extension-based check (not used currently but future-proof)
            let ext = &pattern[1..]; // ".key", ".pem"
            if path_str.ends_with(ext) {
                return true;
            }
        } else {
            // Exact filename check
            if let Some(name) = path.file_name() {
                if name.to_string_lossy() == *pattern {
                    return true;
                }
            }
            // Also check as path component
            if path_str.contains(&format!("/{}", pattern))
                || path_str.contains(&format!("/{}/", pattern))
            {
                return true;
            }
        }
    }
    // Also block files ending with .key or .pem
    if let Some(ext) = path.extension() {
        let ext = ext.to_string_lossy();
        if ext == "key" || ext == "pem" {
            return true;
        }
    }
    false
}

// ── Allowed-dir check ──────────────────────────────────────────────────────

pub(crate) fn is_path_allowed(canonical: &Path, inbox_dir: &Path, outbox_dirs: &[PathBuf]) -> bool {
    // Allow files in inbox dir (agent returning processed files)
    if canonical.starts_with(inbox_dir) {
        return true;
    }
    // Check against allowed outbox dirs
    outbox_dirs.iter().any(|d| canonical.starts_with(d))
}

// ── Recoverable-source check ───────────────────────────────────────────────

/// Whether a canonical path is eligible for auto-recovery into the inbox.
/// Restricted to system temp roots (the only place agents legitimately write
/// scratch output) so recovery cannot become an arbitrary-file exfiltration
/// path. Roots are canonicalized so macOS `/tmp` → `/private/tmp` matches.
pub(crate) fn is_recoverable_source(canonical: &Path) -> bool {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(t) = std::env::temp_dir().canonicalize() {
        roots.push(t);
    }
    for p in ["/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp"] {
        if let Ok(c) = Path::new(p).canonicalize() {
            roots.push(c);
        }
    }
    roots.iter().any(|r| canonical.starts_with(r))
}

// ── Recovery into inbox ────────────────────────────────────────────────────

/// Copy a readable file from a recoverable temp root into the inbox so it can
/// be delivered. Returns the canonical path of the copy. Caller must have
/// already run the blocked-pattern and `is_recoverable_source` checks on `src`.
pub(crate) fn recover_into_inbox(src: &Path, inbox_dir: &Path) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(inbox_dir)?;
    let filename = src
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("file"));
    let dest = inbox_dir.join(filename);
    std::fs::copy(src, &dest)?;
    dest.canonicalize()
}

// ── Missing-path resolver ──────────────────────────────────────────────────

pub(crate) enum ResolveResult {
    Found(PathBuf),
    Ambiguous(Vec<PathBuf>),
}

/// If the requested absolute path doesn't exist, try a safe, bounded
/// recovery by looking for the same filename in known roots.
pub(crate) fn resolve_missing_path_by_filename(
    requested: &Path,
    cwd: &Path,
    inbox_dir: &Path,
    outbox_dirs: &[PathBuf],
) -> Option<ResolveResult> {
    let file_name = match requested.file_name() {
        Some(name) if !name.is_empty() => name.to_os_string(),
        _ => return None,
    };

    let mut matches: Vec<PathBuf> = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut check_candidate = |candidate: PathBuf| {
        if !candidate.exists() {
            return;
        }
        if let Ok(md) = std::fs::metadata(&candidate) {
            if !md.is_file() {
                return;
            }
        } else {
            return;
        }
        if let Ok(canonical) = candidate.canonicalize() {
            if seen.insert(canonical.clone()) {
                matches.push(canonical);
            }
        }
    };

    check_candidate(cwd.join(&file_name));
    check_candidate(inbox_dir.join(&file_name));
    for outbox in outbox_dirs {
        check_candidate(outbox.join(&file_name));
    }

    match matches.len() {
        0 => None,
        1 => Some(ResolveResult::Found(matches.into_iter().next().unwrap())),
        _ => Some(ResolveResult::Ambiguous(matches)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_delivery_recovers_temp_file_into_inbox() {
        let tmp = tempfile::tempdir().unwrap();
        let inbox = tmp.path().join("inbox");
        // A file in a recognized temp root (the tempdir is under the OS temp root in CI).
        let src = std::env::temp_dir().join("fd_unit_probe_results.txt");
        std::fs::write(&src, b"line1\nline2\n").unwrap();

        let ready = prepare_delivery(src.to_str().unwrap(), tmp.path(), &inbox, &[])
            .expect("temp-root file should be recoverable");

        assert_eq!(ready.filename, "fd_unit_probe_results.txt");
        assert_eq!(ready.size_bytes, 12);
        assert!(ready.recovered_into_inbox);
        assert!(ready
            .canonical_path
            .starts_with(inbox.canonicalize().unwrap()));
        let _ = std::fs::remove_file(&src);
    }

    #[test]
    fn prepare_delivery_blocks_sensitive_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let err = prepare_delivery("/etc/passwd", tmp.path(), &tmp.path().join("inbox"), &[]);
        assert!(matches!(
            err,
            Err(DeliveryError::Blocked(_)) | Err(DeliveryError::OutsideAllowedDirs(_))
        ));
    }

    #[test]
    fn prepare_delivery_reports_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let err = prepare_delivery(
            &tmp.path().join("does_not_exist.txt").to_string_lossy(),
            tmp.path(),
            &tmp.path().join("inbox"),
            &[],
        );
        assert!(matches!(err, Err(DeliveryError::FileNotFound(_))));
    }
}
