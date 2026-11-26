//! Files one agent gives another: a screenshot the owner sent, a review a
//! peer wrote. A path alone is not enough, since a confined agent's sandbox
//! reaches only its own folder; so each file is copied into the recipient's
//! working folder, under [`SHARED_DIR`] (which ignores itself in git), and
//! the recipient is told where.

use std::path::{Path, PathBuf};
use std::time::Duration;

/// The folder, in a recipient's working folder, that shared files are
/// copied into. Its own `.gitignore` keeps it out of the repository.
pub(super) const SHARED_DIR: &str = ".aidaemon-shared";

/// The most files one call may share.
pub(super) const MAX_SHARED_FILES: usize = 8;

/// The largest file one call may share.
pub(super) const MAX_SHARED_BYTES: u64 = 20 * 1024 * 1024;

/// How long a shared copy is kept: older ones go the next time a file is
/// shared into the same folder.
const KEEP_SHARED_FOR: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// A file a sender may share, checked.
#[derive(Debug, Clone)]
pub(super) struct SharedSource {
    /// Its canonical path.
    pub(super) path: PathBuf,
    /// Its name, as the copy is named.
    pub(super) name: String,
}

/// Check the file `requested` (relative to `cwd` when not absolute) may be
/// shared: a regular file, not a secret, within `MAX_SHARED_BYTES`, and,
/// when `roots` is `Some`, inside one of them or a temp folder (where agents
/// write scratch files). `None`: the caller's own authority already covers
/// the read (the kernel judged it), so any place but a secret's will do.
pub(super) fn check_source(
    requested: &str,
    cwd: Option<&Path>,
    roots: Option<&[PathBuf]>,
) -> Result<SharedSource, String> {
    let expanded = shellexpand::tilde(requested.trim()).to_string();
    let path = match (Path::new(&expanded).is_absolute(), cwd) {
        (false, Some(dir)) => dir.join(&expanded),
        _ => PathBuf::from(&expanded),
    };
    let canonical = dunce::canonicalize(&path).map_err(|_| format!("no file at `{requested}`"))?;
    let metadata =
        std::fs::metadata(&canonical).map_err(|_| format!("no file at `{requested}`"))?;
    if !metadata.is_file() {
        return Err(format!("`{requested}` is not a regular file"));
    }
    if aidaemon_tools::file_delivery::is_path_blocked(&canonical) {
        return Err(format!(
            "`{requested}` may hold secrets and is never shared"
        ));
    }
    if metadata.len() > MAX_SHARED_BYTES {
        return Err(format!(
            "`{requested}` is {} bytes, more than the {MAX_SHARED_BYTES} one file may be",
            metadata.len()
        ));
    }
    if let Some(roots) = roots {
        let inside = roots.iter().any(|root| {
            let root = dunce::canonicalize(root).unwrap_or_else(|_| root.clone());
            canonical.starts_with(root)
        });
        if !inside && !aidaemon_tools::file_delivery::is_recoverable_source(&canonical) {
            return Err(format!(
                "`{requested}` is outside your folders; copy it into your working folder first"
            ));
        }
    }
    let name = canonical
        .file_name()
        .map(|name| safe_name(&name.to_string_lossy()))
        .unwrap_or_else(|| "file".to_string());
    Ok(SharedSource {
        path: canonical,
        name,
    })
}

/// Check every one of `requested` ([`check_source`]), at most
/// [`MAX_SHARED_FILES`].
pub(super) fn check_sources(
    requested: &[String],
    cwd: Option<&Path>,
    roots: Option<&[PathBuf]>,
) -> Result<Vec<SharedSource>, String> {
    let requested: Vec<&String> = requested
        .iter()
        .filter(|path| !path.trim().is_empty())
        .collect();
    if requested.len() > MAX_SHARED_FILES {
        return Err(format!(
            "{} files is more than the {MAX_SHARED_FILES} one call may share",
            requested.len()
        ));
    }
    requested
        .into_iter()
        .map(|path| check_source(path, cwd, roots))
        .collect()
}

/// Copy `sources` into each of `folders` (the recipients' working folders)
/// and return the note that tells a recipient where they are. Blocking.
pub(super) fn stage(sources: &[SharedSource], folders: &[PathBuf]) -> Result<String, String> {
    if sources.is_empty() {
        return Ok(String::new());
    }
    let mut folders: Vec<PathBuf> = folders.to_vec();
    folders.sort();
    folders.dedup();
    if folders.is_empty() {
        return Err(
            "the recipient has no working folder to put the files in; name their paths instead"
                .to_string(),
        );
    }
    let batch: String = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
    let relative: Vec<PathBuf> = sources
        .iter()
        .map(|source| Path::new(SHARED_DIR).join(&batch).join(&source.name))
        .collect();
    for folder in &folders {
        let shared = folder.join(SHARED_DIR);
        std::fs::create_dir_all(shared.join(&batch))
            .map_err(|error| format!("could not make {}: {error}", shared.display()))?;
        // Ignores itself and everything in it, so the copies never show in
        // the recipient's repository.
        let ignore = shared.join(".gitignore");
        if !ignore.exists() {
            std::fs::write(&ignore, "*\n")
                .map_err(|error| format!("could not write {}: {error}", ignore.display()))?;
        }
        prune(&shared, &batch);
        for (source, relative) in sources.iter().zip(&relative) {
            let dest = folder.join(relative);
            std::fs::copy(&source.path, &dest).map_err(|error| {
                format!(
                    "could not copy `{}` to {}: {error}",
                    source.path.display(),
                    dest.display()
                )
            })?;
        }
    }
    let one_folder = (folders.len() == 1).then(|| &folders[0]);
    let lines: Vec<String> = relative
        .iter()
        .map(|relative| match one_folder {
            Some(folder) => format!("- `{}`", folder.join(relative).display()),
            None => format!("- `{}`", relative.display()),
        })
        .collect();
    let heading = match one_folder {
        Some(_) => "[Files shared with you, copied into your working folder:",
        None => "[Files shared with you, copied into your working folder (paths relative to it):",
    };
    Ok(format!("{heading}\n{}]", lines.join("\n")))
}

/// Remove batches in `shared` older than [`KEEP_SHARED_FOR`], but `keep`.
fn prune(shared: &Path, keep: &str) {
    let Ok(entries) = std::fs::read_dir(shared) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_name() == keep {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let old = metadata
            .modified()
            .ok()
            .and_then(|modified| modified.elapsed().ok())
            .is_some_and(|age| age > KEEP_SHARED_FOR);
        if metadata.is_dir() && old {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// `name` as a single safe file name.
fn safe_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_control() || matches!(c, '/' | '\\' | '`') {
                '_'
            } else {
                c
            }
        })
        .collect();
    let cleaned = cleaned.trim().trim_start_matches('.').to_string();
    if cleaned.is_empty() {
        "file".to_string()
    } else {
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shared_file_is_copied_into_each_folder_and_ignored_by_git() {
        let from = tempfile::tempdir().unwrap();
        let source = from.path().join("screen.png");
        std::fs::write(&source, b"\x89PNGsynthetic").unwrap();
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let sources = check_sources(
            &["screen.png".to_string()],
            Some(from.path()),
            Some(&[from.path().to_path_buf()]),
        )
        .unwrap();
        let note = stage(&sources, &[a.path().into(), b.path().into()]).unwrap();
        assert!(note.contains("relative to it"), "{note}");
        let relative = note
            .lines()
            .nth(1)
            .unwrap()
            .trim_start_matches("- `")
            .trim_end_matches("`]")
            .to_string();
        for folder in [a.path(), b.path()] {
            assert_eq!(
                std::fs::read(folder.join(&relative)).unwrap(),
                b"\x89PNGsynthetic"
            );
            assert_eq!(
                std::fs::read_to_string(folder.join(SHARED_DIR).join(".gitignore")).unwrap(),
                "*\n"
            );
        }
    }

    #[test]
    fn one_folder_gets_absolute_paths() {
        let from = tempfile::tempdir().unwrap();
        let source = from.path().join("review.md");
        std::fs::write(&source, b"notes").unwrap();
        let to = tempfile::tempdir().unwrap();
        let sources = check_sources(&[source.to_string_lossy().into_owned()], None, None).unwrap();
        let note = stage(&sources, &[to.path().into()]).unwrap();
        let path = note
            .lines()
            .nth(1)
            .unwrap()
            .trim_start_matches("- `")
            .trim_end_matches("`]")
            .to_string();
        assert!(Path::new(&path).is_absolute(), "{note}");
        assert_eq!(std::fs::read(path).unwrap(), b"notes");
    }

    #[test]
    fn secrets_and_files_outside_the_senders_folders_are_refused() {
        let root = tempfile::tempdir().unwrap();
        let secret = root.path().join(".env");
        std::fs::write(&secret, b"TOKEN=synthetic").unwrap();
        let error = check_source(&secret.to_string_lossy(), None, None).unwrap_err();
        assert!(error.contains("secrets"), "{error}");

        // Outside the sender's folders and not in a temp folder: the home
        // folder stands in, as a place that is neither.
        let home = dirs::home_dir().unwrap();
        let outside = tempfile::Builder::new()
            .prefix("aidaemon-shared-test")
            .tempfile_in(&home)
            .unwrap();
        let mine = tempfile::tempdir().unwrap();
        let error = check_source(
            &outside.path().to_string_lossy(),
            None,
            Some(&[mine.path().to_path_buf()]),
        )
        .unwrap_err();
        assert!(error.contains("outside your folders"), "{error}");
    }

    #[test]
    fn too_many_files_are_refused() {
        let many: Vec<String> = (0..=MAX_SHARED_FILES).map(|i| format!("f{i}")).collect();
        assert!(check_sources(&many, None, None).is_err());
    }
}
