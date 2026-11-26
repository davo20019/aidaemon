//! Files one agent gives another: a screenshot the owner sent, a review a
//! peer wrote. A recipient that can read a file where it is gets its path;
//! a confined one, whose sandbox reaches only its own folder, gets a copy
//! there, under [`SHARED_DIR`] (which ignores itself in git).

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

/// Who a file is shared with: where it works, and whether it reads any
/// file on the machine (a `full_access` run its mode is enforced on) or only
/// its own working folder.
#[derive(Debug, Clone)]
pub(super) struct Recipient {
    pub(super) folder: Option<PathBuf>,
    pub(super) reads_anywhere: bool,
}

impl Recipient {
    fn can_read(&self, path: &Path) -> bool {
        self.reads_anywhere
            || self.folder.as_ref().is_some_and(|folder| {
                path.starts_with(dunce::canonicalize(folder).unwrap_or_else(|_| folder.clone()))
            })
    }
}

/// Tell `recipients` where `sources` are, copying a file only into the
/// working folders of those that cannot read it where it is: the note for
/// their prompt or message. Blocking.
pub(super) fn share(sources: &[SharedSource], recipients: &[Recipient]) -> Result<String, String> {
    if sources.is_empty() {
        return Ok(String::new());
    }
    let batch: String = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
    let mut lines = Vec::new();
    for source in sources {
        let mut folders: Vec<&PathBuf> = Vec::new();
        for recipient in recipients.iter().filter(|r| !r.can_read(&source.path)) {
            let folder = recipient.folder.as_ref().ok_or(
                "a recipient has no working folder to copy the files into; name their paths instead",
            )?;
            if !folders.contains(&folder) {
                folders.push(folder);
            }
        }
        if folders.is_empty() {
            lines.push(format!("- `{}`", source.path.display()));
            continue;
        }
        let relative = Path::new(SHARED_DIR).join(&batch).join(&source.name);
        for folder in &folders {
            copy_into(folder, &batch, &source.path, &relative)?;
        }
        lines.push(if recipients.len() == 1 {
            format!("- `{}`", folders[0].join(&relative).display())
        } else {
            format!(
                "- `{}` (if your sandbox can't read it there: `{}` in your working folder)",
                source.path.display(),
                relative.display()
            )
        });
    }
    Ok(format!("[Files shared with you:\n{}]", lines.join("\n")))
}

/// Copy `source` to `folder/relative`, under the self-ignoring
/// [`SHARED_DIR`].
fn copy_into(folder: &Path, batch: &str, source: &Path, relative: &Path) -> Result<(), String> {
    let shared = folder.join(SHARED_DIR);
    std::fs::create_dir_all(shared.join(batch))
        .map_err(|error| format!("could not make {}: {error}", shared.display()))?;
    // Ignores itself and everything in it, so the copies never show in the
    // recipient's repository.
    let ignore = shared.join(".gitignore");
    if !ignore.exists() {
        std::fs::write(&ignore, "*\n")
            .map_err(|error| format!("could not write {}: {error}", ignore.display()))?;
    }
    prune(&shared, batch);
    let dest = folder.join(relative);
    std::fs::copy(source, &dest).map_err(|error| {
        format!(
            "could not copy `{}` to {}: {error}",
            source.display(),
            dest.display()
        )
    })?;
    Ok(())
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

    fn path_in(note: &str) -> PathBuf {
        let line = note.lines().nth(1).unwrap();
        line.trim_start_matches("- `")
            .split('`')
            .next()
            .unwrap()
            .into()
    }

    fn source(dir: &Path, name: &str, bytes: &[u8]) -> Vec<SharedSource> {
        std::fs::write(dir.join(name), bytes).unwrap();
        check_sources(&[name.to_string()], Some(dir), Some(&[dir.to_path_buf()])).unwrap()
    }

    #[test]
    fn a_recipient_that_can_read_the_file_gets_its_path_and_no_copy() {
        let from = tempfile::tempdir().unwrap();
        let sources = source(from.path(), "screen.png", b"synthetic");
        let to = tempfile::tempdir().unwrap();
        // Unconfined, or confined to the folder the file is in.
        for recipient in [
            Recipient {
                folder: Some(to.path().into()),
                reads_anywhere: true,
            },
            Recipient {
                folder: Some(from.path().into()),
                reads_anywhere: false,
            },
        ] {
            let note = share(&sources, &[recipient]).unwrap();
            assert_eq!(path_in(&note), sources[0].path, "{note}");
            assert!(!to.path().join(SHARED_DIR).exists());
            assert!(!from.path().join(SHARED_DIR).exists());
        }
    }

    #[test]
    fn a_confined_recipient_gets_a_copy_in_its_folder_ignored_by_git() {
        let from = tempfile::tempdir().unwrap();
        let sources = source(from.path(), "screen.png", b"synthetic");
        let to = tempfile::tempdir().unwrap();
        let note = share(
            &sources,
            &[Recipient {
                folder: Some(to.path().into()),
                reads_anywhere: false,
            }],
        )
        .unwrap();
        let copy = path_in(&note);
        assert!(copy.starts_with(to.path().join(SHARED_DIR)), "{note}");
        assert_eq!(std::fs::read(copy).unwrap(), b"synthetic");
        assert_eq!(
            std::fs::read_to_string(to.path().join(SHARED_DIR).join(".gitignore")).unwrap(),
            "*\n"
        );
    }

    #[test]
    fn only_the_recipients_that_need_a_copy_get_one() {
        let from = tempfile::tempdir().unwrap();
        let sources = source(from.path(), "review.md", b"notes");
        let (confined, free) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let note = share(
            &sources,
            &[
                Recipient {
                    folder: Some(confined.path().into()),
                    reads_anywhere: false,
                },
                Recipient {
                    folder: Some(free.path().into()),
                    reads_anywhere: true,
                },
            ],
        )
        .unwrap();
        assert_eq!(path_in(&note), sources[0].path, "{note}");
        assert!(note.contains("in your working folder"), "{note}");
        assert!(confined.path().join(SHARED_DIR).exists());
        assert!(!free.path().join(SHARED_DIR).exists());
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
