//! A project's files, read-only, for the app's Files view (protocol 1.32):
//! one folder at a time, and one file's text.
//!
//! Every path stays inside the project's folder: a client sends paths
//! relative to it, `..` is refused, and a symlink is followed only when it
//! resolves inside the folder (one that leads out is left out of a listing
//! and refused when read). `.git` never shows. Files git ignores are left
//! out unless asked for. Files that hold secrets show by name, never by
//! text, judged by the host's [`crate::ClientHost::is_secret_path`].
//!
//! A file comes with `version`, a hash of all its bytes: the base a later
//! change to the file will name, so a write made against a file that
//! changed meanwhile can be refused instead of overwriting it.

use std::collections::{BTreeMap, HashSet};
use std::io::Read as _;
use std::path::{Component, Path, PathBuf};
use std::process::Command as Process;
use std::time::UNIX_EPOCH;

use aidaemon_client_protocol::{
    EntryKind, ErrorCode, FileStatus, ProjectDir, ProjectEntry, ProjectFile, ProtocolError,
    Withheld,
};
use sha2::{Digest, Sha256};

/// The most entries one listing sends.
pub const MAX_ENTRIES: usize = 2_000;
/// The most text one read sends; a longer file comes cut, `truncated`.
pub const MAX_TEXT_BYTES: usize = 512 * 1024;

/// Folders never shown: git's own.
const HIDDEN: &[&str] = &[".git"];

fn invalid(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ErrorCode::Invalid, message)
}

fn not_found(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ErrorCode::NotFound, message)
}

/// The project's folder, resolved.
fn root_of(folder: &Path) -> Result<PathBuf, ProtocolError> {
    let root = dunce::canonicalize(folder)
        .map_err(|_| not_found("The project's folder does not exist"))?;
    if !root.is_dir() {
        return Err(not_found("The project's folder is not a folder"));
    }
    Ok(root)
}

/// `rel`'s parts: plain names only, `/`-separated (empty: the folder).
fn parts(rel: &str) -> Result<Vec<&str>, ProtocolError> {
    let mut out = Vec::new();
    for part in rel.split('/').filter(|p| !p.is_empty() && *p != ".") {
        let plain = matches!(
            Path::new(part).components().collect::<Vec<_>>().as_slice(),
            [Component::Normal(_)]
        );
        if !plain || part.contains('\0') || part.contains('\\') {
            return Err(invalid("Paths are relative to the project's folder"));
        }
        if HIDDEN.contains(&part) {
            return Err(not_found("No such file in this project"));
        }
        out.push(part);
    }
    Ok(out)
}

/// `rel` under `root`, resolved, and its clean relative form. Refused when
/// it resolves outside `root`.
fn resolve(root: &Path, rel: &str) -> Result<(PathBuf, String), ProtocolError> {
    let parts = parts(rel)?;
    let joined = parts
        .iter()
        .fold(root.to_path_buf(), |p, part| p.join(part));
    let target =
        dunce::canonicalize(&joined).map_err(|_| not_found("No such file in this project"))?;
    if !target.starts_with(root) {
        return Err(not_found("No such file in this project"));
    }
    Ok((target, parts.join("/")))
}

fn modified_ms(meta: &std::fs::Metadata) -> Option<u64> {
    meta.modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as u64)
}

/// One folder of the project at `folder`.
pub fn list(
    folder: &Path,
    project_id: &str,
    rel: &str,
    with_ignored: bool,
    secret: &dyn Fn(&Path) -> bool,
) -> Result<ProjectDir, ProtocolError> {
    let root = root_of(folder)?;
    let (dir, rel) = resolve(&root, rel)?;
    if !dir.is_dir() {
        return Err(invalid("That is a file, not a folder"));
    }
    let read = std::fs::read_dir(&dir).map_err(|_| not_found("This folder cannot be read"))?;
    let kept = not_ignored(&dir);
    let git = GitState::of(&dir);
    let mut entries = Vec::new();
    let mut truncated = false;
    for item in read.flatten() {
        let name = item.file_name().to_string_lossy().into_owned();
        if HIDDEN.contains(&name.as_str()) {
            continue;
        }
        let path = dir.join(&name);
        let ignored = !kept.contains(&name);
        if ignored && !with_ignored {
            continue;
        }
        // Followed, so a symlink shows as what it points at; one leading
        // out of the project is left out.
        let Ok(real) = dunce::canonicalize(&path) else {
            continue;
        };
        if !real.starts_with(&root) {
            continue;
        }
        let Ok(meta) = std::fs::metadata(&real) else {
            continue;
        };
        if entries.len() == MAX_ENTRIES {
            truncated = true;
            break;
        }
        let folder = meta.is_dir();
        entries.push(ProjectEntry {
            kind: if folder {
                EntryKind::Folder
            } else {
                EntryKind::File
            },
            size_bytes: (!folder).then_some(meta.len()),
            modified_ms: modified_ms(&meta),
            status: if folder {
                None
            } else {
                git.as_ref().and_then(|g| g.status(&path))
            },
            changed: folder && git.as_ref().is_some_and(|g| g.changed_under(&path)),
            ignored,
            secret: !folder && secret(&path),
            name,
        });
    }
    entries.sort_by(|a, b| {
        (a.kind != EntryKind::Folder)
            .cmp(&(b.kind != EntryKind::Folder))
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
            .then_with(|| a.name.cmp(&b.name))
    });
    Ok(ProjectDir {
        project_id: project_id.to_owned(),
        path: rel,
        entries,
        truncated,
        branch: git.and_then(|g| g.branch),
    })
}

/// One file of the project at `folder`.
pub fn read(
    folder: &Path,
    project_id: &str,
    rel: &str,
    secret: &dyn Fn(&Path) -> bool,
) -> Result<ProjectFile, ProtocolError> {
    let root = root_of(folder)?;
    let (target, rel) = resolve(&root, rel)?;
    if rel.is_empty() || target.is_dir() {
        return Err(invalid("That is a folder, not a file"));
    }
    let meta = std::fs::metadata(&target).map_err(|_| not_found("No such file in this project"))?;
    let unresolved = rel.split('/').fold(root.clone(), |p, part| p.join(part));
    let status =
        GitState::of(unresolved.parent().unwrap_or(&root)).and_then(|g| g.status(&unresolved));
    let mut file = ProjectFile {
        project_id: project_id.to_owned(),
        path: rel,
        size_bytes: meta.len(),
        modified_ms: modified_ms(&meta),
        status,
        ..ProjectFile::default()
    };
    // Judged by its name where it sits and where it leads.
    if secret(&unresolved) || secret(&target) {
        file.withheld = Some(Withheld::Secret);
        file.version = String::new();
        return Ok(file);
    }
    let (version, head) =
        hash_and_head(&target).map_err(|_| not_found("This file cannot be read"))?;
    file.version = version;
    match text_of(&head, meta.len() as usize > MAX_TEXT_BYTES) {
        Some(text) => {
            file.truncated = meta.len() as usize > text.len();
            file.text = Some(text);
        }
        None => file.withheld = Some(Withheld::Binary),
    }
    Ok(file)
}

/// The hash of all of `path`'s bytes, and its first [`MAX_TEXT_BYTES`].
fn hash_and_head(path: &Path) -> std::io::Result<(String, Vec<u8>)> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut head = Vec::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        if head.len() < MAX_TEXT_BYTES {
            let take = n.min(MAX_TEXT_BYTES - head.len());
            head.extend_from_slice(&buf[..take]);
        }
        hasher.update(&buf[..n]);
    }
    Ok((format!("sha256:{}", hex::encode(hasher.finalize())), head))
}

/// `head` as text, or `None` when it is not text. A cut file may end
/// inside a character, which is dropped.
fn text_of(head: &[u8], cut: bool) -> Option<String> {
    if head.iter().take(8 * 1024).any(|b| *b == 0) {
        return None;
    }
    match std::str::from_utf8(head) {
        Ok(text) => Some(text.to_owned()),
        Err(e) if cut && e.error_len().is_none() => {
            Some(String::from_utf8_lossy(&head[..e.valid_up_to()]).into_owned())
        }
        Err(_) => None,
    }
}

/// The names in `dir` git does not ignore (all of them outside a
/// repository with no ignore files).
fn not_ignored(dir: &Path) -> HashSet<String> {
    ignore::WalkBuilder::new(dir)
        .max_depth(Some(1))
        .hidden(false)
        .parents(true)
        .ignore(false)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .require_git(false)
        .build()
        .flatten()
        .filter(|e| e.depth() == 1)
        .filter_map(|e| e.file_name().to_str().map(str::to_owned))
        .collect()
}

/// The git repository a folder is in: its branch and its uncommitted
/// changes.
struct GitState {
    top: PathBuf,
    branch: Option<String>,
    /// Changed paths relative to `top`, `/`-separated.
    changes: BTreeMap<String, FileStatus>,
}

impl GitState {
    fn of(dir: &Path) -> Option<GitState> {
        let out = git(
            dir,
            &["rev-parse", "--show-toplevel", "--abbrev-ref", "HEAD"],
        )?;
        let mut lines = out.lines();
        let top = dunce::canonicalize(lines.next()?.trim()).ok()?;
        let branch = lines
            .next()
            .map(str::trim)
            .filter(|b| !b.is_empty() && *b != "HEAD")
            .map(str::to_owned);
        let status = git(
            &top,
            &[
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignore-submodules",
            ],
        )
        .unwrap_or_default();
        Some(GitState {
            top,
            branch,
            changes: parse_status(&status),
        })
    }

    /// `path` relative to the repository, when it is inside it.
    fn rel(&self, path: &Path) -> Option<String> {
        let path = match dunce::canonicalize(path.parent()?) {
            Ok(parent) => parent.join(path.file_name()?),
            Err(_) => path.to_path_buf(),
        };
        let rel = path.strip_prefix(&self.top).ok()?;
        Some(
            rel.components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/"),
        )
    }

    fn status(&self, path: &Path) -> Option<FileStatus> {
        self.changes.get(&self.rel(path)?).copied()
    }

    fn changed_under(&self, folder: &Path) -> bool {
        let Some(rel) = self.rel(folder) else {
            return false;
        };
        let prefix = format!("{rel}/");
        self.changes
            .range(prefix.clone()..)
            .next()
            .is_some_and(|(path, _)| path.starts_with(&prefix))
    }
}

/// `git status --porcelain=v1 -z`'s records: path to how it changed. A
/// rename's record is followed by its old path, skipped.
fn parse_status(out: &str) -> BTreeMap<String, FileStatus> {
    let mut changes = BTreeMap::new();
    let mut records = out.split('\0');
    while let Some(record) = records.next() {
        if record.len() < 4 {
            continue;
        }
        let (code, path) = record.split_at(3);
        let mut code = code.chars();
        let (x, y) = (code.next().unwrap_or(' '), code.next().unwrap_or(' '));
        let status = match (x, y) {
            ('?', _) => FileStatus::Untracked,
            ('R', _) | (_, 'R') => {
                records.next();
                FileStatus::Renamed
            }
            ('A', _) => FileStatus::Added,
            ('D', _) | (_, 'D') => FileStatus::Deleted,
            _ => FileStatus::Modified,
        };
        changes.insert(path.to_owned(), status);
    }
    changes
}

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Process::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "core.quotepath=false"])
        .args(args)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_secrets(_: &Path) -> bool {
        false
    }

    fn env_is_secret(path: &Path) -> bool {
        path.file_name().is_some_and(|n| n == ".env")
    }

    fn project() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("src/forms")).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(root.join("target")).unwrap();
        std::fs::write(root.join("src/contact.rs"), "fn send() {}\n").unwrap();
        std::fs::write(root.join("README.md"), "# Acme\n").unwrap();
        std::fs::write(root.join(".gitignore"), "target/\n").unwrap();
        std::fs::write(root.join(".env"), "TOKEN=synthetic\n").unwrap();
        std::fs::write(root.join("logo.bin"), [0u8, 1, 2, 3]).unwrap();
        dir
    }

    #[test]
    fn lists_folders_first_without_git_or_ignored() {
        let dir = project();
        let listing = list(dir.path(), "p", "", false, &env_is_secret).unwrap();
        let names: Vec<_> = listing.entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            ["src", ".env", ".gitignore", "logo.bin", "README.md"]
        );
        assert!(
            listing
                .entries
                .iter()
                .find(|e| e.name == ".env")
                .unwrap()
                .secret
        );
        let all = list(dir.path(), "p", "", true, &no_secrets).unwrap();
        let target = all.entries.iter().find(|e| e.name == "target").unwrap();
        assert!(target.ignored);
        assert!(!all.entries.iter().any(|e| e.name == ".git"));
    }

    #[test]
    fn stays_inside_the_project() {
        let dir = project();
        for path in [
            "..",
            "../etc",
            "src/../../x",
            "/etc/passwd",
            ".git/config",
            "src\\x",
        ] {
            assert!(read(dir.path(), "p", path, &no_secrets).is_err(), "{path}");
            assert!(
                list(dir.path(), "p", path, false, &no_secrets).is_err(),
                "{path}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_out_of_the_project_is_neither_listed_nor_read() {
        let dir = project();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "out").unwrap();
        std::os::unix::fs::symlink(outside.path().join("secret.txt"), dir.path().join("link"))
            .unwrap();
        let listing = list(dir.path(), "p", "", true, &no_secrets).unwrap();
        assert!(!listing.entries.iter().any(|e| e.name == "link"));
        assert!(read(dir.path(), "p", "link", &no_secrets).is_err());
    }

    #[test]
    fn reads_text_withholds_secrets_and_binaries() {
        let dir = project();
        let file = read(dir.path(), "p", "src/contact.rs", &env_is_secret).unwrap();
        assert_eq!(file.text.as_deref(), Some("fn send() {}\n"));
        assert!(file.version.starts_with("sha256:"));
        let env = read(dir.path(), "p", ".env", &env_is_secret).unwrap();
        assert_eq!((env.text, env.withheld), (None, Some(Withheld::Secret)));
        let bin = read(dir.path(), "p", "logo.bin", &no_secrets).unwrap();
        assert_eq!(bin.withheld, Some(Withheld::Binary));
        assert!(read(dir.path(), "p", "src", &no_secrets).is_err());
    }

    #[test]
    fn a_long_file_comes_cut_with_the_whole_files_version() {
        let dir = project();
        let long = "é".repeat(MAX_TEXT_BYTES);
        std::fs::write(dir.path().join("long.txt"), &long).unwrap();
        let file = read(dir.path(), "p", "long.txt", &no_secrets).unwrap();
        assert!(file.truncated);
        assert_eq!(file.text.unwrap().len(), MAX_TEXT_BYTES);
        let whole = format!("sha256:{}", hex::encode(Sha256::digest(long.as_bytes())));
        assert_eq!(file.version, whole);
    }

    #[test]
    fn status_records_parse() {
        let out = " M src/a.rs\0?? notes.md\0R  new.rs\0old.rs\0D  gone.rs\0";
        let changes = parse_status(out);
        assert_eq!(changes["src/a.rs"], FileStatus::Modified);
        assert_eq!(changes["notes.md"], FileStatus::Untracked);
        assert_eq!(changes["new.rs"], FileStatus::Renamed);
        assert_eq!(changes["gone.rs"], FileStatus::Deleted);
        assert!(!changes.contains_key("old.rs"));
    }
}
