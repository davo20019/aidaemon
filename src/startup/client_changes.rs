//! A session's code changes for the app's code view (`get_changes`): the git
//! repositories its tools worked in, each with its uncommitted changes.
//!
//! Which repositories comes from the session's recorded tool calls: the
//! paths file tools read or wrote, and the directories commands, CLI agents
//! and sub-agents were pointed at. What changed comes from git itself, so
//! edits made by a command or a CLI agent show too. Git runs read-only
//! (`GIT_OPTIONAL_LOCKS=0`): the owner and other sessions may be working in
//! the same tree.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use aidaemon_client_protocol::{FileChange, FileStatus, RepoChanges, SessionChanges};
use serde_json::Value;

/// How many of the session's latest tool calls are read.
const CALLS_READ: i64 = 2000;
/// Repositories shown at most.
const MAX_REPOS: usize = 6;
/// Changed files listed per repository at most.
const MAX_FILES: usize = 500;
/// The patch per repository is cut at about this many bytes.
const MAX_PATCH_BYTES: usize = 512 * 1024;
/// An untracked file larger than this is listed but not shown in the patch.
const MAX_UNTRACKED_BYTES: u64 = 256 * 1024;
/// One git command may take this long.
const GIT_TIMEOUT: Duration = Duration::from_secs(15);

/// The changes of session `session_id`.
pub(crate) async fn session_changes(
    pool: &sqlx::SqlitePool,
    session_id: &str,
) -> anyhow::Result<SessionChanges> {
    let calls: Vec<(Option<String>, String)> = sqlx::query_as(
        "SELECT tool_name, data FROM events \
         WHERE session_id = ? AND event_type = 'tool_call' \
         ORDER BY id DESC LIMIT ?",
    )
    .bind(session_id)
    .bind(CALLS_READ)
    .fetch_all(pool)
    .await?;
    // The project's folder comes first: the conversation works there.
    let folder: Option<String> = sqlx::query_scalar(
        "SELECT p.path FROM session_work_projects s \
         JOIN work_projects p ON p.id = s.project_id \
         WHERE s.session_id = ? AND p.path IS NOT NULL",
    )
    .bind(session_id)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    let mut touched = touched_paths(calls.iter().filter_map(|(tool, data)| {
        let data: Value = serde_json::from_str(data).ok()?;
        let name = tool
            .clone()
            .or_else(|| data.get("name")?.as_str().map(str::to_string))?;
        Some((name, data.get("arguments").cloned().unwrap_or(Value::Null)))
    }));
    if let Some(folder) = folder.map(PathBuf::from)
        && !touched.seen.contains(&folder)
    {
        touched.seen.insert(0, folder);
    }
    changes_for(touched).await
}

/// Paths a session's tool calls name: `written` by its file tools, and
/// `seen` (worked in) by any of them.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Touched {
    pub written: BTreeSet<PathBuf>,
    pub seen: Vec<PathBuf>,
}

/// The paths in `calls` (tool name, arguments), latest first.
pub(crate) fn touched_paths(calls: impl IntoIterator<Item = (String, Value)>) -> Touched {
    let mut touched = Touched::default();
    let mut seen = BTreeSet::new();
    for (tool, args) in calls {
        let mut paths = Vec::new();
        let mut written = false;
        match tool.as_str() {
            "edit_file" | "write_file" => {
                written = true;
                paths.extend(first_str(&args, &["path", "file_path", "file", "filename"]));
            }
            "read_file" | "search_files" => {
                paths.extend(first_str(&args, &["path", "file_path", "file", "filename"]));
            }
            "terminal" | "cli_agent" => paths.extend(first_str(&args, &["working_dir"])),
            "spawn_agent" => {
                paths.extend(strings(args.get("scope")));
                if let Some(tasks) = args.get("tasks").and_then(Value::as_array) {
                    for task in tasks {
                        paths.extend(strings(task.get("scope")));
                    }
                }
            }
            _ => {}
        }
        for path in paths {
            let Some(path) = absolute(&path) else {
                continue;
            };
            if written {
                touched.written.insert(path.clone());
            }
            if seen.insert(path.clone()) {
                touched.seen.push(path);
            }
        }
    }
    touched
}

fn first_str(args: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| args.get(*key)?.as_str())
        .map(str::to_string)
}

fn strings(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// `path` as an absolute path (`~/` expanded); `None` for a relative one,
/// which says nothing about where it was.
fn absolute(path: &str) -> Option<PathBuf> {
    let path = path.trim();
    let path = match path.strip_prefix("~/") {
        Some(rest) => dirs::home_dir()?.join(rest),
        None => PathBuf::from(path),
    };
    path.is_absolute().then_some(path)
}

/// The repositories `touched` names, latest first, with their changes.
pub(crate) async fn changes_for(touched: Touched) -> anyhow::Result<SessionChanges> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for path in &touched.seen {
        if roots.len() >= MAX_REPOS {
            break;
        }
        let Some(dir) = existing_dir(path) else {
            continue;
        };
        if roots.iter().any(|root| dir.starts_with(root)) {
            continue;
        }
        if let Some(root) = repo_root(&dir).await
            && !roots.contains(&root)
        {
            roots.push(root);
        }
    }
    let mut repos = Vec::new();
    for root in roots {
        let written: BTreeSet<String> = touched
            .written
            .iter()
            .filter_map(|path| relative(&root, path))
            .collect();
        if let Some(repo) = repo_changes(&root, &written).await {
            repos.push(repo);
        }
    }
    Ok(SessionChanges { repos })
}

/// The nearest existing directory at or above `path`.
fn existing_dir(path: &Path) -> Option<PathBuf> {
    path.ancestors()
        .find(|dir| dir.is_dir())
        .map(Path::to_path_buf)
}

/// `path` relative to `root`, `/`-separated, when it is inside it.
fn relative(root: &Path, path: &Path) -> Option<String> {
    let canonical = canonical(path);
    let rest = canonical.strip_prefix(root).ok()?;
    let parts: Vec<String> = rest
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect();
    (!parts.is_empty()).then(|| parts.join("/"))
}

/// `path` with its existing part resolved, so it compares with git's root.
fn canonical(path: &Path) -> PathBuf {
    for dir in path.ancestors() {
        // dunce: std's canonicalize, without Windows' `\\?\` prefix.
        if let Ok(real) = dunce::canonicalize(dir) {
            let rest = path.strip_prefix(dir).unwrap_or(Path::new(""));
            return real.join(rest);
        }
    }
    path.to_path_buf()
}

/// The top of the git work tree holding `dir`.
async fn repo_root(dir: &Path) -> Option<PathBuf> {
    let out = git(dir, &["rev-parse", "--show-toplevel"]).await?;
    let top = String::from_utf8_lossy(&out).trim().to_string();
    (!top.is_empty()).then(|| canonical(Path::new(&top)))
}

/// Git's empty tree: what a repository with no commit yet is diffed against.
const EMPTY_TREE: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

async fn repo_changes(root: &Path, written: &BTreeSet<String>) -> Option<RepoChanges> {
    let branch = git(root, &["symbolic-ref", "--short", "-q", "HEAD"])
        .await
        .map(|out| String::from_utf8_lossy(&out).trim().to_string())
        .filter(|branch| !branch.is_empty());
    let base = if git(root, &["rev-parse", "-q", "--verify", "HEAD"])
        .await
        .is_some()
    {
        "HEAD"
    } else {
        EMPTY_TREE
    };
    let status = git(
        root,
        &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )
    .await?;
    let numstat = git(root, &["diff", base, "--numstat", "-z", "-M"])
        .await
        .unwrap_or_default();
    let counts = parse_numstat(&numstat);

    let mut files: Vec<FileChange> = parse_status(&status)
        .into_iter()
        .take(MAX_FILES)
        .map(|(path, status)| {
            let (added, removed) = match status {
                FileStatus::Untracked => (line_count(&root.join(&path)), 0),
                _ => counts.get(&path).copied().unwrap_or((0, 0)),
            };
            FileChange {
                by_session: written.contains(&path),
                path,
                status,
                added,
                removed,
            }
        })
        .collect();
    files.sort_by(|a, b| b.by_session.cmp(&a.by_session).then(a.path.cmp(&b.path)));

    let mut patch = String::from_utf8_lossy(
        &git(root, &["diff", base, "--no-color", "--no-ext-diff", "-M"])
            .await
            .unwrap_or_default(),
    )
    .into_owned();
    for file in files.iter().filter(|f| f.status == FileStatus::Untracked) {
        if patch.len() >= MAX_PATCH_BYTES {
            break;
        }
        patch.push_str(&untracked_patch(&root.join(&file.path), &file.path));
    }
    let truncated = patch.len() > MAX_PATCH_BYTES;
    if truncated {
        let mut cut = MAX_PATCH_BYTES;
        while !patch.is_char_boundary(cut) {
            cut -= 1;
        }
        // End on a whole line.
        cut = patch[..cut].rfind('\n').map_or(cut, |at| at + 1);
        patch.truncate(cut);
    }
    Some(RepoChanges {
        root: root.to_string_lossy().into_owned(),
        branch,
        files,
        patch,
        truncated,
    })
}

/// `git status --porcelain=v1 -z` entries: path and status.
pub(crate) fn parse_status(out: &[u8]) -> Vec<(String, FileStatus)> {
    let mut entries = Vec::new();
    let mut fields = out.split(|b| *b == 0).filter(|f| !f.is_empty());
    while let Some(entry) = fields.next() {
        if entry.len() < 4 {
            continue;
        }
        let (x, y) = (entry[0], entry[1]);
        let path = String::from_utf8_lossy(&entry[3..]).into_owned();
        let status = match (x, y) {
            (b'?', b'?') => FileStatus::Untracked,
            (b'!', b'!') => continue,
            (b'R', _) | (_, b'R') | (b'C', _) => {
                // The old path follows as its own field.
                fields.next();
                FileStatus::Renamed
            }
            (b'D', _) | (_, b'D') => FileStatus::Deleted,
            (b'A', _) => FileStatus::Added,
            _ => FileStatus::Modified,
        };
        entries.push((path, status));
    }
    entries
}

/// `git diff --numstat -z` lines: path to (added, removed). A rename is
/// keyed by its new path; a binary file counts 0 and 0.
pub(crate) fn parse_numstat(out: &[u8]) -> BTreeMap<String, (u32, u32)> {
    let mut counts = BTreeMap::new();
    let mut fields = out.split(|b| *b == 0);
    while let Some(field) = fields.next() {
        if field.is_empty() {
            continue;
        }
        let text = String::from_utf8_lossy(field);
        let mut parts = text.splitn(3, '\t');
        let (Some(added), Some(removed), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let path = if path.is_empty() {
            // A rename: old path, then new path, as their own fields.
            fields.next();
            match fields.next() {
                Some(new) => String::from_utf8_lossy(new).into_owned(),
                None => continue,
            }
        } else {
            path.to_string()
        };
        counts.insert(
            path,
            (added.parse().unwrap_or(0), removed.parse().unwrap_or(0)),
        );
    }
    counts
}

/// Lines in a small text file; 0 for a large, binary or unreadable one.
fn line_count(path: &Path) -> u32 {
    small_text(path).map_or(0, |text| text.lines().count() as u32)
}

fn small_text(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() > MAX_UNTRACKED_BYTES {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    if bytes.contains(&0) {
        return None;
    }
    String::from_utf8(bytes).ok()
}

/// A new file as a unified diff against nothing.
fn untracked_patch(path: &Path, shown: &str) -> String {
    let mut patch =
        format!("diff --git a/{shown} b/{shown}\nnew file\n--- /dev/null\n+++ b/{shown}\n");
    match small_text(path) {
        Some(text) if !text.is_empty() => {
            let lines: Vec<&str> = text.lines().collect();
            patch.push_str(&format!("@@ -0,0 +1,{} @@\n", lines.len()));
            for line in lines {
                patch.push('+');
                patch.push_str(line);
                patch.push('\n');
            }
        }
        Some(_) => {}
        None => patch.push_str("Binary or large file not shown\n"),
    }
    patch
}

/// Run git in `dir`; its stdout when it succeeds.
async fn git(dir: &Path, args: &[&str]) -> Option<Vec<u8>> {
    let mut command = tokio::process::Command::new("git");
    command
        .arg("-C")
        .arg(dir)
        .args(["-c", "core.quotepath=false"])
        .args(args)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true);
    let out = tokio::time::timeout(GIT_TIMEOUT, command.output())
        .await
        .ok()?
        .ok()?;
    out.status.success().then_some(out.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn touched_paths_marks_writes_and_keeps_every_place_worked_in() {
        let touched = touched_paths([
            (
                "edit_file".to_string(),
                json!({"path": "/work/acme/src/lib.rs"}),
            ),
            (
                "write_file".to_string(),
                json!({"file_path": "/work/acme/new.md"}),
            ),
            ("terminal".to_string(), json!({"working_dir": "/work/acme"})),
            (
                "spawn_agent".to_string(),
                json!({"tasks": [{"scope": ["/work/zeta"]}]}),
            ),
            ("read_file".to_string(), json!({"path": "relative.txt"})),
            ("web_fetch".to_string(), json!({"url": "https://x.invalid"})),
        ]);
        assert_eq!(
            touched.written,
            BTreeSet::from([
                PathBuf::from("/work/acme/new.md"),
                PathBuf::from("/work/acme/src/lib.rs")
            ])
        );
        assert_eq!(
            touched.seen,
            vec![
                PathBuf::from("/work/acme/src/lib.rs"),
                PathBuf::from("/work/acme/new.md"),
                PathBuf::from("/work/acme"),
                PathBuf::from("/work/zeta"),
            ]
        );
    }

    #[test]
    fn status_and_numstat_parse_renames_deletes_and_untracked() {
        let status = b" M src/a.rs\0R  new.rs\0old.rs\0D  gone.rs\0A  added.rs\0?? notes.md\0";
        assert_eq!(
            parse_status(status),
            vec![
                ("src/a.rs".to_string(), FileStatus::Modified),
                ("new.rs".to_string(), FileStatus::Renamed),
                ("gone.rs".to_string(), FileStatus::Deleted),
                ("added.rs".to_string(), FileStatus::Added),
                ("notes.md".to_string(), FileStatus::Untracked),
            ]
        );
        let numstat = b"2\t1\tsrc/a.rs\x000\t0\t\x00old.rs\x00new.rs\x00-\t-\timg.png\0";
        let counts = parse_numstat(numstat);
        assert_eq!(counts.get("src/a.rs"), Some(&(2, 1)));
        assert_eq!(counts.get("new.rs"), Some(&(0, 0)));
        assert_eq!(counts.get("img.png"), Some(&(0, 0)));
    }

    fn run(dir: &Path, args: &[&str]) {
        let ok = std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .env("GIT_AUTHOR_NAME", "Alice Rivera")
            .env("GIT_AUTHOR_EMAIL", "alice@example.invalid")
            .env("GIT_COMMITTER_NAME", "Alice Rivera")
            .env("GIT_COMMITTER_EMAIL", "alice@example.invalid")
            .output()
            .expect("git runs")
            .status
            .success();
        assert!(ok, "git {args:?}");
    }

    #[tokio::test]
    async fn a_repository_the_session_wrote_in_shows_its_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("acme-site");
        std::fs::create_dir_all(repo.join("src")).unwrap();
        run(&repo, &["init", "-q", "-b", "main"]);
        std::fs::write(repo.join("src/contact.rs"), "fn send() {}\nfn main() {}\n").unwrap();
        run(&repo, &["add", "."]);
        run(&repo, &["commit", "-q", "-m", "start"]);

        std::fs::write(
            repo.join("src/contact.rs"),
            "fn send() {\n}\nfn main() {}\n",
        )
        .unwrap();
        std::fs::write(repo.join("notes.md"), "a note\n").unwrap();

        let touched = touched_paths([(
            "edit_file".to_string(),
            json!({"path": repo.join("src/contact.rs").to_string_lossy()}),
        )]);
        let changes = changes_for(touched).await.unwrap();
        assert_eq!(changes.repos.len(), 1);
        let found = &changes.repos[0];
        assert_eq!(found.branch.as_deref(), Some("main"));
        assert_eq!(
            found
                .files
                .iter()
                .map(|f| (f.path.as_str(), f.status, f.added, f.removed, f.by_session))
                .collect::<Vec<_>>(),
            vec![
                ("src/contact.rs", FileStatus::Modified, 2, 1, true),
                ("notes.md", FileStatus::Untracked, 1, 0, false),
            ]
        );
        assert!(found.patch.contains("+fn send() {"), "{}", found.patch);
        assert!(found.patch.contains("+++ b/notes.md"), "{}", found.patch);
        assert!(!found.truncated);
    }

    #[tokio::test]
    async fn a_session_outside_any_repository_has_no_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let touched = touched_paths([(
            "write_file".to_string(),
            json!({"path": tmp.path().join("loose.txt").to_string_lossy()}),
        )]);
        assert!(changes_for(touched).await.unwrap().repos.is_empty());
    }
}
