//! Git in a project's folder, for the project file browser and the
//! session-changes view: one guarded runner and one `git status` parser.

use std::path::Path;
use std::time::{Duration, Instant};

use aidaemon_client_protocol::FileStatus;

/// How long one git command may run before it is killed.
pub const GIT_TIMEOUT: Duration = Duration::from_secs(15);

/// Run `git args` in `dir` (blocking) and return its stdout on success. The
/// folder's repository is not the daemon's: none of the programs its own
/// configuration names runs (`aidaemon_tools::git_guard`), git never prompts
/// for credentials, and a command still running after [`GIT_TIMEOUT`] is
/// killed.
pub fn git(dir: &Path, args: &[&str]) -> Option<Vec<u8>> {
    let listing =
        run_git(dir, &[], aidaemon_tools::git_guard::DRIVER_LISTING_ARGS).unwrap_or_default();
    let guard = aidaemon_tools::git_guard::untrusted_repo_config_args(&listing).ok()?;
    run_git(dir, &guard, args)
}

fn run_git(dir: &Path, guard: &[String], args: &[&str]) -> Option<Vec<u8>> {
    use std::io::Read as _;
    let mut child = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "core.quotepath=false"])
        .args(guard)
        .args(args)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    // Read on another thread so a large output never fills the pipe while
    // this one waits.
    let mut stdout = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut out = Vec::new();
        stdout.read_to_end(&mut out).map(|_| out)
    });
    let deadline = Instant::now() + GIT_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };
    let out = reader.join().ok()?.ok()?;
    status.success().then_some(out)
}

/// `git status --porcelain=v1 -z` entries: path and status. A rename's or
/// copy's old path (its own field) is skipped; ignored files (`!!`) are
/// left out.
pub fn parse_status(out: &[u8]) -> Vec<(String, FileStatus)> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_copy_s_old_path_is_not_a_file_of_its_own() {
        let out = b" M src/a.rs\0C  copy.rs\0orig.rs\0!! target/\0?? notes.md\0";
        assert_eq!(
            parse_status(out),
            vec![
                ("src/a.rs".to_string(), FileStatus::Modified),
                ("copy.rs".to_string(), FileStatus::Renamed),
                ("notes.md".to_string(), FileStatus::Untracked),
            ]
        );
    }
}
