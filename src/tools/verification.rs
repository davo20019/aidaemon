use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use tokio::sync::RwLock;

use super::command_risk::split_by_operators;

/// Commands that modify files on disk.
const FILE_MODIFYING_COMMANDS: &[&str] = &[
    "rm", "shred", "mv", "cp", "chmod", "chown", "chattr", "dd", "mkfs", "ln", "touch", "mkdir",
];

/// Commands that modify files only when a specific flag is present.
const CONDITIONAL_MODIFYING: &[(&str, &str)] = &[
    ("sed", "-i"),
    ("tee", ""), // tee always writes
];

/// Read-only commands whose path arguments should be recorded as "seen."
const READ_ONLY_COMMANDS: &[&str] = &[
    "ls", "cat", "head", "tail", "less", "more", "file", "stat", "wc", "du", "find", "tree", "fd",
    "grep", "rg", "diff", "bat", "exa", "eza", "readlink", "test",
];

/// A warning returned when a modifying command targets unverified paths.
pub struct VerificationWarning {
    pub unverified_paths: Vec<String>,
    pub message: String,
}

/// Tracks which filesystem paths a session has "seen" (via read-only commands)
/// and gates file-modifying commands that target unverified paths.
pub struct VerificationTracker {
    seen_paths: RwLock<HashMap<String, HashSet<PathBuf>>>,
}

impl VerificationTracker {
    pub fn new() -> Self {
        Self {
            seen_paths: RwLock::new(HashMap::new()),
        }
    }

    /// Record a single path (and its parent directory) as seen for a session.
    pub async fn record_seen_path(&self, session_id: &str, path: &str) {
        let expanded = shellexpand::tilde(path).to_string();
        let pb = PathBuf::from(&expanded);
        let canonical = if pb.is_absolute() {
            pb
        } else {
            // Best-effort: store as-is for relative paths
            pb
        };

        let mut map = self.seen_paths.write().await;
        let set = map.entry(session_id.to_string()).or_default();
        // Record the path itself
        set.insert(canonical.clone());
        // Record parent directory so that `ls /foo` verifies `rm /foo/bar`
        if let Some(parent) = canonical.parent() {
            set.insert(parent.to_path_buf());
        }
    }

    /// Parse a read-only command and record all extracted path arguments.
    pub async fn record_from_command(&self, session_id: &str, command: &str) {
        let segments = split_by_operators(command);
        for (segment, _) in &segments {
            let trimmed = segment.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some((base_cmd, args)) = parse_command_and_args(trimmed) {
                let cmd_name = strip_sudo(&base_cmd);
                if READ_ONLY_COMMANDS.contains(&cmd_name.as_str()) {
                    let paths = extract_path_args(&args);
                    for p in paths {
                        self.record_seen_path(session_id, &p).await;
                    }
                }
                // Also record paths from cd commands (the target dir is now "seen")
                if cmd_name == "cd" {
                    let paths = extract_path_args(&args);
                    for p in paths {
                        self.record_seen_path(session_id, &p).await;
                    }
                }
            }
        }
    }

    /// Check a potentially file-modifying command. Returns a warning if any
    /// target paths have not been previously seen in this session.
    pub async fn check_modifying_command(
        &self,
        session_id: &str,
        command: &str,
    ) -> Option<VerificationWarning> {
        let segments = split_by_operators(command);
        let mut unverified = Vec::new();

        for (segment, _) in &segments {
            let trimmed = segment.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some((base_cmd, args)) = parse_command_and_args(trimmed) {
                let cmd_name = strip_sudo(&base_cmd);

                let is_modifying = FILE_MODIFYING_COMMANDS.contains(&cmd_name.as_str())
                    || CONDITIONAL_MODIFYING.iter().any(|(cmd, flag)| {
                        cmd_name == *cmd && (flag.is_empty() || args.iter().any(|a| a == flag))
                    });

                if !is_modifying {
                    continue;
                }

                let paths = extract_path_args(&args);
                if paths.is_empty() {
                    continue;
                }

                let map = self.seen_paths.read().await;
                let seen = map.get(session_id);

                for p in &paths {
                    let expanded = shellexpand::tilde(p).to_string();
                    let pb = PathBuf::from(&expanded);

                    let is_verified = if let Some(seen_set) = seen {
                        // Check exact path or if any ancestor directory was seen
                        seen_set.contains(&pb) || ancestors_seen(&pb, seen_set)
                    } else {
                        false
                    };

                    if !is_verified {
                        unverified.push(p.clone());
                    }
                }
            }
        }

        if unverified.is_empty() {
            None
        } else {
            Some(VerificationWarning {
                message: format!(
                    "The following paths have not been verified to exist in this session: {}. \
                     Use 'ls' or 'stat' to verify before modifying.",
                    unverified.join(", ")
                ),
                unverified_paths: unverified,
            })
        }
    }

    /// Remove all tracked paths for a session.
    #[allow(dead_code)]
    pub async fn clear_session(&self, session_id: &str) {
        let mut map = self.seen_paths.write().await;
        map.remove(session_id);
    }
}

/// Check if any ancestor of `path` is in the seen set.
fn ancestors_seen(path: &std::path::Path, seen: &HashSet<PathBuf>) -> bool {
    let mut current = path.parent();
    while let Some(ancestor) = current {
        if seen.contains(ancestor) {
            return true;
        }
        current = ancestor.parent();
    }
    false
}

/// Strip `sudo` prefix from a command name.
fn strip_sudo(cmd: &str) -> String {
    if cmd == "sudo" {
        // caller should have already handled sudo in parse_command_and_args
        return cmd.to_string();
    }
    cmd.to_string()
}

/// Parse a single command segment into (command_name, args).
/// Handles sudo by skipping it and any flags (like -u user).
fn parse_command_and_args(segment: &str) -> Option<(String, Vec<String>)> {
    let tokens = match shell_words::split(segment) {
        Ok(t) => t,
        Err(_) => return None,
    };
    if tokens.is_empty() {
        return None;
    }

    let mut idx = 0;

    // Skip sudo and its flags
    if tokens[idx] == "sudo" {
        idx += 1;
        // Skip sudo flags like -u, -E, etc
        while idx < tokens.len() {
            if tokens[idx].starts_with('-') {
                idx += 1;
                // If the flag takes a value (e.g., -u root), skip that too
                if idx < tokens.len() && !tokens[idx].starts_with('-') {
                    // Check if previous flag was -u, -g, -C etc that take arguments
                    let prev = &tokens[idx - 1];
                    if prev == "-u" || prev == "-g" || prev == "-C" {
                        idx += 1;
                    }
                }
            } else {
                break;
            }
        }
    }

    if idx >= tokens.len() {
        return None;
    }

    let cmd = tokens[idx].clone();
    let args = tokens[idx + 1..].to_vec();
    Some((cmd, args))
}

/// Extract path-like arguments from a token list, filtering out flags.
fn extract_path_args(args: &[String]) -> Vec<String> {
    let mut paths = Vec::new();
    let mut skip_next = false;

    for (i, arg) in args.iter().enumerate() {
        if skip_next {
            skip_next = false;
            continue;
        }

        // Skip flags
        if arg.starts_with('-') {
            // Some flags take a value argument, skip the next token too.
            // Common examples: -o output, -f file, --output file
            if i + 1 < args.len() {
                let next = &args[i + 1];
                // Heuristic: if the flag is short (-X) and next arg doesn't start with -,
                // it might be a flag value. Skip it to be safe for common cases.
                if (arg.len() == 2 || arg.starts_with("--")) && !next.starts_with('-') {
                    // Only skip for known value-taking flags
                    let value_flags = [
                        "-o",
                        "-f",
                        "-t",
                        "-m",
                        "-T",
                        "--target-directory",
                        "--output",
                        "--suffix",
                        "--backup",
                    ];
                    if value_flags.contains(&arg.as_str()) {
                        skip_next = true;
                    }
                }
            }
            continue;
        }

        // Skip things that look like shell variables or globs with braces
        if arg.starts_with('$') || arg.contains("$(") {
            continue;
        }

        // Looks like a path argument
        paths.push(arg.clone());
    }

    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_record_and_verify_path() {
        let tracker = VerificationTracker::new();
        let sid = "test-session";

        // Not yet seen — should warn
        let warning = tracker
            .check_modifying_command(sid, "rm /tmp/foo.txt")
            .await;
        assert!(warning.is_some());
        assert!(warning
            .unwrap()
            .unverified_paths
            .contains(&"/tmp/foo.txt".to_string()));

        // Record the path
        tracker.record_seen_path(sid, "/tmp/foo.txt").await;

        // Now verified — should pass
        let warning = tracker
            .check_modifying_command(sid, "rm /tmp/foo.txt")
            .await;
        assert!(warning.is_none());
    }

    #[tokio::test]
    async fn test_parent_directory_verification() {
        let tracker = VerificationTracker::new();
        let sid = "test-session";

        // ls /foo records /foo as seen parent
        tracker.record_from_command(sid, "ls /foo").await;

        // rm /foo/bar should pass because /foo was seen
        let warning = tracker.check_modifying_command(sid, "rm /foo/bar").await;
        assert!(warning.is_none());
    }

    #[tokio::test]
    async fn test_unverified_path_returns_warning() {
        let tracker = VerificationTracker::new();
        let sid = "test-session";

        let warning = tracker
            .check_modifying_command(sid, "rm -rf /important/data")
            .await;
        assert!(warning.is_some());
        let w = warning.unwrap();
        assert!(w.unverified_paths.contains(&"/important/data".to_string()));
        assert!(w.message.contains("not been verified"));
    }

    #[tokio::test]
    async fn test_compound_commands() {
        let tracker = VerificationTracker::new();
        let sid = "test-session";

        // Record via compound read command
        tracker.record_from_command(sid, "cd /foo && ls /bar").await;

        // Both /foo and /bar should be seen
        let warning = tracker
            .check_modifying_command(sid, "rm /foo/file.txt")
            .await;
        assert!(warning.is_none());

        let warning = tracker
            .check_modifying_command(sid, "rm /bar/file.txt")
            .await;
        assert!(warning.is_none());
    }

    #[tokio::test]
    async fn test_flag_filtering() {
        let tracker = VerificationTracker::new();
        let sid = "test-session";

        // rm -rf /dir should extract /dir, not -rf
        let warning = tracker.check_modifying_command(sid, "rm -rf /dir").await;
        assert!(warning.is_some());
        let w = warning.unwrap();
        assert_eq!(w.unverified_paths, vec!["/dir".to_string()]);
    }

    #[tokio::test]
    async fn test_session_isolation() {
        let tracker = VerificationTracker::new();

        tracker.record_seen_path("session-a", "/tmp/file").await;

        // session-b should not see session-a's paths
        let warning = tracker
            .check_modifying_command("session-b", "rm /tmp/file")
            .await;
        assert!(warning.is_some());

        // session-a should see it
        let warning = tracker
            .check_modifying_command("session-a", "rm /tmp/file")
            .await;
        assert!(warning.is_none());
    }

    #[tokio::test]
    async fn test_sudo_handling() {
        let tracker = VerificationTracker::new();
        let sid = "test-session";

        // sudo rm /etc/foo should extract /etc/foo
        let warning = tracker
            .check_modifying_command(sid, "sudo rm /etc/foo")
            .await;
        assert!(warning.is_some());
        let w = warning.unwrap();
        assert!(w.unverified_paths.contains(&"/etc/foo".to_string()));
    }

    #[tokio::test]
    async fn test_tilde_expansion() {
        let tracker = VerificationTracker::new();
        let sid = "test-session";

        // Record home dir file
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/user".to_string());
        let expanded_path = format!("{}/file.txt", home);
        tracker.record_seen_path(sid, &expanded_path).await;

        // rm ~/file.txt should be verified (tilde expands to same path)
        let warning = tracker.check_modifying_command(sid, "rm ~/file.txt").await;
        assert!(warning.is_none());
    }

    #[tokio::test]
    async fn test_read_only_commands_not_flagged() {
        let tracker = VerificationTracker::new();
        let sid = "test-session";

        // Read-only commands should never return a warning
        let warning = tracker
            .check_modifying_command(sid, "cat /etc/passwd")
            .await;
        assert!(warning.is_none());

        let warning = tracker.check_modifying_command(sid, "ls /var/log").await;
        assert!(warning.is_none());
    }

    #[tokio::test]
    async fn test_sed_inplace_flagged() {
        let tracker = VerificationTracker::new();
        let sid = "test-session";

        // sed -i is file-modifying
        let warning = tracker
            .check_modifying_command(sid, "sed -i 's/foo/bar/' /tmp/config.txt")
            .await;
        assert!(warning.is_some());

        // sed without -i is not modifying (just outputs to stdout)
        let warning = tracker
            .check_modifying_command(sid, "sed 's/foo/bar/' /tmp/config.txt")
            .await;
        assert!(warning.is_none());
    }

    #[tokio::test]
    async fn test_tee_flagged() {
        let tracker = VerificationTracker::new();
        let sid = "test-session";

        // tee always writes
        let warning = tracker
            .check_modifying_command(sid, "tee /tmp/output.txt")
            .await;
        assert!(warning.is_some());
    }

    #[tokio::test]
    async fn test_clear_session() {
        let tracker = VerificationTracker::new();
        let sid = "test-session";

        tracker.record_seen_path(sid, "/tmp/file.txt").await;
        let warning = tracker
            .check_modifying_command(sid, "rm /tmp/file.txt")
            .await;
        assert!(warning.is_none());

        tracker.clear_session(sid).await;

        let warning = tracker
            .check_modifying_command(sid, "rm /tmp/file.txt")
            .await;
        assert!(warning.is_some());
    }

    #[tokio::test]
    async fn test_record_from_command_ls() {
        let tracker = VerificationTracker::new();
        let sid = "test-session";

        tracker.record_from_command(sid, "ls /var/log").await;

        // Files under /var/log should be verified
        let warning = tracker
            .check_modifying_command(sid, "rm /var/log/syslog")
            .await;
        assert!(warning.is_none());
    }

    #[tokio::test]
    async fn test_mv_flagged() {
        let tracker = VerificationTracker::new();
        let sid = "test-session";

        let warning = tracker
            .check_modifying_command(sid, "mv /tmp/a /tmp/b")
            .await;
        assert!(warning.is_some());
        let w = warning.unwrap();
        // Both source and destination should be unverified
        assert!(w.unverified_paths.contains(&"/tmp/a".to_string()));
        assert!(w.unverified_paths.contains(&"/tmp/b".to_string()));
    }

    #[test]
    fn test_parse_command_and_args_basic() {
        let (cmd, args) = parse_command_and_args("rm -rf /foo").unwrap();
        assert_eq!(cmd, "rm");
        assert_eq!(args, vec!["-rf", "/foo"]);
    }

    #[test]
    fn test_parse_command_and_args_sudo() {
        let (cmd, args) = parse_command_and_args("sudo rm -f /etc/file").unwrap();
        assert_eq!(cmd, "rm");
        assert_eq!(args, vec!["-f", "/etc/file"]);
    }

    #[test]
    fn test_extract_path_args_filters_flags() {
        let args: Vec<String> = vec!["-rf".into(), "/dir".into(), "-v".into()];
        let paths = extract_path_args(&args);
        assert_eq!(paths, vec!["/dir".to_string()]);
    }

    #[test]
    fn test_extract_path_args_skips_shell_vars() {
        let args: Vec<String> = vec!["$HOME/file".into(), "/real/path".into()];
        let paths = extract_path_args(&args);
        assert_eq!(paths, vec!["/real/path".to_string()]);
    }
}
