use shell_words;
use std::path::{Component, Path, PathBuf};

pub use crate::types::{PermissionMode, RiskLevel};

/// Explicit credential/private-key path segments form a narrow deterministic
/// floor. Open-ended command danger is assessed semantically elsewhere.
const SENSITIVE_PATH_SEGMENTS: &[&str] = &[
    ".env",
    "id_rsa",
    "id_ed25519",
    "id_dsa",
    "id_ecdsa",
    "shadow",
    "sudoers",
    "master.key",
    "credentials",
    "secrets",
    ".netrc",
    ".pgpass",
    ".npmrc",
    ".pypirc",
];

/// Split a command string by shell operators while respecting quotes.
/// Returns `(segment, operator_after)` pairs.
pub fn split_by_operators(cmd: &str) -> Vec<(String, Option<String>)> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut chars = cmd.chars().peekable();
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut escape_next = false;

    while let Some(ch) = chars.next() {
        if escape_next {
            current.push(ch);
            escape_next = false;
            continue;
        }
        if ch == '\\' && !in_single_quote {
            escape_next = true;
            current.push(ch);
            continue;
        }
        if ch == '\'' && !in_double_quote {
            in_single_quote = !in_single_quote;
            current.push(ch);
            continue;
        }
        if ch == '"' && !in_single_quote {
            in_double_quote = !in_double_quote;
            current.push(ch);
            continue;
        }

        if !in_single_quote && !in_double_quote {
            if ch == '&' && chars.peek() == Some(&'&') {
                chars.next();
                segments.push((current.trim().to_string(), Some("&&".to_string())));
                current.clear();
                continue;
            }
            if ch == '|' && chars.peek() == Some(&'|') {
                chars.next();
                segments.push((current.trim().to_string(), Some("||".to_string())));
                current.clear();
                continue;
            }
            if ch == '|' || ch == ';' {
                segments.push((current.trim().to_string(), Some(ch.to_string())));
                current.clear();
                continue;
            }
        }
        current.push(ch);
    }

    let final_segment = current.trim();
    if !final_segment.is_empty() {
        segments.push((final_segment.to_string(), None));
    }
    segments
}

fn contains_sensitive_path(arg: &str) -> Option<&'static str> {
    let segments = arg.split(&['/', '\\'][..]);
    for segment in segments {
        for sensitive in SENSITIVE_PATH_SEGMENTS {
            if segment == *sensitive {
                return Some(sensitive);
            }
            if segment.starts_with(sensitive)
                && segment.chars().nth(sensitive.len()) == Some('.')
                && !matches!(
                    segment,
                    ".env.example" | ".env.sample" | ".env.template" | ".env.dist"
                )
            {
                return Some(sensitive);
            }
        }
    }
    None
}

/// Return a concrete protected-path reason that semantic assessment cannot
/// downgrade. This recognizes explicit parsed path segments only; it is not a
/// general executable, subcommand, or natural-language risk classifier.
pub fn approval_floor_reason(command: &str) -> Option<String> {
    for (segment, _) in split_by_operators(command) {
        let parts = shell_words::split(&segment).ok()?;
        for arg in parts.iter().skip(1) {
            if let Some(sensitive) = contains_sensitive_path(arg) {
                return Some(format!(
                    "Command explicitly accesses protected credential path segment: {sensitive}"
                ));
            }
        }
    }
    None
}

fn is_recursive_force_delete(parts: &[String]) -> bool {
    let mut recursive = false;
    let mut force = false;
    for arg in parts.iter().skip(1) {
        recursive |= arg == "--recursive";
        force |= arg == "--force";
        if arg.starts_with('-') && !arg.starts_with("--") {
            recursive |= arg.contains('r');
            force |= arg.contains('f');
        }
    }
    recursive && force
}

fn uses_find_delete(parts: &[String]) -> bool {
    parts.iter().skip(1).any(|arg| arg == "-delete")
}

fn find_roots(parts: &[String]) -> Vec<&str> {
    let mut roots = parts
        .iter()
        .skip(1)
        .take_while(|arg| !arg.starts_with('-') && !matches!(arg.as_str(), "(" | ")" | "!" | ","))
        .map(String::as_str)
        .collect::<Vec<_>>();
    if roots.is_empty() {
        roots.push(".");
    }
    roots
}

fn normalize_lexically(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

fn target_is_within_workspace(target: &str, workspace_root: &str) -> bool {
    if workspace_root.trim().is_empty() || target.contains('*') || target.starts_with('~') {
        return false;
    }
    let root = normalize_lexically(Path::new(workspace_root));
    let target_path = Path::new(target);
    let resolved = if target_path.is_absolute() {
        normalize_lexically(target_path)
    } else {
        normalize_lexically(&root.join(target_path))
    };
    resolved == root || resolved.starts_with(&root)
}

fn is_broad_or_sensitive_delete_target(raw_target: &str, workspace_root: &str) -> bool {
    let target = raw_target.trim_matches(|c| c == '"' || c == '\'');
    if target.is_empty() {
        return false;
    }
    if contains_sensitive_path(target).is_some() {
        return true;
    }
    if matches!(target, "/" | "/*" | "~" | "~/" | "$HOME" | "${HOME}")
        || target.starts_with("~/")
        || target.starts_with("$HOME/")
        || target.starts_with("${HOME}/")
    {
        return true;
    }
    if target_is_within_workspace(target, workspace_root) {
        return false;
    }
    if !Path::new(target).is_absolute() {
        return true;
    }

    let broad_prefixes = [
        "/home", "/Users", "/root", "/etc", "/boot", "/sys", "/proc", "/dev", "/usr", "/var",
        "/opt", "/bin", "/sbin", "/lib",
    ];
    broad_prefixes
        .iter()
        .any(|prefix| target == *prefix || target.starts_with(&format!("{prefix}/")))
}

/// Return a hard-block reason for an explicit irreversible broad-path delete.
/// Scoped workspace deletion is left to semantic effect assessment so ordinary
/// cleanup is not mistaken for system destruction.
pub fn hard_block_reason(command: &str, workspace_root: &str) -> Option<String> {
    for (segment, _) in split_by_operators(command) {
        let Ok(parts) = shell_words::split(&segment) else {
            continue;
        };
        let Some(program) = parts.first() else {
            continue;
        };
        let base_cmd = Path::new(program)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(program);

        if base_cmd == "rm" && is_recursive_force_delete(&parts) {
            if let Some(target) = parts
                .iter()
                .skip(1)
                .filter(|arg| !arg.starts_with('-'))
                .find(|arg| is_broad_or_sensitive_delete_target(arg, workspace_root))
            {
                return Some(format!(
                    "Blocked irreversible delete: `rm -rf` targeting broad/sensitive path `{target}`."
                ));
            }
        }

        if base_cmd == "find" && uses_find_delete(&parts) {
            if let Some(root) = find_roots(&parts)
                .into_iter()
                .find(|root| is_broad_or_sensitive_delete_target(root, workspace_root))
            {
                return Some(format!(
                    "Blocked irreversible delete: `find ... -delete` targeting broad/sensitive path `{root}`."
                ));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORKSPACE: &str = "/Users/alice/projects/acme";

    #[test]
    fn operator_split_respects_quotes() {
        assert_eq!(split_by_operators("echo 'a | b' && cargo test").len(), 2);
        assert_eq!(split_by_operators("grep 'x; y' file").len(), 1);
    }

    #[test]
    fn approval_floor_is_limited_to_explicit_sensitive_paths() {
        assert!(approval_floor_reason("cat ~/.ssh/id_ed25519").is_some());
        assert!(approval_floor_reason("source .env.local").is_some());
        assert!(approval_floor_reason("cat .env.example").is_none());
        assert!(approval_floor_reason("echo $(date)").is_none());
        assert!(approval_floor_reason("python3 script.py").is_none());
        assert!(approval_floor_reason("cat password_reset.txt").is_none());
    }

    #[test]
    fn hard_blocks_broad_deletes_but_not_scoped_workspace_cleanup() {
        assert!(hard_block_reason("rm -rf /", WORKSPACE).is_some());
        assert!(hard_block_reason("/bin/rm -fr /Users/alice", WORKSPACE).is_some());
        assert!(hard_block_reason("find /etc -name '*.tmp' -delete", WORKSPACE).is_some());
        assert!(hard_block_reason("rm -rf .env", WORKSPACE).is_some());

        assert!(hard_block_reason("rm -rf ./target", WORKSPACE).is_none());
        assert!(hard_block_reason("rm -rf /Users/alice/projects/acme/target", WORKSPACE).is_none());
        assert!(hard_block_reason("find . -name '*.tmp' -delete", WORKSPACE).is_none());
    }

    #[test]
    fn relative_parent_escape_is_not_treated_as_workspace_scoped() {
        assert!(hard_block_reason("rm -rf ../../..", WORKSPACE).is_some());
    }
}
