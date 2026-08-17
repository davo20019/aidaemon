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
            if ch == '|' && chars.peek() == Some(&'&') {
                chars.next();
                segments.push((current.trim().to_string(), Some("|&".to_string())));
                current.clear();
                continue;
            }
            if matches!(ch, '|' | ';' | '\n') {
                segments.push((current.trim().to_string(), Some(ch.to_string())));
                current.clear();
                continue;
            }
            if ch == '&'
                && !current.ends_with('>')
                && !current.ends_with('<')
                && chars.peek() != Some(&'>')
            {
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

/// One executable invocation recovered from shell syntax. `program` preserves
/// the executable token (including an absolute path); `arguments` are parsed
/// argv values. This is a syntax tree projection, not a natural-language
/// classifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShellInvocation {
    pub program: String,
    pub arguments: Vec<String>,
}

const MAX_SHELL_INVOCATION_DEPTH: usize = 8;

fn heredoc_specs(line: &str) -> Vec<(String, bool)> {
    let bytes = line.as_bytes();
    let mut specs = Vec::new();
    let mut index = 0;
    let mut quote = None;
    let mut escaped = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if escaped {
            escaped = false;
            index += 1;
            continue;
        }
        if byte == b'\\' && quote != Some(b'\'') {
            escaped = true;
            index += 1;
            continue;
        }
        if matches!(byte, b'\'' | b'"') {
            if quote == Some(byte) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(byte);
            }
            index += 1;
            continue;
        }
        if quote.is_some()
            || byte != b'<'
            || bytes.get(index + 1) != Some(&b'<')
            || bytes.get(index + 2) == Some(&b'<')
        {
            index += 1;
            continue;
        }

        index += 2;
        let strip_tabs = bytes.get(index) == Some(&b'-');
        if strip_tabs {
            index += 1;
        }
        while bytes.get(index).is_some_and(u8::is_ascii_whitespace) {
            index += 1;
        }
        let mut delimiter = Vec::new();
        let mut delimiter_quote = None;
        while let Some(&current) = bytes.get(index) {
            if delimiter_quote.is_none()
                && (current.is_ascii_whitespace() || matches!(current, b';' | b'|' | b'&' | b'<'))
            {
                break;
            }
            if current == b'\\' && delimiter_quote != Some(b'\'') {
                if let Some(&literal) = bytes.get(index + 1) {
                    delimiter.push(literal);
                    index += 2;
                    continue;
                }
            }
            if matches!(current, b'\'' | b'"') {
                if delimiter_quote == Some(current) {
                    delimiter_quote = None;
                    index += 1;
                    continue;
                }
                if delimiter_quote.is_none() {
                    delimiter_quote = Some(current);
                    index += 1;
                    continue;
                }
            }
            delimiter.push(current);
            index += 1;
        }
        if !delimiter.is_empty() {
            specs.push((String::from_utf8_lossy(&delimiter).into_owned(), strip_tabs));
        }
    }
    specs
}

/// Remove here-document payloads before projecting executable invocations.
/// Payload lines are data for the preceding redirection, not shell programs.
/// Keeping them in the projection fabricates executable and path identities
/// from embedded Python, JSON, or configuration content.
fn command_without_heredoc_bodies(command: &str) -> String {
    use std::collections::VecDeque;

    let mut pending = VecDeque::new();
    let mut projected = String::with_capacity(command.len());
    for line in command.split_inclusive('\n') {
        let content = line.trim_end_matches(['\r', '\n']);
        if let Some((delimiter, strip_tabs)) = pending.front() {
            let candidate = if *strip_tabs {
                content.trim_start_matches('\t')
            } else {
                content
            };
            if candidate == delimiter {
                pending.pop_front();
            }
            if line.ends_with('\n') {
                projected.push('\n');
            }
            continue;
        }

        projected.push_str(line);
        pending.extend(heredoc_specs(content));
    }
    projected
}

fn is_environment_assignment(token: &str) -> bool {
    let Some((name, _)) = token.split_once('=') else {
        return false;
    };
    !name.is_empty()
        && name.chars().enumerate().all(|(index, ch)| {
            ch == '_' || ch.is_ascii_alphanumeric() && (index > 0 || !ch.is_ascii_digit())
        })
}

fn executable_basename(program: &str) -> &str {
    Path::new(program)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(program)
}

fn command_start(tokens: &[String]) -> Option<usize> {
    let mut index = 0;
    while tokens
        .get(index)
        .is_some_and(|token| is_environment_assignment(token))
    {
        index += 1;
    }
    while tokens
        .get(index)
        .is_some_and(|token| matches!(executable_basename(token), "command" | "builtin"))
    {
        index += 1;
    }
    if tokens
        .get(index)
        .is_some_and(|token| executable_basename(token) == "env")
    {
        index += 1;
        while let Some(token) = tokens.get(index) {
            if token == "--" {
                index += 1;
                break;
            }
            if is_environment_assignment(token) {
                index += 1;
                continue;
            }
            if matches!(
                token.as_str(),
                "-u" | "--unset" | "-C" | "--chdir" | "-S" | "--split-string"
            ) {
                index = index.saturating_add(2);
                continue;
            }
            if token.starts_with('-') {
                index += 1;
                continue;
            }
            break;
        }
    }
    while tokens
        .get(index)
        .is_some_and(|token| matches!(executable_basename(token), "command" | "builtin"))
    {
        index += 1;
    }
    (index < tokens.len()).then_some(index)
}

fn shell_script_operand(arguments: &[String]) -> Option<&str> {
    arguments.iter().enumerate().find_map(|(index, argument)| {
        let command_flag = argument == "--command"
            || argument
                .strip_prefix('-')
                .filter(|flags| !flags.starts_with('-'))
                .is_some_and(|flags| flags.contains('c'));
        command_flag
            .then(|| arguments.get(index + 1).map(String::as_str))
            .flatten()
    })
}

fn collect_shell_invocations(command: &str, depth: usize, output: &mut Vec<ShellInvocation>) {
    if depth > MAX_SHELL_INVOCATION_DEPTH {
        return;
    }
    for (segment, _) in split_by_operators(&command_without_heredoc_bodies(command)) {
        if segment.is_empty() {
            continue;
        }
        let Ok(tokens) = shell_words::split(&segment) else {
            continue;
        };
        let Some(program_index) = command_start(&tokens) else {
            continue;
        };
        let invocation = ShellInvocation {
            program: tokens[program_index].clone(),
            arguments: tokens[program_index + 1..].to_vec(),
        };
        let program_name = executable_basename(&invocation.program);
        let nested_script = if matches!(program_name, "sh" | "bash" | "dash" | "ksh" | "zsh") {
            shell_script_operand(&invocation.arguments).map(str::to_string)
        } else if program_name == "trap" {
            invocation
                .arguments
                .first()
                .filter(|body| !body.starts_with('-'))
                .cloned()
        } else {
            None
        };
        if !output.contains(&invocation) {
            output.push(invocation);
        }
        if let Some(script) = nested_script {
            collect_shell_invocations(&script, depth + 1, output);
        }
    }
}

/// Parse top-level and nested `shell -c` executable invocations through one
/// shared shell-grammar boundary. Runtime dependency discovery and workload
/// classification consume this same projection so wrapper depth cannot make
/// their views drift.
pub(crate) fn structural_shell_invocations(command: &str) -> Vec<ShellInvocation> {
    let mut invocations = Vec::new();
    collect_shell_invocations(command, 0, &mut invocations);
    invocations
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
    fn nested_shell_invocations_are_parsed_from_combined_command_flags() {
        let invocations = structural_shell_invocations(
            "/bin/sh -lc 'trap '\"'\"'rm -rf -- \"$root\"'\"'\"' EXIT; /opt/homebrew/bin/python3 -m venv \"$root/.venv\"'",
        );
        assert!(invocations.iter().any(|invocation| {
            invocation.program == "/bin/sh"
                && invocation.arguments.first().is_some_and(|arg| arg == "-lc")
        }));
        assert!(invocations.iter().any(|invocation| {
            invocation.program == "/opt/homebrew/bin/python3"
                && invocation
                    .arguments
                    .starts_with(&["-m".to_string(), "venv".to_string()])
        }));
    }

    #[test]
    fn environment_wrappers_preserve_the_executed_program_identity() {
        let invocations = structural_shell_invocations(
            "MODE=test /usr/bin/env -u TOKEN PATH=/usr/bin command /usr/bin/false",
        );
        assert_eq!(
            invocations,
            [ShellInvocation {
                program: "/usr/bin/false".to_string(),
                arguments: Vec::new(),
            }]
        );
    }

    #[test]
    fn projection_preserves_redirections_and_pipe_stderr_operators() {
        assert_eq!(
            split_by_operators("first 2>&1 & second &>output |& third"),
            [
                ("first 2>&1".to_string(), Some("&".to_string())),
                ("second &>output".to_string(), Some("|&".to_string())),
                ("third".to_string(), None),
            ]
        );
    }

    #[test]
    fn heredoc_payload_is_not_projected_as_executable_source() {
        let invocations = structural_shell_invocations(
            "python3 <<'PY'\ntarget = Path(root) / 'fabricated'\narchive.write(target)\nPY\n/usr/bin/false",
        );
        assert_eq!(
            invocations,
            [
                ShellInvocation {
                    program: "python3".to_string(),
                    arguments: vec!["<<PY".to_string()],
                },
                ShellInvocation {
                    program: "/usr/bin/false".to_string(),
                    arguments: Vec::new(),
                },
            ]
        );
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
