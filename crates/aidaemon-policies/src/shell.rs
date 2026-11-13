//! A small, conservative shell tokenizer used by command policies.
//!
//! It is not a shell. It splits a command line into simple-command segments
//! at control operators (`;`, `&&`, `||`, `|`, `&`, newlines, subshell and
//! command-substitution boundaries), honours quotes and backslashes, and
//! unwraps common launchers (`sudo`, `env`, `nohup`, `bash -c "..."`) so a
//! policy sees the program that actually runs and its real arguments.

use std::path::{Component, Path, PathBuf};

/// Split `command` into segments of tokens, with launchers unwrapped.
/// `bash -c "a; b"` contributes the segments of `a; b`.
pub fn command_segments(command: &str) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    collect_segments(command, 0, &mut out);
    out
}

fn collect_segments(command: &str, depth: usize, out: &mut Vec<Vec<String>>) {
    for segment in split_raw(command) {
        let stripped = strip_launchers(&segment);
        if stripped.is_empty() {
            continue;
        }
        if depth < 4 && basename(&stripped[0]) == "eval" {
            collect_segments(&stripped[1..].join(" "), depth + 1, out);
            continue;
        }
        if depth < 4 && is_shell(&stripped[0])
            && let Some(pos) = stripped.iter().position(|t| runs_a_string(t))
                && let Some(inner) = stripped.get(pos + 1) {
                    collect_segments(inner, depth + 1, out);
                    continue;
                }
        out.push(stripped);
    }
}

/// A shell option that runs its argument: `-c`, or a cluster such as `-lc`.
fn runs_a_string(token: &str) -> bool {
    token.starts_with('-') && !token.starts_with("--") && token.contains('c')
}

fn is_shell(program: &str) -> bool {
    matches!(
        basename(program),
        "sh" | "bash" | "zsh" | "dash" | "ksh" | "fish"
    )
}

/// The final path component of a program token (`/bin/rm` -> `rm`).
pub(crate) fn basename(program: &str) -> &str {
    program.rsplit('/').next().unwrap_or(program)
}

/// Tokenize and split at control operators, without unwrapping.
fn split_raw(command: &str) -> Vec<Vec<String>> {
    let mut segments: Vec<Vec<String>> = Vec::new();
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_token = false;
    let mut chars = command.chars().peekable();

    fn end_token(current: &mut String, in_token: &mut bool, tokens: &mut Vec<String>) {
        if *in_token {
            tokens.push(std::mem::take(current));
            *in_token = false;
        }
    }
    fn end_segment(tokens: &mut Vec<String>, segments: &mut Vec<Vec<String>>) {
        if !tokens.is_empty() {
            segments.push(std::mem::take(tokens));
        }
    }

    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                in_token = true;
                for q in chars.by_ref() {
                    if q == '\'' {
                        break;
                    }
                    current.push(q);
                }
            }
            '"' => {
                in_token = true;
                while let Some(q) = chars.next() {
                    match q {
                        '"' => break,
                        '\\' => {
                            if let Some(next) = chars.next() {
                                current.push(next);
                            }
                        }
                        _ => current.push(q),
                    }
                }
            }
            '\\' => {
                in_token = true;
                if let Some(next) = chars.next()
                    && next != '\n' {
                        current.push(next);
                    }
            }
            '$' if chars.peek() == Some(&'(') => {
                chars.next();
                end_token(&mut current, &mut in_token, &mut tokens);
                end_segment(&mut tokens, &mut segments);
            }
            ';' | '&' | '|' | '\n' | '(' | ')' | '`' => {
                end_token(&mut current, &mut in_token, &mut tokens);
                end_segment(&mut tokens, &mut segments);
            }
            c if c.is_whitespace() => end_token(&mut current, &mut in_token, &mut tokens),
            c => {
                in_token = true;
                current.push(c);
            }
        }
    }
    end_token(&mut current, &mut in_token, &mut tokens);
    end_segment(&mut tokens, &mut segments);
    segments
}

/// Drop leading environment assignments and launcher programs.
fn strip_launchers(tokens: &[String]) -> Vec<String> {
    let mut i = 0;
    while i < tokens.len() {
        let token = tokens[i].as_str();
        if is_assignment(token) {
            i += 1;
            continue;
        }
        match basename(token) {
            "sudo" | "doas" => {
                i = skip_options(
                    tokens,
                    i + 1,
                    &[
                        "-u",
                        "-g",
                        "-C",
                        "-h",
                        "-p",
                        "-U",
                        "-r",
                        "-t",
                        "--user",
                        "--group",
                        "--close-from",
                        "--host",
                        "--prompt",
                        "--other-user",
                        "--role",
                        "--type",
                        "--chdir",
                        "--chroot",
                    ],
                );
            }
            "env" => {
                i = skip_options(
                    tokens,
                    i + 1,
                    &["-u", "--unset", "-C", "--chdir", "-S", "--split-string"],
                );
                while i < tokens.len() && is_assignment(&tokens[i]) {
                    i += 1;
                }
            }
            "timeout" => {
                i = skip_options(tokens, i + 1, &["-s", "--signal", "-k", "--kill-after"]);
                // The duration.
                i += 1;
            }
            "nice" | "ionice" => {
                i = skip_options(
                    tokens,
                    i + 1,
                    &[
                        "-n",
                        "--adjustment",
                        "-c",
                        "--class",
                        "--classdata",
                        "-p",
                        "-P",
                        "-u",
                    ],
                );
            }
            "stdbuf" => i = skip_options(tokens, i + 1, &["-i", "-o", "-e"]),
            "nohup" | "command" | "exec" | "time" | "builtin" | "setsid" | "unbuffer" => i += 1,
            _ => break,
        }
    }
    tokens[i.min(tokens.len())..].to_vec()
}

/// The index of the first token after a launcher's options, stepping over
/// the values of `with_value` options.
fn skip_options(tokens: &[String], mut i: usize, with_value: &[&str]) -> usize {
    while i < tokens.len() && tokens[i].starts_with('-') {
        i += if with_value.contains(&tokens[i].as_str()) {
            2
        } else {
            1
        };
    }
    i
}

fn is_assignment(token: &str) -> bool {
    match token.split_once('=') {
        Some((name, _)) => {
            !name.is_empty()
                && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && !name.starts_with(|c: char| c.is_ascii_digit())
        }
        None => false,
    }
}

const SYSTEM_DIRS: &[&str] = &[
    "/bin",
    "/boot",
    "/dev",
    "/etc",
    "/home",
    "/lib",
    "/lib64",
    "/opt",
    "/proc",
    "/root",
    "/sbin",
    "/sys",
    "/usr",
    "/var",
    "/System",
    "/Users",
    "/Applications",
    "/Library",
];

/// Normalize a path argument and decide whether wiping it recursively would
/// be catastrophic: the filesystem root, a home directory, or a top-level
/// system directory (with or without a trailing `/` or `/*`).
fn is_catastrophic_target(raw: &str) -> bool {
    let mut t = raw.trim().to_string();
    while t.contains("//") {
        t = t.replace("//", "/");
    }
    loop {
        let before = t.len();
        for suffix in ["/*", "/.", "/"] {
            if let Some(stripped) = t.strip_suffix(suffix) {
                t = stripped.to_string();
            }
        }
        if t == "*" {
            t.clear();
        }
        if t.len() == before {
            break;
        }
    }
    if raw.trim().starts_with('/') && t.is_empty() {
        return true; // "/", "/*", "//"
    }
    matches!(t.as_str(), "~" | "$HOME" | "${HOME}")
        || (t.starts_with('~') && !t.contains('/'))
        || SYSTEM_DIRS.contains(&t.as_str())
}

/// Why a single command line is catastrophic, if it is. Checks every
/// segment structurally: `rm` with a recursive flag on the root, a home
/// directory or a system directory, `--no-preserve-root`, `mkfs*`, raw
/// device writes with `dd`/`shred`/`wipefs`, and recursive
/// `chmod`/`chown`/`chgrp` on those same targets.
pub fn catastrophic_reason(command: &str) -> Option<String> {
    command_segments(command)
        .iter()
        .find_map(|segment| segment_reason(segment))
}

fn segment_reason(tokens: &[String]) -> Option<String> {
    let program = basename(tokens.first()?);
    let args = &tokens[1..];
    let (flags, operands) = split_flags(args);
    let recursive = flags.iter().any(|f| {
        f == "--recursive" || (f.starts_with('-') && !f.starts_with("--") && f.contains(['r', 'R']))
    });
    match program {
        "rm" => {
            if flags.iter().any(|f| f == "--no-preserve-root") {
                return Some("`rm --no-preserve-root` is never allowed".into());
            }
            if recursive
                && let Some(target) = operands.iter().find(|o| is_catastrophic_target(o)) {
                    return Some(format!("recursive `rm` of `{target}` is never allowed"));
                }
            None
        }
        "chmod" | "chown" | "chgrp" if recursive => operands
            .iter()
            .find(|o| is_catastrophic_target(o))
            .map(|target| format!("recursive `{program}` of `{target}` is never allowed")),
        p if p == "mkfs" || p.starts_with("mkfs.") => {
            Some(format!("`{p}` formats a filesystem and is never allowed"))
        }
        "dd" => args
            .iter()
            .filter_map(|a| a.strip_prefix("of="))
            .find(|dev| is_block_device(dev))
            .map(|dev| format!("`dd` writing to raw device `{dev}` is never allowed")),
        "shred" | "wipefs" => operands
            .iter()
            .find(|o| is_block_device(o))
            .map(|dev| format!("`{program}` on raw device `{dev}` is never allowed")),
        _ => None,
    }
}

/// Path segments that hold credentials.
const CREDENTIAL_PATH_SEGMENTS: &[&str] = &[
    ".ssh",
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

/// The credential segment `arg` names as a path, if any. `.env.example`
/// style templates hold placeholders and do not count.
pub fn credential_path_segment(arg: &str) -> Option<&'static str> {
    for segment in arg.split(&['/', '\\'][..]) {
        for sensitive in CREDENTIAL_PATH_SEGMENTS {
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

/// Why `command` is an irreversible delete of something outside
/// `workspace_root`: a recursive `rm`, or a `find` that deletes, aimed at a
/// credential path, the home directory, a system directory or a relative
/// path leaving the workspace. Launchers and subshells are unwrapped.
/// Scoped cleanup inside the workspace is not reported.
pub fn irreversible_delete_reason(command: &str, workspace_root: &str) -> Option<String> {
    command_segments(command)
        .iter()
        .find_map(|segment| segment_delete_reason(segment, workspace_root))
}

fn segment_delete_reason(tokens: &[String], workspace_root: &str) -> Option<String> {
    // `xargs` runs the command it is given; its own options come first.
    let tokens = match tokens.first().map(|program| basename(program)) {
        Some("xargs") => {
            let start = tokens
                .iter()
                .position(|token| matches!(basename(token), "rm" | "find"))?;
            &tokens[start..]
        }
        _ => tokens,
    };
    let broad = |target: &str| is_broad_delete_target(target, workspace_root);
    let reason = |form: &str, target: &str| {
        format!(
            "irreversible delete: `{form}` targeting broad/sensitive path `{target}`. \
             Use scoped, non-destructive commands instead."
        )
    };
    match basename(tokens.first()?) {
        "rm" => {
            let (flags, operands) = split_flags(&tokens[1..]);
            let recursive = flags.iter().any(|f| {
                f == "--recursive"
                    || (f.starts_with('-') && !f.starts_with("--") && f.contains(['r', 'R']))
            });
            if !recursive {
                return None;
            }
            operands
                .iter()
                .find(|operand| broad(operand))
                .map(|target| reason("rm -r", target))
        }
        "find" if find_deletes(tokens) => find_roots(tokens)
            .into_iter()
            .find(|root| broad(root))
            .map(|root| reason("find ... -delete", root)),
        _ => None,
    }
}

fn find_deletes(tokens: &[String]) -> bool {
    tokens.iter().enumerate().skip(1).any(|(i, token)| {
        token == "-delete"
            || (matches!(token.as_str(), "-exec" | "-execdir" | "-ok" | "-okdir")
                && tokens
                    .get(i + 1)
                    .is_some_and(|program| basename(program) == "rm"))
    })
}

fn find_roots(tokens: &[String]) -> Vec<&str> {
    let mut roots: Vec<&str> = tokens
        .iter()
        .skip(1)
        .take_while(|arg| !arg.starts_with('-') && !matches!(arg.as_str(), "(" | ")" | "!" | ","))
        .map(String::as_str)
        .collect();
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

fn is_within_workspace(target: &str, workspace_root: &str) -> bool {
    if workspace_root.trim().is_empty() || target.contains('*') || target.starts_with('~') {
        return false;
    }
    let root = normalize_lexically(Path::new(workspace_root));
    let path = Path::new(target);
    let resolved = if path.is_absolute() {
        normalize_lexically(path)
    } else {
        normalize_lexically(&root.join(path))
    };
    resolved == root || resolved.starts_with(&root)
}

fn is_broad_delete_target(raw_target: &str, workspace_root: &str) -> bool {
    let target = raw_target.trim_matches(|c| c == '"' || c == '\'');
    if target.is_empty() {
        return false;
    }
    if credential_path_segment(target).is_some() {
        return true;
    }
    if matches!(target, "/" | "/*" | "~" | "~/" | "$HOME" | "${HOME}")
        || target.starts_with("~/")
        || target.starts_with("$HOME/")
        || target.starts_with("${HOME}/")
    {
        return true;
    }
    if is_within_workspace(target, workspace_root) {
        return false;
    }
    if !Path::new(target).is_absolute() {
        return true;
    }
    const BROAD_PREFIXES: &[&str] = &[
        "/home", "/Users", "/root", "/etc", "/boot", "/sys", "/proc", "/dev", "/usr", "/var",
        "/opt", "/bin", "/sbin", "/lib",
    ];
    BROAD_PREFIXES
        .iter()
        .any(|prefix| target == *prefix || target.starts_with(&format!("{prefix}/")))
}

fn is_block_device(path: &str) -> bool {
    path.starts_with("/dev/")
        && !matches!(
            path,
            "/dev/null" | "/dev/zero" | "/dev/stdout" | "/dev/stderr" | "/dev/stdin"
        )
        && !path.starts_with("/dev/fd/")
}

/// Split arguments into flags and operands; everything after `--` is an
/// operand.
fn split_flags(args: &[String]) -> (Vec<String>, Vec<String>) {
    let mut flags = Vec::new();
    let mut operands = Vec::new();
    let mut only_operands = false;
    for arg in args {
        if only_operands {
            operands.push(arg.clone());
        } else if arg == "--" {
            only_operands = true;
        } else if arg.starts_with('-') && arg.len() > 1 {
            flags.push(arg.clone());
        } else {
            operands.push(arg.clone());
        }
    }
    (flags, operands)
}

/// Does `segment` start with the tokens of `prefix`? The program token is
/// compared by basename, so `/usr/bin/git push` matches `git push`.
pub(crate) fn segment_matches_prefix(segment: &[String], prefix: &[String]) -> bool {
    if prefix.is_empty() || segment.len() < prefix.len() {
        return false;
    }
    segment
        .iter()
        .zip(prefix)
        .enumerate()
        .all(|(i, (have, want))| {
            if i == 0 {
                basename(have) == basename(want)
            } else {
                have == want
            }
        })
}

/// How a shell would read a command line: where it splits and whether it runs
/// anything nested. Quoting follows the shell: nothing is special inside
/// single quotes, and `$(...)` and backticks still run inside double quotes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellScan<'a> {
    /// The simple commands, split at `;`, newlines, `|`, `||`, `&&` and `&`.
    pub segments: Vec<&'a str>,
    /// The line has a control operator or a nested command.
    pub has_operator: bool,
    /// A command runs nested inside another: `$(...)`, backticks, `<(...)`,
    /// `>(...)`. The segments do not show what it is.
    pub has_substitution: bool,
}

/// Read `cmd` the way a shell would.
pub fn scan_shell(cmd: &str) -> ShellScan<'_> {
    let bytes = cmd.as_bytes();
    let at = |index: usize| bytes.get(index).copied();
    let mut segments = Vec::new();
    let (mut start, mut i) = (0, 0);
    let (mut in_single, mut in_double) = (false, false);
    let (mut split, mut has_substitution) = (false, false);
    while i < bytes.len() {
        let byte = bytes[i];
        if in_single {
            in_single = byte != b'\'';
            i += 1;
            continue;
        }
        if in_double {
            match byte {
                b'\\' => i += 1,
                b'"' => in_double = false,
                b'`' => has_substitution = true,
                b'$' if at(i + 1) == Some(b'(') => has_substitution = true,
                _ => {}
            }
            i += 1;
            continue;
        }
        let operator_len = match byte {
            b'\\' => {
                i += 2;
                continue;
            }
            b'\'' => {
                in_single = true;
                0
            }
            b'"' => {
                in_double = true;
                0
            }
            b';' | b'\n' => 1,
            b'|' => {
                if at(i + 1) == Some(b'|') {
                    2
                } else {
                    1
                }
            }
            b'&' if at(i + 1) == Some(b'&') => 2,
            // `2>&1` and `&>file` are redirections, not background jobs.
            b'&' if at(i + 1) == Some(b'>') || (i > 0 && matches!(bytes[i - 1], b'>' | b'<')) => 0,
            b'&' => 1,
            b'`' => {
                has_substitution = true;
                0
            }
            b'$' | b'<' | b'>' if at(i + 1) == Some(b'(') => {
                has_substitution = true;
                0
            }
            _ => 0,
        };
        if operator_len > 0 {
            segments.push(&cmd[start..i]);
            i += operator_len;
            start = i;
            split = true;
        } else {
            i += 1;
        }
    }
    if start < cmd.len() {
        segments.push(&cmd[start..]);
    }
    ShellScan {
        segments: segments
            .into_iter()
            .map(str::trim)
            .filter(|segment| !segment.is_empty())
            .collect(),
        has_operator: split || has_substitution,
        has_substitution,
    }
}

/// Whether a shell would treat `cmd` as more than one plain command.
pub fn contains_shell_operator(cmd: &str) -> bool {
    scan_shell(cmd).has_operator
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs(command: &str) -> Vec<Vec<String>> {
        command_segments(command)
    }

    const WORKSPACE: &str = "/Users/alice/projects/acme";

    #[test]
    fn command_substitution_counts_inside_double_quotes_only() {
        assert!(contains_shell_operator("ls \"$(id)\""));
        assert!(contains_shell_operator("ls \"`id`\""));
        assert!(!contains_shell_operator("echo '$(id)'"));
        assert!(!contains_shell_operator("echo \"a; b | c\""));
        assert!(!contains_shell_operator("echo \"\\$(id)\""));
    }

    #[test]
    fn a_background_job_is_an_operator_but_a_redirection_is_not() {
        assert!(contains_shell_operator("ls & rm x"));
        assert_eq!(scan_shell("ls & rm x").segments, ["ls", "rm x"]);
        assert!(!contains_shell_operator("cargo test 2>&1"));
        assert!(!contains_shell_operator("cargo test &>out.log"));
    }

    #[test]
    fn an_escaped_quote_outside_quotes_does_not_open_one() {
        assert!(contains_shell_operator("echo \\\" ; rm x \\\""));
        assert!(!contains_shell_operator("echo a\\;b"));
    }

    #[test]
    fn a_substitution_is_reported_apart_from_splitting() {
        let scan = scan_shell("ls \"$(id)\"");
        assert!(scan.has_substitution && scan.has_operator);
        assert_eq!(scan.segments, ["ls \"$(id)\""]);
        assert!(!scan_shell("ls -la; pwd").has_substitution);
    }

    #[test]
    fn a_backslash_does_not_escape_inside_single_quotes() {
        assert!(contains_shell_operator("echo 'a\\'; rm x"));
    }

    #[test]
    fn launchers_do_not_hide_a_catastrophic_command() {
        for command in [
            "bash -lc 'rm -rf ~'",
            "zsh -ic 'rm -rf /'",
            "timeout 5 rm -rf ~",
            "timeout -s KILL 5 rm -rf ~",
            "env -u FOO rm -rf ~",
            "env -C /tmp rm -rf ~",
            "sudo --user root rm -rf ~",
            "sudo -u root rm -rf ~",
            "eval \"rm -rf ~\"",
            "stdbuf -o L rm -rf ~",
            "setsid rm -rf ~",
        ] {
            assert!(catastrophic_reason(command).is_some(), "{command}");
            assert!(
                irreversible_delete_reason(command, WORKSPACE).is_some(),
                "{command}"
            );
        }
    }

    #[test]
    fn an_ordinary_launched_command_is_left_alone() {
        for command in [
            "timeout 5 ls",
            "bash -lc 'ls -la'",
            "env FOO=1 git status",
            "nice -n 5 cargo build",
        ] {
            assert!(catastrophic_reason(command).is_none(), "{command}");
        }
    }

    fn blocked(command: &str) -> bool {
        irreversible_delete_reason(command, WORKSPACE).is_some()
    }

    #[test]
    fn broad_deletes_are_blocked_and_scoped_workspace_cleanup_is_not() {
        for command in [
            "rm -rf /",
            "/bin/rm -fr /Users/alice",
            "find /etc -name '*.tmp' -delete",
            "rm -rf .env",
            "rm -rf ../../..",
        ] {
            assert!(blocked(command), "{command}");
        }
        for command in [
            "rm -rf ./target",
            "rm -rf /Users/alice/projects/acme/target",
            "find . -name '*.tmp' -delete",
            "rm -rf -- \"$root\"",
            "rm notes.txt",
            "rm ~/notes.txt",
            "git rm -r ~/Documents",
            "echo 'rm -rf ~/Documents'",
        ] {
            assert!(!blocked(command), "{command}");
        }
    }

    #[test]
    fn launchers_and_subshells_do_not_hide_a_delete() {
        for command in [
            "sudo rm -rf ~/Documents",
            "bash -c 'rm -rf ~/Documents'",
            "env X=1 rm -rf ~/Documents",
            "nohup rm -rf /Users/alice/Documents",
            "sudo find /etc -delete",
            "(rm -rf ~/Documents)",
            "echo ok; $(rm -rf ~/Documents)",
            "true && sh -c \"bash -c 'rm -rf ~/Documents'\"",
        ] {
            assert!(blocked(command), "{command}");
        }
    }

    #[test]
    fn every_delete_form_counts() {
        for command in [
            "rm -r ~/Documents",
            "rm --recursive ~/Documents",
            "rm -R /etc/nginx",
            "rm -rf -- ~/Documents",
            "xargs rm -rf ~/Documents",
            "xargs -n 1 rm -r ~/Documents",
            "find ~/Documents -exec rm -rf {} +",
            "find /etc -execdir rm {} ;",
        ] {
            assert!(blocked(command), "{command}");
        }
        assert!(!blocked("find . -exec rm {} +"));
        assert!(!blocked("find /etc -name '*.conf'"));
        assert!(!blocked("find . | xargs rm -f"));
    }

    #[test]
    fn splits_on_operators_and_honours_quotes() {
        assert_eq!(
            segs("git status && echo 'a; b' | wc -l"),
            vec![
                vec!["git".to_string(), "status".into()],
                vec!["echo".into(), "a; b".into()],
                vec!["wc".into(), "-l".into()],
            ]
        );
    }

    #[test]
    fn unwraps_launchers_and_shell_dash_c() {
        assert_eq!(
            segs("sudo -u root FOO=1 env BAR=2 bash -c \"rm -rf /tmp/x; ls\""),
            vec![
                vec!["rm".to_string(), "-rf".into(), "/tmp/x".into()],
                vec!["ls".to_string()],
            ]
        );
    }

    #[test]
    fn catastrophic_commands_are_detected_structurally() {
        for cmd in [
            "rm -rf /",
            "rm -rf /*",
            "rm -r -f ~",
            "rm -fr ~/",
            "sudo rm --recursive --force /usr",
            "cd /tmp && rm -rf $HOME",
            "echo hi; /bin/rm -Rf //",
            "bash -c 'rm -rf /'",
            "rm --no-preserve-root -rf /x",
            "mkfs.ext4 /dev/sda1",
            "dd if=/dev/zero of=/dev/sda bs=1M",
            "chmod -R 777 /",
            "x=$(rm -rf /)",
        ] {
            assert!(catastrophic_reason(cmd).is_some(), "{cmd}");
        }
    }

    #[test]
    fn ordinary_commands_and_prose_are_not_blocked() {
        for cmd in [
            "rm -rf ./build",
            "rm -rf /tmp/scratch",
            "rm /",
            "echo 'rm -rf /'",
            "git commit -m \"never run rm -rf / again\"",
            "grep -r 'rm -rf ~' docs/",
            "dd if=in.img of=out.img",
            "chmod -R 755 ./dist",
            "ls ~/projects",
        ] {
            assert!(catastrophic_reason(cmd).is_none(), "{cmd}");
        }
    }

    #[test]
    fn prefixes_match_by_tokens() {
        let seg = vec!["/usr/bin/git".to_string(), "push".into(), "origin".into()];
        assert!(segment_matches_prefix(&seg, &["git".into(), "push".into()]));
        assert!(!segment_matches_prefix(&seg, &["git".into(), "pu".into()]));
        assert!(!segment_matches_prefix(&seg, &[]));
    }

    #[test]
    fn test_shell_operator_semicolon() {
        assert!(contains_shell_operator("ls; rm -rf"));
    }

    #[test]
    fn test_shell_operator_pipe() {
        assert!(contains_shell_operator("cat file | grep pattern"));
    }

    #[test]
    fn test_shell_operator_backtick() {
        assert!(contains_shell_operator("echo `whoami`"));
    }

    #[test]
    fn test_shell_operator_and() {
        assert!(contains_shell_operator("cmd1 && cmd2"));
    }

    #[test]
    fn test_shell_operator_subshell() {
        assert!(contains_shell_operator("echo $(whoami)"));
    }

    #[test]
    fn test_no_shell_operator_clean() {
        assert!(!contains_shell_operator("cargo build --release"));
    }

    #[test]
    fn test_no_shell_operator_flags() {
        assert!(!contains_shell_operator("ls -la /tmp"));
    }

    #[test]
    fn test_contains_shell_operator_respects_quotes() {
        // Semicolon inside double quotes — not a shell operator
        assert!(!contains_shell_operator(
            r#"python3 -c "import os; print(1)""#
        ));

        // Semicolon inside single quotes — not a shell operator
        assert!(!contains_shell_operator("python3 -c 'x=1; y=2'"));

        // Real semicolon outside quotes — IS a shell operator
        assert!(contains_shell_operator("echo hello; echo world"));

        // && outside quotes
        assert!(contains_shell_operator("cd /tmp && ls"));

        // && inside quotes — not a shell operator
        assert!(!contains_shell_operator(r#"echo "a && b""#));

        // Pipe inside quotes — not a shell operator
        assert!(!contains_shell_operator(r#"echo "hello | world""#));

        // Real pipe
        assert!(contains_shell_operator("ls | grep test"));
    }
}
