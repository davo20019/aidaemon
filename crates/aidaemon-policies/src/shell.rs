//! A small, conservative shell tokenizer used by command policies.
//!
//! It is not a shell. It splits a command line into simple-command segments
//! at control operators (`;`, `&&`, `||`, `|`, `&`, newlines, subshell and
//! command-substitution boundaries), honours quotes and backslashes, and
//! unwraps common launchers (`sudo`, `env`, `nohup`, `bash -c "..."`) so a
//! policy sees the program that actually runs and its real arguments.

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
        if depth < 4 && is_shell(&stripped[0]) {
            if let Some(pos) = stripped.iter().position(|t| t == "-c") {
                if let Some(inner) = stripped.get(pos + 1) {
                    collect_segments(inner, depth + 1, out);
                    continue;
                }
            }
        }
        out.push(stripped);
    }
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
                if let Some(next) = chars.next() {
                    if next != '\n' {
                        current.push(next);
                    }
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
                i += 1;
                while i < tokens.len() && tokens[i].starts_with('-') {
                    let takes_value = matches!(
                        tokens[i].as_str(),
                        "-u" | "-g" | "-C" | "-h" | "-p" | "-U" | "-r" | "-t"
                    );
                    i += if takes_value { 2 } else { 1 };
                }
            }
            "env" => {
                i += 1;
                while i < tokens.len() && (tokens[i].starts_with('-') || is_assignment(&tokens[i]))
                {
                    i += 1;
                }
            }
            "nohup" | "command" | "exec" | "time" | "builtin" => i += 1,
            "nice" | "ionice" => {
                i += 1;
                while i < tokens.len() && tokens[i].starts_with('-') {
                    i += if tokens[i] == "-n" { 2 } else { 1 };
                }
            }
            _ => break,
        }
    }
    tokens[i.min(tokens.len())..].to_vec()
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
            if recursive {
                if let Some(target) = operands.iter().find(|o| is_catastrophic_target(o)) {
                    return Some(format!("recursive `rm` of `{target}` is never allowed"));
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn segs(command: &str) -> Vec<Vec<String>> {
        command_segments(command)
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
}
