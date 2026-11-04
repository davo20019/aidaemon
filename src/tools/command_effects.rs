//! Structural typing of a shell command's effects for authority decisions.
//!
//! The classifier answers two narrow questions from shell structure alone:
//! does every part of the command only observe, and does any part delete.
//! It uses an exact executable table plus shell-syntax markers (redirects,
//! substitutions); it never reads the command's wording for intent. Anything
//! it cannot prove observational is treated as a mutation (fail closed).

use crate::tools::command_risk::structural_shell_invocations;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CommandEffects {
    /// Every executable is a known observer, with no file redirect, `tee`,
    /// or command substitution.
    pub observation_only: bool,
    /// Some invocation removes files (`rm`, `find -delete`, `git clean`, ...).
    pub deletes: bool,
}

/// Executables whose invocation only reads or prints. Matched on the
/// executable basename; arguments are further checked by
/// [`observation_args_ok`].
const OBSERVATION_PROGRAMS: &[&str] = &[
    "ls", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "find", "stat", "file",
    "pwd", "echo", "printf", "which", "whoami", "id", "uname", "date", "du", "df", "tree", "sort",
    "uniq", "cut", "tr", "diff", "cmp", "basename", "dirname", "realpath", "readlink", "jq", "sed",
    "ps", "hostname", "sw_vers", "true", "git",
];

const DELETE_PROGRAMS: &[&str] = &["rm", "rmdir", "unlink", "shred"];

const FIND_MUTATING_FLAGS: &[&str] = &[
    "-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls",
];

const GIT_OBSERVATION_SUBCOMMANDS: &[&str] = &[
    "status",
    "log",
    "diff",
    "show",
    "rev-parse",
    "ls-files",
    "blame",
];

/// Executables that reach the network whatever their arguments.
const NETWORK_PROGRAMS: &[&str] = &[
    "curl", "wget", "http", "https", "xh", "nc", "ncat", "netcat", "telnet", "ssh", "scp", "sftp",
    "rsync", "ftp", "lftp", "aria2c", "gh",
];

const GIT_NETWORK_SUBCOMMANDS: &[&str] = &["fetch", "clone", "pull", "ls-remote", "submodule"];

const PACKAGE_MANAGERS: &[&str] = &["npm", "pnpm", "yarn", "pip", "pip3", "cargo"];

const PACKAGE_NETWORK_SUBCOMMANDS: &[&str] = &[
    "install", "add", "i", "update", "fetch", "search", "view", "info",
];

/// Whether any structural invocation (including inside `sh -c`) is a known
/// network-reaching executable or subcommand.
pub(crate) fn fetches_network(command: &str) -> bool {
    structural_shell_invocations(command)
        .iter()
        .any(|invocation| {
            let program = basename(&invocation.program);
            let args = &invocation.arguments;
            if NETWORK_PROGRAMS.contains(&program) {
                return true;
            }
            match program {
                // Any argument, so `git -C dir pull` is seen; a stray match only
                // over-taints.
                "git" => args
                    .iter()
                    .any(|arg| GIT_NETWORK_SUBCOMMANDS.contains(&arg.as_str())),
                _ if PACKAGE_MANAGERS.contains(&program) => args
                    .first()
                    .is_some_and(|sub| PACKAGE_NETWORK_SUBCOMMANDS.contains(&sub.as_str())),
                _ => false,
            }
        })
}

pub(crate) fn command_effects(command: &str) -> CommandEffects {
    let invocations = structural_shell_invocations(command);
    let deletes = invocations.iter().any(invocation_deletes);
    let observation_only = !invocations.is_empty()
        && !has_unsafe_shell_syntax(command)
        && invocations.iter().all(|invocation| {
            let program = basename(&invocation.program);
            OBSERVATION_PROGRAMS.contains(&program)
                && observation_args_ok(program, &invocation.arguments)
        });
    CommandEffects {
        observation_only,
        deletes,
    }
}

fn basename(program: &str) -> &str {
    program.rsplit('/').next().unwrap_or(program)
}

fn invocation_deletes(invocation: &crate::tools::command_risk::ShellInvocation) -> bool {
    let program = basename(&invocation.program);
    let args = &invocation.arguments;
    if DELETE_PROGRAMS.contains(&program) {
        return true;
    }
    match program {
        "find" if args.iter().any(|arg| arg == "-delete") => true,
        "git" => args.iter().any(|arg| arg == "clean"),
        // Wrappers that run another program named in their arguments.
        "find" | "xargs" => args
            .iter()
            .any(|arg| DELETE_PROGRAMS.contains(&basename(arg))),
        _ => false,
    }
}

fn observation_args_ok(program: &str, args: &[String]) -> bool {
    match program {
        "find" => !args
            .iter()
            .any(|arg| FIND_MUTATING_FLAGS.contains(&arg.as_str())),
        "sed" => !args.iter().any(|arg| {
            arg == "--in-place"
                || arg.starts_with("--in-place=")
                || (arg.starts_with('-') && !arg.starts_with("--") && arg.contains('i'))
        }),
        // `sort -o FILE` and `tree -o FILE` write their result to a file.
        "sort" => !args.iter().any(|arg| {
            arg == "--output"
                || arg.starts_with("--output=")
                || (arg.starts_with('-') && !arg.starts_with("--") && arg.contains('o'))
        }),
        "tree" => !args.iter().any(|arg| arg == "-o"),
        // `uniq IN OUT` writes OUT.
        "uniq" => args.iter().filter(|arg| !arg.starts_with('-')).count() < 2,
        "git" => {
            if args.iter().any(|arg| arg.starts_with("--output")) {
                return false;
            }
            let Some(index) = args.iter().position(|arg| !arg.starts_with('-')) else {
                return false;
            };
            let subcommand = args[index].as_str();
            if GIT_OBSERVATION_SUBCOMMANDS.contains(&subcommand) {
                return true;
            }
            subcommand == "branch"
                && args[index + 1..]
                    .iter()
                    .all(|arg| matches!(arg.as_str(), "--list" | "-a" | "-r" | "-v"))
        }
        _ => true,
    }
}

/// Quote-aware scan (same style as the terminal's operator scan) for syntax
/// that can write or run something the executable table cannot see: a file
/// redirect, process substitution, command substitution, `tee`, or quoting
/// the scan cannot close. An fd duplication (`2>&1`) and `/dev/null` targets
/// are not writes.
fn has_unsafe_shell_syntax(command: &str) -> bool {
    let chars: Vec<char> = command.chars().collect();
    let mut i = 0;
    let mut in_single = false;
    let mut in_double = false;
    while i < chars.len() {
        let c = chars[i];
        if in_single {
            in_single = c != '\'';
            i += 1;
            continue;
        }
        match c {
            '\\' => {
                i += 2;
                continue;
            }
            '\'' if !in_double => in_single = true,
            '"' => in_double = !in_double,
            // Substitutions expand inside double quotes too.
            '`' => return true,
            '$' if chars.get(i + 1) == Some(&'(') => return true,
            '>' | '<' if !in_double => {
                let next = chars.get(i + 1).copied();
                if next == Some('(') {
                    return true;
                }
                if c == '>' {
                    let mut j = i + 1;
                    if chars.get(j) == Some(&'>') || chars.get(j) == Some(&'|') {
                        j += 1;
                    }
                    if chars.get(j) == Some(&'&') {
                        let dup = chars.get(j + 1).copied();
                        if dup.is_some_and(|d| d.is_ascii_digit() || d == '-') {
                            i = j + 1;
                            continue;
                        }
                        j += 1;
                    }
                    while chars.get(j).is_some_and(|c| c.is_whitespace()) {
                        j += 1;
                    }
                    let target: String = chars[j..]
                        .iter()
                        .take_while(|c| !c.is_whitespace() && !matches!(c, ';' | '|' | '&'))
                        .collect();
                    if target != "/dev/null" {
                        return true;
                    }
                    i = j;
                    continue;
                }
            }
            _ => {}
        }
        i += 1;
    }
    if in_single || in_double {
        return true;
    }
    // `tee` writes its stdin to files; it is not in the observation table,
    // and this keeps a `tee` hidden behind a wrapper from slipping through.
    structural_shell_invocations(command)
        .iter()
        .any(|invocation| basename(&invocation.program) == "tee")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_effects_types_observation_by_structure() {
        for observing in [
            "ls /tmp",
            "git status",
            "grep -r x .",
            "cat a 2>&1 | head",
            "ls > /dev/null",
            "ls 2>/dev/null",
            "echo 'a > b'",
            "git branch --list",
            "git branch",
            "find . -name '*.rs'",
            "sed -n 1,5p f",
        ] {
            assert!(
                command_effects(observing).observation_only,
                "{observing} should be observation-only"
            );
        }
        for mutating in [
            "rm -rf /tmp/aidaemon-test/x",
            "ls /tmp; rm -rf /tmp/aidaemon-test/x",
            "ls && rm x",
            "cat a > b",
            "echo x >> b",
            "ls | xargs rm",
            "ls | tee out",
            "echo $(curl example.invalid)",
            "echo \"$(curl example.invalid)\"",
            "echo `id`",
            "find . -delete",
            "find . -exec rm {} ;",
            "sed -i s/a/b/ f",
            "sed -ni p f",
            "sed --in-place p f",
            "git push",
            "git branch -D x",
            "git clean -fd",
            "python3 script.py",
            "curl https://example.invalid",
            "sh -c 'rm x'",
            "cat a >&out",
            "echo hi &> out",
            "sort -o out in",
            "git diff --output=out",
            "uniq in out",
            "echo 'unterminated",
            "",
        ] {
            assert!(
                !command_effects(mutating).observation_only,
                "{mutating} should not be observation-only"
            );
        }
    }

    #[test]
    fn fetches_network_by_executable_table() {
        for fetching in [
            "curl https://example.invalid",
            "ls && wget x",
            "git clone u",
            "git -C d pull",
            "npm install x",
            "cargo add serde",
            "sh -c 'ssh host'",
            "gh pr list",
        ] {
            assert!(fetches_network(fetching), "{fetching}");
        }
        for local in ["git status", "npm test", "cargo build", "echo curl", "ls"] {
            assert!(!fetches_network(local), "{local}");
        }
    }

    #[test]
    fn command_effects_flags_deletion_structurally() {
        for deleting in [
            "rm x",
            "find . -delete",
            "git clean -fd",
            "ls; rmdir d",
            "sh -c 'unlink x'",
            "ls | xargs rm",
            "find . -exec shred {} ;",
        ] {
            assert!(command_effects(deleting).deletes, "{deleting}");
        }
        for benign in [
            "ls",
            "git status",
            "echo rm",
            "grep rm file",
            "find . -name x",
        ] {
            assert!(!command_effects(benign).deletes, "{benign}");
        }
    }
}
