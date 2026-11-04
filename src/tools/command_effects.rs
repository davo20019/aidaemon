//! Structural typing of a shell command's effects for authority decisions.
//!
//! Fail closed: `observation_only` is an allowlist of shapes this module fully
//! understands (an exact executable table, per-program option rules, and a
//! quote-aware syntax scan). Anything unusual (env assignments, heredocs,
//! comments, newlines, groupings, wrappers, unparsable segments) is simply not
//! observation-only. `deletes` and `fetches_network` err the other way: they
//! unwrap wrappers and groupings and over-report rather than miss.
//! No wording of the command is ever read for intent.

use crate::tools::command_risk::structural_shell_invocations;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommandEffects {
    /// Every executable is a known observer used in a known observing shape.
    pub observation_only: bool,
    /// Some command removes files (`rm`, `find -delete`, `git clean`, ...),
    /// including behind wrappers and groupings.
    pub deletes: bool,
    /// Every command is ordinary local work: an exact `LOCAL_WORK` program
    /// in an allowed shape, or an observation. Fail closed like
    /// `observation_only`, except that output redirection into files and
    /// newline-separated scripts are allowed.
    pub local_work_only: bool,
    /// The paths deleting commands name, as written. `None` when a delete
    /// target cannot be read structurally (variables, substitutions, piped
    /// arguments, `git clean`).
    pub delete_targets: Option<Vec<String>>,
}

/// Executables whose invocation only reads or prints, matched on the bare
/// name (or an absolute path in a system bin directory). `sed` is absent on
/// purpose: its script can write (`w`) or execute (`e`) and cannot be checked
/// structurally.
const OBSERVATION_PROGRAMS: &[&str] = &[
    "ls", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "find", "stat", "file",
    "pwd", "echo", "printf", "which", "whoami", "id", "uname", "date", "du", "df", "tree", "sort",
    "uniq", "cut", "tr", "diff", "cmp", "basename", "dirname", "realpath", "readlink", "jq", "ps",
    "hostname", "sw_vers", "true", "git",
];

const SYSTEM_BIN_DIRS: &[&str] = &[
    "/bin",
    "/usr/bin",
    "/usr/local/bin",
    "/opt/homebrew/bin",
    "/sbin",
    "/usr/sbin",
];

const DELETE_PROGRAMS: &[&str] = &["rm", "rmdir", "unlink", "shred"];

/// Ordinary local work on the owner's machine, matched on the bare name:
/// builds, tests, local version control and file shuffling. Programs with a
/// subcommand list are local work only for those subcommands; the rest take
/// any arguments. `cd` changes only the shell's own directory. Deleting
/// programs are listed because where they point is judged separately.
const LOCAL_WORK_PROGRAMS: &[&str] = &[
    "make",
    "cmake",
    "ninja",
    "pytest",
    "rustc",
    "xcodebuild",
    "mkdir",
    "touch",
    "cp",
    "mv",
    "ln",
    "chmod",
    "cd",
    "rm",
    "rmdir",
    "unlink",
];

const LOCAL_WORK_SUBCOMMANDS: &[(&str, &[&str])] = &[
    (
        "cargo",
        &[
            "build", "check", "test", "clippy", "fmt", "run", "bench", "doc", "tree", "metadata",
        ],
    ),
    (
        "git",
        &[
            "add",
            "commit",
            "status",
            "diff",
            "log",
            "show",
            "branch",
            "switch",
            "checkout",
            "restore",
            "stash",
            "merge",
            "rebase",
            "reset",
            "tag",
            "rm",
            "mv",
            "init",
            "worktree",
            "rev-parse",
            "ls-files",
            "blame",
        ],
    ),
    ("npm", &["test", "run", "build", "lint"]),
    ("pnpm", &["test", "run", "build", "lint"]),
    ("yarn", &["test", "run", "build", "lint"]),
    ("go", &["build", "test", "vet", "fmt", "run"]),
    ("swift", &["build", "test"]),
];

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

/// Programs that run another command named in their arguments (and shell
/// reserved words that introduce one). The inner command is searched at every
/// later argument position, so option operands cannot hide it.
const WRAPPERS: &[&str] = &[
    "sudo", "doas", "nice", "nohup", "timeout", "stdbuf", "time", "command", "exec", "xargs",
    "env", "builtin", "ionice", "setsid", "{", "!", "if", "then", "else", "elif", "do", "while",
    "until",
];

/// Executables that reach the network whatever their arguments.
const NETWORK_PROGRAMS: &[&str] = &[
    "curl", "wget", "http", "https", "xh", "nc", "ncat", "netcat", "telnet", "ssh", "scp", "sftp",
    "rsync", "ftp", "lftp", "aria2c", "gh", "npx", "bunx", "uvx", "svn", "socat", "ping", "dig",
    "host", "nslookup", "whois", "tftp", "mosh", "rclone", "aws", "gcloud", "kubectl",
];

/// (program, subcommands): network when ANY non-option argument matches.
const NETWORK_SUBCOMMANDS: &[(&str, &[&str])] = &[
    (
        "git",
        &[
            "fetch",
            "clone",
            "pull",
            "ls-remote",
            "submodule",
            "push",
            "update",
        ],
    ),
    (
        "npm",
        &[
            "install", "add", "i", "ci", "exec", "update", "fetch", "search", "view", "info",
            "publish",
        ],
    ),
    (
        "pnpm",
        &[
            "install", "add", "i", "dlx", "update", "fetch", "search", "view", "info", "publish",
        ],
    ),
    (
        "yarn",
        &[
            "install", "add", "i", "update", "fetch", "search", "view", "info", "publish",
        ],
    ),
    ("pip", &["install", "download"]),
    ("pip3", &["install", "download"]),
    (
        "cargo",
        &[
            "install", "add", "update", "fetch", "search", "view", "info", "publish",
        ],
    ),
    ("bun", &["install", "add", "i"]),
    ("uv", &["pip", "add", "sync"]),
    ("brew", &["install", "update", "upgrade", "fetch"]),
    ("go", &["get", "install", "download"]),
    ("docker", &["pull", "run", "build", "push"]),
    ("gem", &["install"]),
    ("apt", &["install", "update"]),
    ("apt-get", &["install", "update"]),
    ("hg", &["pull", "clone"]),
    ("openssl", &["s_client"]),
];

/// Whether any command (behind wrappers, groupings, substitutions and
/// `sh -c`) is a known network-reaching executable or subcommand.
pub(crate) fn fetches_network(command: &str) -> bool {
    if command.contains("/dev/tcp/") || command.contains("/dev/udp/") || command.contains("$'") {
        return true;
    }
    expanded_commands(command).iter().any(|tokens| {
        let program = basename(&tokens[0]);
        let args = &tokens[1..];
        if NETWORK_PROGRAMS.contains(&program) {
            return true;
        }
        if program == "yarn" && args.iter().all(|arg| arg.starts_with('-')) {
            return true;
        }
        if program == "git" {
            let has = |word: &str| args.iter().any(|arg| arg == word);
            if (has("remote") && has("update"))
                || (has("archive") && args.iter().any(|arg| arg.starts_with("--remote")))
            {
                return true;
            }
        }
        NETWORK_SUBCOMMANDS.iter().any(|(name, subs)| {
            *name == program
                && args
                    .iter()
                    .any(|arg| !arg.starts_with('-') && subs.contains(&arg.as_str()))
        })
    })
}

/// Whether writing `path` can plant git configuration. A structural path
/// fact, deliberately over-reporting:
/// - a `.git`, `.gitattributes` or `.gitmodules` component, or a `.gitconfig`
///   file name, compared ASCII-case-insensitively (APFS and NTFS fold case)
///   after stripping trailing dots and spaces (Windows drops them);
/// - global and system configuration: `~/.config/git/**`,
///   `$XDG_CONFIG_HOME/git/**`, `/etc/gitconfig`, and any `.config/git`
///   component pair (relative paths may resolve under the home directory).
///
/// `~`, `$HOME` and `${HOME}` prefixes are expanded and `.`/`..` are
/// normalized lexically before the global checks.
pub(crate) fn is_git_config_path(path: &str) -> bool {
    let raw = path_components(path);
    if raw.iter().any(|part| is_repo_config_component(part)) {
        return true;
    }
    let expanded = expand_home(path);
    let normalized = normalize_components(&path_components(&expanded));
    if normalized.iter().any(|part| is_repo_config_component(part))
        || normalized.windows(2).any(|pair| {
            pair[0].eq_ignore_ascii_case(".config") && pair[1].eq_ignore_ascii_case("git")
        })
    {
        return true;
    }
    let absolute = expanded.starts_with('/') || expanded.starts_with('\\');
    if absolute
        && normalized.len() == 2
        && normalized[0].eq_ignore_ascii_case("etc")
        && normalized[1].eq_ignore_ascii_case("gitconfig")
    {
        return true;
    }
    let under = |base: std::path::PathBuf| {
        let base = normalize_components(&path_components(&base.to_string_lossy()));
        absolute
            && !base.is_empty()
            && normalized.len() > base.len()
            && normalized
                .iter()
                .zip(&base)
                .all(|(a, b)| a.eq_ignore_ascii_case(b))
    };
    if dirs::home_dir().is_some_and(|home| under(home.join(".config").join("git"))) {
        return true;
    }
    std::env::var_os("XDG_CONFIG_HOME")
        .filter(|value| !value.is_empty())
        .is_some_and(|xdg| under(std::path::PathBuf::from(xdg).join("git")))
}

/// Components of `path` with Windows-insignificant trailing dots and spaces
/// removed (`.` and `..` are kept as-is).
fn path_components(path: &str) -> Vec<String> {
    path.split(['/', '\\'])
        .filter(|part| !part.is_empty())
        .map(|part| {
            if part == "." || part == ".." {
                part.to_string()
            } else {
                part.trim_end_matches(['.', ' ']).to_string()
            }
        })
        .filter(|part| !part.is_empty())
        .collect()
}

fn normalize_components(parts: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for part in parts {
        match part.as_str() {
            "." => {}
            ".." => {
                if out.last().is_some_and(|last| last != "..") {
                    out.pop();
                } else {
                    out.push(part.clone());
                }
            }
            _ => out.push(part.clone()),
        }
    }
    out
}

fn is_repo_config_component(part: &str) -> bool {
    [".git", ".gitattributes", ".gitmodules", ".gitconfig"]
        .iter()
        .any(|name| part.eq_ignore_ascii_case(name))
}

fn expand_home(path: &str) -> String {
    let rest = ["~", "$HOME", "${HOME}"].iter().find_map(|prefix| {
        path.strip_prefix(prefix)
            .filter(|rest| rest.is_empty() || rest.starts_with(['/', '\\']))
    });
    match (rest, dirs::home_dir()) {
        (Some(rest), Some(home)) => format!("{}{rest}", home.to_string_lossy()),
        _ => path.to_string(),
    }
}

/// Whether the directory `root` (home-expanded, resolved against `base` when
/// relative) holds a `.git` entry directly inside it, so a write anywhere
/// under it can plant repository configuration.
pub(crate) fn root_holds_repository(root: &str, base: Option<&str>) -> bool {
    let root = root.trim();
    if root.is_empty() {
        return false;
    }
    let mut path = std::path::PathBuf::from(expand_home(root));
    if path.is_relative() {
        if let Some(base) = base.map(str::trim).filter(|base| !base.is_empty()) {
            path = std::path::PathBuf::from(expand_home(base)).join(path);
        }
    }
    path.join(".git").exists()
}

/// Whether any shell word of `command` (including redirect targets,
/// `--opt=value` operands, substitution bodies and nested `sh -c` scripts)
/// names a git configuration path per `is_git_config_path`.
pub(crate) fn command_names_git_config_path(command: &str) -> bool {
    names_git_config(command, 0)
}

fn names_git_config(command: &str, depth: usize) -> bool {
    if depth > MAX_DEPTH {
        return false;
    }
    let scanned = scan(command);
    for segment in &scanned.segments {
        let words = shell_words::split(segment)
            .unwrap_or_else(|_| segment.split_whitespace().map(str::to_string).collect());
        for word in words {
            let target = word.trim_start_matches(|c: char| {
                c.is_ascii_digit() || matches!(c, '>' | '<' | '&' | '|')
            });
            let operand = word.split_once('=').map(|(_, value)| value);
            if [Some(word.as_str()), Some(target), operand]
                .into_iter()
                .flatten()
                .any(|candidate| !candidate.is_empty() && is_git_config_path(candidate))
            {
                return true;
            }
            if word.contains(char::is_whitespace) && names_git_config(&word, depth + 1) {
                return true;
            }
        }
    }
    substitution_bodies(command)
        .iter()
        .any(|body| names_git_config(body, depth + 1))
}

pub(crate) fn command_effects(command: &str) -> CommandEffects {
    // ANSI-C quoting (`$'..'`) escapes quotes in a way the scan cannot
    // follow, so a command may hide behind it: over-report.
    let deletes = command.contains("$'")
        || expanded_commands(command).iter().any(|tokens| {
            let program = basename(&tokens[0]);
            let args = &tokens[1..];
            DELETE_PROGRAMS.contains(&program)
                || (program == "find" && args.iter().any(|arg| arg == "-delete"))
                || (program == "git" && args.iter().any(|arg| arg == "clean"))
        });
    CommandEffects {
        observation_only: is_observation_only(command),
        deletes,
        local_work_only: is_local_work_only(command),
        delete_targets: if command.contains("$'") {
            None
        } else {
            delete_targets(command)
        },
    }
}

/// Paths named by every deleting command, or `None` when any cannot be
/// determined: a target with shell expansion (variables, substitutions,
/// globs, braces, `~`) or a `find` placeholder, a deleting command with no
/// target (arguments arriving through a pipe), `find` without explicit
/// start paths, `git clean` (its targets are whatever is untracked), or a
/// relative target in a command that changes directory (`cd`, `pushd`,
/// `popd`), since the shell then deletes relative to somewhere else.
fn delete_targets(command: &str) -> Option<Vec<String>> {
    let unreadable = |target: &str| {
        target.is_empty() || target.contains(['$', '`', '\\', '{', '}', '*', '?', '[', '~'])
    };
    let commands = expanded_commands(command);
    let changes_directory = commands
        .iter()
        .any(|tokens| matches!(basename(&tokens[0]), "cd" | "pushd" | "popd"));
    let mut targets: Vec<String> = Vec::new();
    for tokens in &commands {
        let program = basename(&tokens[0]);
        let args = &tokens[1..];
        let named: Vec<&String> = if DELETE_PROGRAMS.contains(&program) {
            let mut options_done = false;
            args.iter()
                .filter(|arg| {
                    if options_done {
                        return true;
                    }
                    if arg.as_str() == "--" {
                        options_done = true;
                        return false;
                    }
                    !arg.starts_with('-')
                })
                .collect()
        } else if program == "find" && args.iter().any(|arg| arg == "-delete") {
            // Start paths come first; a leading option (`-L`, `-H`, `-O3`,
            // `-D x`, BSD `-f path`) makes them unreadable here, and none at
            // all means the shell's own directory.
            if args.first().is_none_or(|arg| arg.starts_with('-')) {
                return None;
            }
            args.iter()
                .take_while(|arg| !arg.starts_with(['-', '(', '!']))
                .collect()
        } else if program == "git" && args.iter().any(|arg| arg == "clean") {
            return None;
        } else {
            continue;
        };
        if named.is_empty() || named.iter().any(|target| unreadable(target)) {
            return None;
        }
        if changes_directory && named.iter().any(|target| !target.starts_with('/')) {
            return None;
        }
        for target in named {
            if !targets.contains(target) {
                targets.push(target.clone());
            }
        }
    }
    Some(targets)
}

/// Every operand a command could read as a path, as written: each
/// non-option argument of every reachable command (patterns included; they
/// are judged as paths, which only over-reports), and the value of a
/// `--option=value`. `None` when any cannot be determined structurally: shell
/// expansion (variables, substitutions, globs, braces, `~` other than a
/// leading `~/`), ANSI-C quoting, an option with a path glued to it
/// (`-I/x`), or a command that changes directory (`cd`, `pushd`, `popd`),
/// since relative operands then resolve somewhere else.
pub(crate) fn path_operands(command: &str) -> Option<Vec<String>> {
    if command.contains("$'") {
        return None;
    }
    let unreadable = |operand: &str| {
        let rest = operand
            .strip_prefix("~/")
            .or_else(|| (operand == "~").then_some(""))
            .unwrap_or(operand);
        rest.contains(['$', '`', '\\', '{', '}', '*', '?', '[', '~'])
    };
    let commands = expanded_commands(command);
    let mut operands: Vec<String> = Vec::new();
    for tokens in &commands {
        if matches!(basename(&tokens[0]), "cd" | "pushd" | "popd") {
            return None;
        }
        let mut options_done = false;
        for arg in &tokens[1..] {
            let operand = if options_done || !arg.starts_with('-') {
                arg.as_str()
            } else if arg == "--" {
                options_done = true;
                continue;
            } else if let Some((_, value)) = arg.split_once('=') {
                value
            } else if arg.contains(['/', '~']) {
                return None;
            } else {
                continue;
            };
            if operand.is_empty() {
                continue;
            }
            if unreadable(operand) {
                return None;
            }
            if !operands.iter().any(|known| known == operand) {
                operands.push(operand.to_string());
            }
        }
    }
    Some(operands)
}

fn is_local_work_only(command: &str) -> bool {
    if command.contains("/dev/tcp/") || command.contains("/dev/udp/") || command.contains("$'") {
        return false;
    }
    let mut any = false;
    // A script's lines are commands like `;`-separated ones.
    for line in command.replace("\\\n", " ").split('\n') {
        let scanned = scan(line);
        if scanned.unsafe_syntax {
            return false;
        }
        for segment in &scanned.segments {
            if segment.trim().is_empty() {
                continue;
            }
            let Ok(tokens) = shell_words::split(segment) else {
                return false;
            };
            let tokens = strip_redirections(tokens);
            let Some(program) = tokens.first() else {
                return false;
            };
            if is_env_assignment(program) {
                return false;
            }
            let path_ok = !program.contains('/')
                || program
                    .rsplit_once('/')
                    .is_some_and(|(dir, _)| SYSTEM_BIN_DIRS.contains(&dir));
            if !path_ok || !local_work_args_ok(basename(program), &tokens[1..]) {
                return false;
            }
            any = true;
        }
    }
    any
}

/// Options through which a local-work program runs code given inline (or
/// read from stdin) rather than a file in the workspace.
fn runs_inline_code(program: &str, args: &[String]) -> bool {
    let option = |names: &[&str]| {
        args.iter().any(|arg| {
            names.iter().any(|name| match arg.strip_prefix(name) {
                // A short option may carry its value attached (`-Ex`).
                Some(_) if name.len() == 2 => true,
                Some(rest) => rest.is_empty() || rest.starts_with('='),
                None => false,
            })
        })
    };
    match program {
        // `-c` anywhere: a configuration override before the subcommand, or
        // a subcommand option that can carry one.
        "git" => {
            option(&["-c", "--config-env", "--exec"])
                || (args.first().is_some_and(|sub| sub == "rebase") && option(&["-x"]))
        }
        "cargo" => option(&["--config"]),
        "make" => option(&["--eval", "-E"]),
        "go" => option(&["-toolexec", "--toolexec", "-exec", "--exec"]),
        // The script arrives on stdin or a descriptor.
        "python" | "python3" | "node" => args.first().is_some_and(|script| {
            script == "-" || script == "/dev/stdin" || script.starts_with("/dev/fd/")
        }),
        _ => false,
    }
}

fn local_work_args_ok(program: &str, args: &[String]) -> bool {
    if runs_inline_code(program, args) {
        return false;
    }
    if let Some((_, subcommands)) = LOCAL_WORK_SUBCOMMANDS
        .iter()
        .find(|(name, _)| *name == program)
    {
        // A toolchain selector may precede cargo's subcommand; any other
        // leading option (git's `-c`, `-C`) is not local work.
        let rest = match program {
            "cargo" => args
                .iter()
                .skip_while(|arg| arg.starts_with('+'))
                .collect::<Vec<_>>(),
            _ => args.iter().collect(),
        };
        return rest
            .first()
            .is_some_and(|subcommand| subcommands.contains(&subcommand.as_str()));
    }
    match program {
        // A script path, never inline code (`-c`, `-e`) or a module (`-m`).
        "python" | "python3" | "node" => args.first().is_some_and(|arg| !arg.starts_with('-')),
        "find" => !args
            .iter()
            .any(|arg| arg != "-delete" && FIND_MUTATING_FLAGS.contains(&arg.as_str())),
        _ if LOCAL_WORK_PROGRAMS.contains(&program) => true,
        _ if OBSERVATION_PROGRAMS.contains(&program) => observation_args_ok(program, args),
        _ => false,
    }
}

fn basename(program: &str) -> &str {
    program.rsplit('/').next().unwrap_or(program)
}

fn is_env_assignment(token: &str) -> bool {
    token.split_once('=').is_some_and(|(name, _)| {
        !name.is_empty()
            && name.chars().enumerate().all(|(index, ch)| {
                ch == '_' || ch.is_ascii_alphanumeric() && (index > 0 || !ch.is_ascii_digit())
            })
    })
}

/// Result of the quote-aware scan.
struct Scan {
    segments: Vec<String>,
    /// Syntax that can write, execute, or hide something: file redirects,
    /// input redirects other than `/dev/null`, heredocs, substitutions,
    /// process substitution, comments, newlines, unclosed quotes.
    unsafe_syntax: bool,
    /// Output redirected into a file (not `/dev/null` or a descriptor).
    writes_file: bool,
}

fn redirect_target(chars: &[char], mut j: usize) -> String {
    while chars.get(j).is_some_and(|c| c.is_whitespace()) {
        j += 1;
    }
    chars[j.min(chars.len())..]
        .iter()
        .take_while(|c| !c.is_whitespace() && !matches!(c, ';' | '|' | '&' | '(' | ')' | '<' | '>'))
        .collect()
}

fn scan(command: &str) -> Scan {
    let chars: Vec<char> = command.chars().collect();
    let mut segments = Vec::new();
    let mut cur = String::new();
    let mut unsafe_syntax = false;
    let mut writes_file = false;
    let (mut in_single, mut in_double) = (false, false);
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if in_single {
            in_single = c != '\'';
            cur.push(c);
            i += 1;
            continue;
        }
        let next = chars.get(i + 1).copied();
        match c {
            '\\' => {
                cur.push(c);
                if let Some(n) = next {
                    cur.push(n);
                }
                i += 2;
                continue;
            }
            '\n' => {
                unsafe_syntax = true;
                segments.push(std::mem::take(&mut cur));
            }
            '\'' if !in_double => {
                in_single = true;
                cur.push(c);
            }
            '"' => {
                in_double = !in_double;
                cur.push(c);
            }
            // Substitutions expand inside double quotes too.
            '`' => {
                unsafe_syntax = true;
                cur.push(c);
            }
            '$' if next == Some('(') || (next == Some('\'') && !in_double) => {
                unsafe_syntax = true;
                cur.push(c);
            }
            _ if in_double => cur.push(c),
            '#' if cur.chars().last().is_none_or(|p| p.is_whitespace()) => {
                unsafe_syntax = true;
                cur.push(c);
            }
            ';' | '(' | ')' => {
                if c != ';' {
                    unsafe_syntax = true;
                }
                segments.push(std::mem::take(&mut cur));
            }
            '|' => segments.push(std::mem::take(&mut cur)),
            '&' if next == Some('>') => cur.push(c),
            '&' => segments.push(std::mem::take(&mut cur)),
            '<' => {
                if matches!(next, Some('<') | Some('(') | Some('>'))
                    || redirect_target(&chars, i + 1) != "/dev/null"
                {
                    unsafe_syntax = true;
                }
                cur.push(c);
            }
            '>' => {
                let mut j = i + 1;
                if matches!(chars.get(j), Some('>') | Some('|')) {
                    j += 1;
                }
                let mut dup = false;
                if chars.get(j) == Some(&'&') {
                    dup = chars
                        .get(j + 1)
                        .is_some_and(|d| d.is_ascii_digit() || *d == '-');
                    j += 1;
                }
                if next == Some('(') {
                    unsafe_syntax = true;
                } else if !dup && redirect_target(&chars, j) != "/dev/null" {
                    writes_file = true;
                }
                cur.extend(&chars[i..j]);
                i = j;
                continue;
            }
            _ => cur.push(c),
        }
        i += 1;
    }
    if in_single || in_double {
        unsafe_syntax = true;
    }
    segments.push(cur);
    Scan {
        segments,
        unsafe_syntax,
        writes_file,
    }
}

fn is_redirect_word(token: &str) -> bool {
    let t = token.trim_start_matches(|c: char| c.is_ascii_digit());
    t.starts_with('>') || t.starts_with('<') || t.starts_with("&>")
}

/// Drop redirection words (and a bare operator's target). The scan has
/// already vetted every redirect.
fn strip_redirections(tokens: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    let mut iter = tokens.into_iter();
    while let Some(token) = iter.next() {
        if is_redirect_word(&token) {
            let bare = token.trim_start_matches(|c: char| c.is_ascii_digit());
            if bare.chars().all(|c| matches!(c, '>' | '<' | '&' | '|')) {
                iter.next();
            }
            continue;
        }
        out.push(token);
    }
    out
}

fn is_observation_only(command: &str) -> bool {
    let scanned = scan(command);
    if scanned.unsafe_syntax
        || scanned.writes_file
        || command.contains("/dev/tcp/")
        || command.contains("/dev/udp/")
    {
        return false;
    }
    let mut any = false;
    for segment in &scanned.segments {
        if segment.trim().is_empty() {
            continue;
        }
        let Ok(tokens) = shell_words::split(segment) else {
            return false;
        };
        let tokens = strip_redirections(tokens);
        let Some(first) = tokens.first() else {
            return false;
        };
        if is_env_assignment(first) {
            return false;
        }
        // `command ls` is the one wrapper that stays an observation.
        let tokens = if first == "command" {
            &tokens[1..]
        } else {
            &tokens[..]
        };
        let Some(program) = tokens.first() else {
            return false;
        };
        let name = basename(program);
        let path_ok = !program.contains('/')
            || program
                .rsplit_once('/')
                .is_some_and(|(dir, _)| SYSTEM_BIN_DIRS.contains(&dir));
        if !path_ok
            || !OBSERVATION_PROGRAMS.contains(&name)
            || !observation_args_ok(name, &tokens[1..])
        {
            return false;
        }
        any = true;
    }
    any
}

/// `tree` is observation-only with only these read-only flags; `-o FILE` and
/// `-R` write files and anything unknown is rejected.
fn tree_args_ok(args: &[String]) -> bool {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if matches!(arg.as_str(), "-L" | "-I" | "-P") {
            if iter.next().is_none() {
                return false;
            }
        } else if let Some(long) = arg.strip_prefix("--") {
            if !matches!(long, "noreport" | "dirsfirst") {
                return false;
            }
        } else if let Some(cluster) = arg.strip_prefix('-') {
            if cluster.is_empty() || !cluster.chars().all(|c| "adfshDCn".contains(c)) {
                return false;
            }
        }
    }
    true
}

fn short_cluster_has(arg: &str, letters: &[char]) -> bool {
    arg.starts_with('-') && !arg.starts_with("--") && arg.chars().any(|c| letters.contains(&c))
}

fn observation_args_ok(program: &str, args: &[String]) -> bool {
    let any = |pred: &dyn Fn(&str) -> bool| args.iter().any(|arg| pred(arg));
    match program {
        "find" => !any(&|a| FIND_MUTATING_FLAGS.contains(&a)),
        "rg" => !any(&|a| {
            a == "--pre"
                || a.starts_with("--pre=")
                || a.starts_with("--pre-glob")
                || a.starts_with("--hostname-bin")
                || a == "--search-zip"
                || short_cluster_has(a, &['z'])
        }),
        "sort" => !any(&|a| {
            a.starts_with("--compress-program")
                || a.starts_with("--output")
                || short_cluster_has(a, &['o'])
        }),
        // A bare `-` is a positional (stdin), so `uniq - out` writes `out`.
        "uniq" => {
            args.iter()
                .filter(|arg| !arg.starts_with('-') || arg.as_str() == "-")
                .count()
                < 2
        }
        "tree" => tree_args_ok(args),
        "date" => args.is_empty() || (args.len() == 1 && args[0].starts_with('+')),
        "hostname" => args.is_empty(),
        "file" => !any(&|a| {
            a == "--compile" || a.starts_with("--magic-file") || short_cluster_has(a, &['C', 'm'])
        }),
        "git" => {
            let mut index = 0;
            while let Some(arg) = args.get(index).filter(|arg| arg.starts_with('-')) {
                // The classifier has no roots, so it cannot vouch for a
                // redirected repository.
                if arg == "-C" || arg.starts_with("--git-dir") || arg.starts_with("--work-tree") {
                    return false;
                }
                if arg == "-c" || arg.starts_with("--config-env") || arg.starts_with("--exec-path")
                {
                    return false;
                }
                index += 1;
            }
            let Some(subcommand) = args.get(index) else {
                return false;
            };
            let rest = &args[index + 1..];
            if rest
                .iter()
                .any(|a| a == "--ext-diff" || a.starts_with("--output") || a == "--textconv")
            {
                return false;
            }
            if GIT_OBSERVATION_SUBCOMMANDS.contains(&subcommand.as_str()) {
                return true;
            }
            subcommand == "branch"
                && rest
                    .iter()
                    .all(|arg| matches!(arg.as_str(), "--list" | "-a" | "-r" | "-v"))
        }
        _ => true,
    }
}

/// Bodies of `$(...)` and backtick substitutions (any nesting), so a command
/// hidden in a double-quoted substitution is still examined.
fn substitution_bodies(command: &str) -> Vec<String> {
    let chars: Vec<char> = command.chars().collect();
    let mut bodies = Vec::new();
    let mut in_single = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\'' {
            in_single = !in_single;
        } else if !in_single && c == '\\' {
            i += 1;
        } else if !in_single && c == '$' && chars.get(i + 1) == Some(&'(') {
            let mut depth = 1;
            let mut j = i + 2;
            while j < chars.len() && depth > 0 {
                match chars[j] {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    _ => {}
                }
                j += 1;
            }
            let end = if depth == 0 { j - 1 } else { j };
            bodies.push(chars[i + 2..end].iter().collect());
        } else if !in_single && c == '`' {
            if let Some(len) = chars[i + 1..].iter().position(|&x| x == '`') {
                bodies.push(chars[i + 1..i + 1 + len].iter().collect());
                i += len + 1;
            }
        }
        i += 1;
    }
    bodies
}

/// Every command reachable from `command`: each segment, substitution body,
/// `sh -c` script, and the inner command behind any wrapper. Each entry is a
/// non-empty token list whose first token is the executable.
fn expanded_commands(command: &str) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    collect_commands(command, 0, &mut out);
    out
}

const MAX_DEPTH: usize = 4;

fn collect_commands(command: &str, depth: usize, out: &mut Vec<Vec<String>>) {
    if depth > MAX_DEPTH {
        return;
    }
    for segment in scan(command).segments {
        if segment.trim().is_empty() {
            continue;
        }
        let tokens = shell_words::split(&segment)
            .unwrap_or_else(|_| segment.split_whitespace().map(str::to_string).collect());
        push_with_wrappers(strip_redirections(tokens), depth, out);
    }
    for body in substitution_bodies(command) {
        collect_commands(&body, depth + 1, out);
    }
    for invocation in structural_shell_invocations(command) {
        let mut tokens = vec![invocation.program];
        tokens.extend(invocation.arguments);
        push_with_wrappers(tokens, depth, out);
    }
}

fn push_with_wrappers(mut tokens: Vec<String>, depth: usize, out: &mut Vec<Vec<String>>) {
    if depth > MAX_DEPTH || tokens.len() > 64 {
        return;
    }
    while tokens.first().is_some_and(|t| is_env_assignment(t)) {
        tokens.remove(0);
    }
    let Some(first) = tokens.first() else {
        return;
    };
    let program = basename(first).to_string();
    out.push(tokens.clone());
    if WRAPPERS.contains(&program.as_str()) {
        for start in 1..tokens.len() {
            push_with_wrappers(tokens[start..].to_vec(), depth + 1, out);
        }
    }
    if program == "find" {
        for (index, token) in tokens.iter().enumerate() {
            if matches!(token.as_str(), "-exec" | "-execdir" | "-ok" | "-okdir") {
                push_with_wrappers(tokens[index + 1..].to_vec(), depth + 1, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_work_and_delete_targets_are_typed_by_structure() {
        for local in [
            "cargo test > out.log 2>&1",
            "cargo fmt\ncargo test",
            "cd /tmp/aidaemon-test/p && git status",
            "rm -f a.o b.o",
        ] {
            assert!(command_effects(local).local_work_only, "{local}");
        }
        for other in [
            "cargo test < input.txt",
            "git -C /tmp/aidaemon-test/p commit -m x",
            "npm install",
            "sh -c 'cargo test'",
            "cargo test # done",
            "tee out.log",
            // Inline code through a local-work program's own options.
            "git rebase -x 'curl x' HEAD~1",
            "git rebase --exec='curl x' HEAD~1",
            "git rebase --exec 'curl x' HEAD~1",
            "git commit -c HEAD",
            "cargo test --config 'target.x.runner=\"sh\"'",
            "cargo build --config=build.rustc-wrapper=x",
            "make --eval='$x'",
            "make -E x",
            "make -E'$x'",
            "go build --toolexec=x",
            "go build -toolexec x ./...",
            "go test -exec=x ./...",
            "python3 /dev/stdin",
            "python3 -",
            "node /dev/fd/3",
            "npm exec x",
            "pnpm dlx x",
            "yarn dlx x",
        ] {
            assert!(!command_effects(other).local_work_only, "{other}");
        }
        for local in [
            "git rebase main",
            "make -j4 all",
            "go test ./...",
            "node scripts/x.js",
        ] {
            assert!(command_effects(local).local_work_only, "{local}");
        }
        assert_eq!(
            command_effects("rm -rf -- -odd build && find out -delete").delete_targets,
            Some(vec!["-odd".into(), "build".into(), "out".into()])
        );
        for unknown in [
            "rm \"$X\"",
            "git clean -fd",
            "xargs rm",
            "rm `ls`",
            "find -delete",
            "find -L / -delete",
            "rm -rf {a,b}",
            "cd x && rm y",
        ] {
            assert_eq!(command_effects(unknown).delete_targets, None, "{unknown}");
        }
        // Observations still refuse file redirects.
        assert!(!command_effects("ls > out").observation_only);
    }

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
            "date +%Y",
            "tree -a -L 2",
            "uniq -c f",
            "date",
            "/usr/bin/ls",
            "rg -n foo src",
            "sort -u f",
            "cat < /dev/null",
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
            "ls; env -S 'rm x'",
            "ls; env -S'rm x'",
            "ls # <<EOF\nrm x\nEOF",
            "ls\nrm x",
            "cat <<EOF\nhi\nEOF",
            "rg --pre rm x .",
            "rg --pre=rm x .",
            "rg --hostname-bin=rm x",
            "rg -z foo",
            "GIT_EXTERNAL_DIFF='rm -rf x' git diff",
            "env GIT_EXTERNAL_DIFF=rm git diff",
            "FOO=1 ls",
            "env ls",
            "git --config-env=core.fsmonitor=F status",
            "git -c core.pager=x status",
            "git diff --ext-diff",
            "sort --compress-program=rm a",
            "cat </dev/tcp/example.invalid/80",
            "cat < f",
            "date -s 2020-01-01",
            "hostname evil",
            "file -C -m m",
            "sed -n p f",
            "sed 's/a/b/w out' f",
            "sed '1e rm x' f",
            "(rm x)",
            "{ rm x; }",
            "sudo ls",
            "nice ls",
            "timeout 5 ls",
            "/tmp/evil/ls",
            "./ls",
            "echo a #b",
            "echo $'\\'' > out \\'",
            "echo $'a\\'' ; rm -rf x ; echo \\'",
            "echo $'a\\'' ; curl example.invalid ; echo \\'",
            "uniq - out",
            "uniq -c - out",
            "tree -ao out",
            "tree -R",
            "tree --weird",
            "git -C d status",
            "git --git-dir=d status",
            "git --git-dir d status",
            "git --work-tree=d status",
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
            "sudo curl x",
            "nice curl x",
            "timeout 5 curl x",
            "xargs curl",
            "find . -exec curl {} ;",
            "(curl x)",
            "{ curl x; }",
            "echo \"$(curl x)\"",
            "cat </dev/tcp/example.invalid/80",
            "pip3 -q install x",
            "npm --prefix d install x",
            "cargo +nightly install x",
            "yarn",
            "npm ci",
            "npx foo",
            "pnpm dlx foo",
            "bunx foo",
            "uvx foo",
            "uv pip install x",
            "pip download x",
            "brew install x",
            "go get x",
            "docker pull x",
            "gem install x",
            "apt-get install x",
            "svn up",
            "hg pull",
            "socat a b",
            "openssl s_client -connect h:1",
            "ping h",
            "dig h",
            "rclone ls r:",
            "aws s3 ls",
            "kubectl get pods",
            "git remote update",
            "git archive --remote=u HEAD",
            "git push",
            "npm publish",
            "cargo publish",
            "FOO=1 curl x",
        ] {
            assert!(fetches_network(fetching), "{fetching}");
        }
        for local in [
            "git status",
            "npm test",
            "cargo build",
            "echo curl",
            "ls",
            "yarn test",
        ] {
            assert!(!fetches_network(local), "{local}");
        }
    }

    #[test]
    fn ansi_c_quoting_over_reports_effects() {
        for hidden in [
            "echo $'a\\'' ; rm -rf x ; echo \\'",
            "echo $'a\\'' ; curl example.invalid ; echo \\'",
        ] {
            assert!(command_effects(hidden).deletes, "{hidden}");
            assert!(fetches_network(hidden), "{hidden}");
        }
    }

    #[test]
    fn git_config_paths_are_structural() {
        for path in [
            "/repo/.git/config",
            ".git/hooks/pre-commit",
            "a/.git",
            "/repo/sub/.gitattributes",
            ".gitmodules",
            "C:\\repo\\.git\\config",
        ] {
            assert!(is_git_config_path(path), "{path}");
        }
        for path in [
            "/repo/.github/x.yml",
            "/repo/src/git.rs",
            "a/.gitignore",
            "/repo/gitattributes",
        ] {
            assert!(!is_git_config_path(path), "{path}");
        }
    }

    #[test]
    fn git_config_path_detection_is_case_insensitive_and_global() {
        for path in [
            ".GIT/config",
            "repo/.Git/config",
            "repo/.GITATTRIBUTES",
            "repo/.GITMODULES",
            "repo/.git./config",
            "repo/.git /config",
            "./repo/x/../.git/config",
            "~/.gitconfig",
            "$HOME/.gitconfig",
            "${HOME}/.gitconfig",
            "~/.config/git/attributes",
            "~/.config/git/config",
            "~/.config/../.config/git/ignore",
            "/etc/gitconfig",
            "/etc/./gitconfig",
            "/srv/user/.GITCONFIG",
        ] {
            assert!(is_git_config_path(path), "{path}");
        }
        if let Some(home) = dirs::home_dir() {
            let absolute = home.join(".config").join("git").join("attributes");
            assert!(is_git_config_path(&absolute.to_string_lossy()));
        }
        // `.gitignore` and `.github` are ordinary project files.
        for path in [
            "repo/src/git.rs",
            "repo/.github/workflows/x.yml",
            "repo/.gitignore",
            "repo/.GITHUB/x.yml",
            "/etc/gitconfig.d/x",
            "repo/config/git.toml",
        ] {
            assert!(!is_git_config_path(path), "{path}");
        }
    }

    #[test]
    fn command_words_naming_git_config_are_detected() {
        for command in [
            "cp x ~/.gitconfig",
            "echo x > ~/.gitconfig",
            "echo x >>.git/config",
            "git config --file=$HOME/.gitconfig a.b c",
            "sh -c 'cp x .GIT/hooks/pre-commit'",
            "echo \"$(cp x /etc/gitconfig)\"",
            "cp x \"repo/.gitattributes\"",
        ] {
            assert!(command_names_git_config_path(command), "{command}");
        }
        for command in ["cp x y", "cat .gitignore", "ls .github", "git status"] {
            assert!(!command_names_git_config_path(command), "{command}");
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
            "sudo rm x",
            "nice rm x",
            "timeout 5 rm x",
            "(rm x)",
            "{ rm x; }",
            "exec rm x",
            "echo \"$(rm x)\"",
            "FOO=1 rm x",
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
