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
use aidaemon_core::{CallEffects, EffectClass, EffectKey};

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
    // Input-redirect targets are read like any operand.
    for line in command.replace("\\\n", " ").split('\n') {
        for target in scan(line).input_targets {
            if unreadable(&target) || target.contains(['"', '\'']) {
                return None;
            }
            if !operands.contains(&target) {
                operands.push(target);
            }
        }
    }
    for tokens in &commands {
        // `cd` moves where relative operands resolve; `xargs` and `parallel`
        // take their operands from stdin, which no argument names.
        if matches!(
            basename(&tokens[0]),
            "cd" | "pushd" | "popd" | "xargs" | "parallel"
        ) {
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
        // Local work never reads stdin from a file: an interpreter fed a
        // file this way runs code no argument names.
        if scanned.unsafe_syntax || !scanned.input_targets.is_empty() {
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
    /// Targets of plain input redirects (`< file`, `<file`, `0< file`),
    /// as written. A here-doc, here-string or process substitution is
    /// unsafe syntax instead.
    input_targets: Vec<String>,
    /// Targets of output redirects into files, as written (not
    /// descriptor duplications or the null and standard-stream devices).
    output_targets: Vec<String>,
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
    let mut input_targets = Vec::new();
    let mut output_targets = Vec::new();
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
                let target = redirect_target(&chars, i + 1);
                if matches!(next, Some('<') | Some('(') | Some('>')) || target.is_empty() {
                    unsafe_syntax = true;
                } else if target != "/dev/null" {
                    input_targets.push(target);
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
                } else if !dup {
                    let target = redirect_target(&chars, j);
                    if target != "/dev/null" {
                        writes_file = true;
                        if !is_standard_stream(&target) {
                            output_targets.push(target);
                        }
                    }
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
        input_targets,
        output_targets,
    }
}

/// A redirect target that is a process's own stream, not a file.
fn is_standard_stream(target: &str) -> bool {
    matches!(target, "/dev/stdout" | "/dev/stderr")
        || target
            .strip_prefix("/dev/fd/")
            .is_some_and(|fd| !fd.is_empty() && fd.chars().all(|c| c.is_ascii_digit()))
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
    is_observation(command, false)
}

/// An observation whose observers may also read plain input redirects
/// (`cat < file`). For judging where output came from, with the redirect
/// targets checked as operands (`path_operands`); the authority view
/// (`CommandEffects::observation_only`) stays the stricter shape.
pub(crate) fn is_observation_reading_input(command: &str) -> bool {
    is_observation(command, true)
}

fn is_observation(command: &str, allow_input_redirects: bool) -> bool {
    let scanned = scan(command);
    if (!allow_input_redirects && !scanned.input_targets.is_empty())
        || scanned.unsafe_syntax
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

// ---------------------------------------------------------------------------
// Typed call effects
// ---------------------------------------------------------------------------
//
// What a call does, for the kernel's decline memory and the approval and
// egress policies. Keys are built only with the kernel constructors. A
// command is `complete` only when every segment is fully understood: a known
// observation, or a program from the tables below in a shape whose every
// option is known. Anything else (scripts, interpreters, wrappers, unknown
// options, shell expansion) leaves the declared keys as a lower bound.
// Configuration files a program reads on its own (`~/.curlrc`,
// `~/.ssh/config` host aliases, git `insteadOf` rewrites) are not consulted.

/// An egress effect on `host`, with IPv6 brackets stripped so `[::1]` and
/// `::1` name one host. `None` for an empty or unreadable host.
pub(crate) fn egress_effect(scheme: &str, host: &str, port: Option<u16>) -> Option<EffectKey> {
    let host = host.trim();
    let host = host
        .strip_prefix('[')
        .and_then(|inner| inner.strip_suffix(']'))
        .unwrap_or(host);
    if host.is_empty()
        || host.contains(|c: char| c.is_whitespace() || matches!(c, '/' | '@' | '$' | '`'))
    {
        return None;
    }
    Some(EffectKey::egress(scheme, host, port))
}

/// Default ports the kernel's egress constructor does not know.
fn scheme_default_port(scheme: &str) -> Option<u16> {
    match scheme {
        "git" => Some(9418),
        "rsync" => Some(873),
        "ssh" | "git+ssh" | "ssh+git" | "sftp" | "scp" => Some(22),
        _ => None,
    }
}

/// The egress effect of a URL, read as `default_scheme` when it names no
/// scheme. `None` when it has no network host or cannot be read
/// structurally (shell expansion, unparsable).
pub(crate) fn url_effect(raw: &str, default_scheme: &str) -> Option<EffectKey> {
    let raw = raw.trim();
    if raw.is_empty() || raw.contains(['$', '`']) {
        return None;
    }
    let text = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("{default_scheme}://{raw}")
    };
    let url = reqwest::Url::parse(&text).ok()?;
    url_host_effect(&url)
}

/// The egress effect of a parsed URL: its host and its explicit port, else
/// the scheme's default.
pub(crate) fn url_host_effect(url: &reqwest::Url) -> Option<EffectKey> {
    let scheme = url.scheme();
    let port = url
        .port()
        .or_else(|| scheme_default_port(scheme))
        .or_else(|| url.port_or_known_default());
    egress_effect(scheme, url.host_str()?, port)
}

/// A path effect on `raw`, absolute and normalized: `~/` expands to the home
/// directory and a relative path resolves against `cwd` (an absolute
/// directory). On disk the nearest existing ancestor is canonicalized, so an
/// existing path is declared by its canonical form (symlinks resolved, and on
/// macOS case and `/tmp` as `/private/tmp`) and a new one below it agrees with
/// it. A `Delete` keeps its last component: removing a symlink removes the
/// link. `None` when the path cannot be read structurally (shell expansion)
/// or is relative with no known working directory.
pub(crate) fn path_effect(class: EffectClass, raw: &str, cwd: Option<&str>) -> Option<EffectKey> {
    let raw = raw.trim();
    if raw.is_empty() || raw.contains(['$', '`', '*', '?', '[', '{', '}', '\\', '\0']) {
        return None;
    }
    let expanded = if raw == "~" || raw.starts_with("~/") {
        format!("{}{}", dirs::home_dir()?.to_str()?, &raw[1..])
    } else if raw.starts_with('~') {
        return None;
    } else {
        raw.to_string()
    };
    let absolute = if expanded.starts_with('/') {
        expanded
    } else {
        let cwd = cwd.map(str::trim).filter(|cwd| cwd.starts_with('/'))?;
        format!("{}/{expanded}", cwd.trim_end_matches('/'))
    };
    let lexical = EffectKey::path(class, &absolute)?;
    Some(resolved_on_disk(class, lexical))
}

fn resolved_on_disk(class: EffectClass, lexical: EffectKey) -> EffectKey {
    let path = std::path::Path::new(&lexical.target);
    let (base, leaf) = match (class, path.parent(), path.file_name()) {
        (EffectClass::Delete, Some(parent), Some(name)) => (parent, Some(name)),
        _ => (path, None),
    };
    for ancestor in base.ancestors() {
        let Ok(mut real) = std::fs::canonicalize(ancestor) else {
            continue;
        };
        if let Ok(rest) = base.strip_prefix(ancestor) {
            if !rest.as_os_str().is_empty() {
                real.push(rest);
            }
        }
        if let Some(leaf) = leaf {
            real.push(leaf);
        }
        return real
            .to_str()
            .and_then(|real| EffectKey::path(class, real))
            .unwrap_or(lexical);
    }
    lexical
}

/// Whether the active execution backend is this machine, so paths resolve on
/// the local disk.
pub(crate) fn execution_backend_is_local() -> bool {
    crate::execution::active_execution_backend().kind() == crate::execution::BackendKind::Local
}

/// Effects on a remote (SSH, Docker) workspace: its paths cannot be
/// resolved on this disk, so path keys are dropped and the effects are
/// incomplete. Egress keys stay.
pub(crate) fn without_local_paths(mut effects: CallEffects) -> CallEffects {
    effects.keys.retain(|key| {
        !matches!(
            key.class,
            EffectClass::Write | EffectClass::Delete | EffectClass::Configuration
        )
    });
    effects.complete = false;
    effects
}

/// The effects of `write_file`/`edit_file`: a write of the path argument
/// (or its aliases). A relative path, or any path on a remote workspace, is
/// undeterminable here.
pub(crate) fn file_tool_write_effects(arguments: &str) -> CallEffects {
    if !execution_backend_is_local() {
        return CallEffects::unknown();
    }
    let path = serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|args| {
            ["path", "file_path", "file", "filename"]
                .iter()
                .find_map(|key| {
                    args.get(*key)
                        .and_then(|value| value.as_str())
                        .map(str::to_string)
                })
        });
    match path.and_then(|path| path_effect(EffectClass::Write, &path, None)) {
        Some(key) => CallEffects {
            keys: vec![key],
            complete: true,
        },
        None => CallEffects::unknown(),
    }
}

/// The egress targets (`host:port`) of a shell command, or `None` when any
/// segment's network reach cannot be determined (an opaque program, a
/// wrapper, an unknown option, a remote named only by configuration).
// Produced for the terminal approval and egress rules that consume it next.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn egress_hosts(command: &str) -> Option<Vec<String>> {
    let analysis = analyze_shell(command, None);
    analysis.egress_complete.then(|| {
        analysis
            .keys
            .into_iter()
            .filter(|key| key.class == EffectClass::Egress)
            .map(|key| key.target)
            .collect()
    })
}

/// The typed effects of a shell command run in `cwd` (an absolute directory,
/// if known): egress hosts from the URL operands of network programs,
/// deletes, and writes from output redirects, `tee`, `cp`/`mv`
/// destinations, `touch` and `mkdir`. Complete only when every segment is
/// determinable.
pub(crate) fn shell_call_effects(command: &str, cwd: Option<&str>) -> CallEffects {
    let analysis = analyze_shell(command, cwd);
    CallEffects {
        keys: analysis.keys,
        complete: analysis.complete,
    }
}

struct ShellAnalysis {
    keys: Vec<EffectKey>,
    /// Every effect is in `keys`.
    complete: bool,
    /// Every egress effect is in `keys` (a local target may still be
    /// unresolved).
    egress_complete: bool,
}

impl ShellAnalysis {
    fn add(&mut self, key: EffectKey) {
        if !self.keys.contains(&key) {
            self.keys.push(key);
        }
    }

    /// A segment that may do anything, including reach the network.
    fn opaque(&mut self) {
        self.complete = false;
        self.egress_complete = false;
    }

    /// A local target that cannot be named.
    fn unresolved(&mut self) {
        self.complete = false;
    }

    fn path(&mut self, class: EffectClass, raw: &str, cwd: Option<&str>) {
        // A word glued to a redirect (`a>b`) is not the operand it looks like.
        if raw.contains(['<', '>']) {
            self.unresolved();
            return;
        }
        match path_effect(class, raw, cwd) {
            Some(key) => self.add(key),
            None => self.unresolved(),
        }
    }

    fn egress(&mut self, key: Option<EffectKey>) {
        match key {
            Some(key) => self.add(key),
            None => self.opaque(),
        }
    }
}

fn analyze_shell(command: &str, cwd: Option<&str>) -> ShellAnalysis {
    let mut out = ShellAnalysis {
        keys: Vec::new(),
        complete: true,
        egress_complete: true,
    };
    if command.contains("/dev/tcp/") || command.contains("/dev/udp/") || command.contains("$'") {
        out.opaque();
    }
    // A `cd` moves where later relative paths resolve.
    let mut cwd = cwd.map(str::to_string);
    for line in command.replace("\\\n", " ").split('\n') {
        let scanned = scan(line);
        // Substitutions, groupings, heredocs and comments hide commands.
        if scanned.unsafe_syntax {
            out.opaque();
        }
        for target in &scanned.output_targets {
            if target.contains(['"', '\'']) {
                out.unresolved();
            } else {
                out.path(EffectClass::Write, target, cwd.as_deref());
            }
        }
        for segment in &scanned.segments {
            if segment.trim().is_empty() {
                continue;
            }
            let Ok(tokens) = shell_words::split(segment) else {
                out.opaque();
                continue;
            };
            segment_effects(&strip_redirections(tokens), &mut cwd, &mut out);
        }
    }
    out
}

fn segment_effects(tokens: &[String], cwd: &mut Option<String>, out: &mut ShellAnalysis) {
    let Some(program) = tokens.first() else {
        return;
    };
    let path_ok = !program.contains('/')
        || program
            .rsplit_once('/')
            .is_some_and(|(dir, _)| SYSTEM_BIN_DIRS.contains(&dir));
    if is_env_assignment(program) || !path_ok {
        out.opaque();
        return;
    }
    let name = basename(program);
    let args = &tokens[1..];
    if OBSERVATION_PROGRAMS.contains(&name) && observation_args_ok(name, args) {
        return;
    }
    let cwd_now = cwd.clone();
    let here = cwd_now.as_deref();
    match name {
        "cd" | "pushd" | "popd" => *cwd = None,
        "rm" | "rmdir" | "unlink" => deleted(tokens, here, out),
        "find"
            if args.iter().any(|arg| arg == "-delete")
                && !args
                    .iter()
                    .any(|arg| arg != "-delete" && FIND_MUTATING_FLAGS.contains(&arg.as_str())) =>
        {
            deleted(tokens, here, out)
        }
        "cp" | "mv" => copied(name, args, here, out),
        "tee" | "touch" | "mkdir" => created(name, args, here, out),
        "curl" => curl(args, here, out),
        "wget" => wget(args, here, out),
        "http" | "https" | "xh" | "xhs" => httpie(name, args, here, out),
        "nc" | "ncat" | "netcat" => netcat(args, out),
        "ssh" => ssh(args, here, out),
        "scp" => scp(args, here, out),
        "rsync" => rsync(args, here, out),
        "git" => git(args, here, out),
        _ => out.opaque(),
    }
}

fn deleted(tokens: &[String], cwd: Option<&str>, out: &mut ShellAnalysis) {
    match delete_targets(&shell_words::join(tokens)) {
        Some(targets) => {
            for target in targets {
                out.path(EffectClass::Delete, &target, cwd);
            }
        }
        None => out.unresolved(),
    }
}

/// What an option's value means for the call's effects.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OptionValue {
    /// Does not change what the call touches (a header, a timeout).
    Ignore,
    /// A URL the program contacts.
    Url,
    /// A proxy the program connects through.
    Proxy,
    /// A file the program writes (`-` is stdout).
    Write,
    /// The file a download is saved to (`-` is stdout).
    Document,
    /// The port of the destination.
    Port,
    /// The directory `cp`/`mv` write into (`-t`).
    TargetDir,
}

/// The options a program is understood with. An option not listed makes
/// the segment undeterminable.
struct OptionSpec {
    /// Short options without a value; they may be clustered.
    flags: &'static str,
    /// Short options with a value, attached or in the next word.
    valued: &'static [(char, OptionValue)],
    long_flags: &'static [&'static str],
    long_valued: &'static [(&'static str, OptionValue)],
    /// Long options accept `--name=value`.
    long_equals: bool,
    /// Whole words without a value (wget's `-nv`).
    exact: &'static [&'static str],
}

#[derive(Default)]
struct ParsedOptions {
    positionals: Vec<String>,
    values: Vec<(OptionValue, String)>,
    long_flags: Vec<String>,
}

fn parse_options(spec: &OptionSpec, args: &[String]) -> Option<ParsedOptions> {
    let mut parsed = ParsedOptions::default();
    let mut index = 0;
    let mut options_done = false;
    while index < args.len() {
        let arg = &args[index];
        index += 1;
        if options_done || arg == "-" || !arg.starts_with('-') {
            parsed.positionals.push(arg.clone());
            continue;
        }
        if arg == "--" {
            options_done = true;
            continue;
        }
        if spec.exact.contains(&arg.as_str()) {
            continue;
        }
        if let Some(long) = arg.strip_prefix("--") {
            if let Some((name, value)) = long.split_once('=') {
                if !spec.long_equals {
                    return None;
                }
                let role = lookup(spec.long_valued, name)?;
                parsed.values.push((role, value.to_string()));
            } else if spec.long_flags.contains(&long) {
                parsed.long_flags.push(long.to_string());
            } else {
                let role = lookup(spec.long_valued, long)?;
                parsed.values.push((role, args.get(index)?.clone()));
                index += 1;
            }
            continue;
        }
        let cluster: Vec<char> = arg[1..].chars().collect();
        for (position, letter) in cluster.iter().enumerate() {
            if spec.flags.contains(*letter) {
                continue;
            }
            let role = spec
                .valued
                .iter()
                .find(|(short, _)| short == letter)
                .map(|(_, role)| *role)?;
            let rest: String = cluster[position + 1..].iter().collect();
            let value = if rest.is_empty() {
                index += 1;
                args.get(index - 1)?.clone()
            } else {
                rest
            };
            parsed.values.push((role, value));
            break;
        }
    }
    Some(parsed)
}

fn lookup(table: &[(&str, OptionValue)], name: &str) -> Option<OptionValue> {
    table
        .iter()
        .find(|(long, _)| *long == name)
        .map(|(_, role)| *role)
}

/// Apply the option values every program shares: written files, proxies
/// and URLs.
fn apply_values(
    parsed: &ParsedOptions,
    default_scheme: &str,
    cwd: Option<&str>,
    out: &mut ShellAnalysis,
) {
    for (role, value) in &parsed.values {
        match role {
            OptionValue::Write | OptionValue::Document if value != "-" => {
                out.path(EffectClass::Write, value, cwd)
            }
            OptionValue::Url => out.egress(url_effect(value, default_scheme)),
            OptionValue::Proxy => out.egress(url_effect(value, "http")),
            _ => {}
        }
    }
}

const CURL: OptionSpec = OptionSpec {
    flags: "sSLfiIkvGNqjgn0123456#Z",
    valued: &[
        ('X', OptionValue::Ignore),
        ('d', OptionValue::Ignore),
        ('H', OptionValue::Ignore),
        ('u', OptionValue::Ignore),
        ('A', OptionValue::Ignore),
        ('e', OptionValue::Ignore),
        ('F', OptionValue::Ignore),
        ('m', OptionValue::Ignore),
        ('T', OptionValue::Ignore),
        ('b', OptionValue::Ignore),
        ('w', OptionValue::Ignore),
        ('r', OptionValue::Ignore),
        ('C', OptionValue::Ignore),
        ('Y', OptionValue::Ignore),
        ('y', OptionValue::Ignore),
        ('z', OptionValue::Ignore),
        ('E', OptionValue::Ignore),
        ('U', OptionValue::Ignore),
        ('x', OptionValue::Proxy),
        ('o', OptionValue::Write),
        ('D', OptionValue::Write),
        ('c', OptionValue::Write),
    ],
    long_flags: &[
        "silent",
        "show-error",
        "location",
        "location-trusted",
        "fail",
        "fail-with-body",
        "fail-early",
        "include",
        "head",
        "insecure",
        "verbose",
        "get",
        "compressed",
        "http0.9",
        "http1.0",
        "http1.1",
        "http2",
        "http2-prior-knowledge",
        "http3",
        "http3-only",
        "no-buffer",
        "globoff",
        "ipv4",
        "ipv6",
        "no-progress-meter",
        "progress-bar",
        "netrc",
        "netrc-optional",
        "raw",
        "tr-encoding",
        "no-keepalive",
        "ssl-reqd",
        "tcp-nodelay",
        "path-as-is",
        "retry-all-errors",
        "retry-connrefused",
        "anyauth",
        "basic",
        "digest",
        "ntlm",
        "negotiate",
        "disable",
        "junk-session-cookies",
        "parallel",
        "tlsv1.2",
        "tlsv1.3",
        "styled-output",
        "no-styled-output",
    ],
    long_valued: &[
        ("url", OptionValue::Url),
        ("request", OptionValue::Ignore),
        ("data", OptionValue::Ignore),
        ("data-raw", OptionValue::Ignore),
        ("data-binary", OptionValue::Ignore),
        ("data-urlencode", OptionValue::Ignore),
        ("data-ascii", OptionValue::Ignore),
        ("json", OptionValue::Ignore),
        ("header", OptionValue::Ignore),
        ("user", OptionValue::Ignore),
        ("user-agent", OptionValue::Ignore),
        ("referer", OptionValue::Ignore),
        ("form", OptionValue::Ignore),
        ("form-string", OptionValue::Ignore),
        ("max-time", OptionValue::Ignore),
        ("connect-timeout", OptionValue::Ignore),
        ("upload-file", OptionValue::Ignore),
        ("cookie", OptionValue::Ignore),
        ("write-out", OptionValue::Ignore),
        ("range", OptionValue::Ignore),
        ("retry", OptionValue::Ignore),
        ("retry-delay", OptionValue::Ignore),
        ("retry-max-time", OptionValue::Ignore),
        ("cacert", OptionValue::Ignore),
        ("capath", OptionValue::Ignore),
        ("cert", OptionValue::Ignore),
        ("cert-type", OptionValue::Ignore),
        ("key", OptionValue::Ignore),
        ("key-type", OptionValue::Ignore),
        ("oauth2-bearer", OptionValue::Ignore),
        ("max-redirs", OptionValue::Ignore),
        ("limit-rate", OptionValue::Ignore),
        ("max-filesize", OptionValue::Ignore),
        ("continue-at", OptionValue::Ignore),
        ("time-cond", OptionValue::Ignore),
        ("proxy-user", OptionValue::Ignore),
        ("proxy", OptionValue::Proxy),
        ("preproxy", OptionValue::Proxy),
        ("output", OptionValue::Write),
        ("dump-header", OptionValue::Write),
        ("cookie-jar", OptionValue::Write),
        ("trace", OptionValue::Write),
        ("trace-ascii", OptionValue::Write),
        ("libcurl", OptionValue::Write),
        ("stderr", OptionValue::Write),
        ("etag-save", OptionValue::Write),
    ],
    long_equals: false,
    exact: &[],
};

fn curl(args: &[String], cwd: Option<&str>, out: &mut ShellAnalysis) {
    let Some(parsed) = parse_options(&CURL, args) else {
        out.opaque();
        return;
    };
    // `--write-out '%output{file}'` writes a file the option names inline.
    if parsed
        .values
        .iter()
        .any(|(_, value)| value.contains("%output{"))
    {
        out.unresolved();
    }
    apply_values(&parsed, "http", cwd, out);
    for url in &parsed.positionals {
        out.egress(url_effect(url, "http"));
    }
}

const WGET: OptionSpec = OptionSpec {
    flags: "qvSc",
    valued: &[
        ('O', OptionValue::Document),
        ('o', OptionValue::Write),
        ('a', OptionValue::Write),
        ('T', OptionValue::Ignore),
        ('t', OptionValue::Ignore),
        ('U', OptionValue::Ignore),
    ],
    long_flags: &[
        "quiet",
        "verbose",
        "no-verbose",
        "server-response",
        "spider",
        "continue",
        "no-check-certificate",
        "no-clobber",
        "inet4-only",
        "inet6-only",
        "no-cache",
        "no-cookies",
        "content-on-error",
        "ignore-length",
        "auth-no-challenge",
        "no-proxy",
    ],
    long_valued: &[
        ("output-document", OptionValue::Document),
        ("output-file", OptionValue::Write),
        ("append-output", OptionValue::Write),
        ("save-cookies", OptionValue::Write),
        ("timeout", OptionValue::Ignore),
        ("tries", OptionValue::Ignore),
        ("header", OptionValue::Ignore),
        ("method", OptionValue::Ignore),
        ("body-data", OptionValue::Ignore),
        ("body-file", OptionValue::Ignore),
        ("post-data", OptionValue::Ignore),
        ("post-file", OptionValue::Ignore),
        ("user-agent", OptionValue::Ignore),
        ("user", OptionValue::Ignore),
        ("password", OptionValue::Ignore),
        ("http-user", OptionValue::Ignore),
        ("http-password", OptionValue::Ignore),
        ("referer", OptionValue::Ignore),
        ("wait", OptionValue::Ignore),
        ("limit-rate", OptionValue::Ignore),
        ("read-timeout", OptionValue::Ignore),
        ("connect-timeout", OptionValue::Ignore),
        ("dns-timeout", OptionValue::Ignore),
        ("max-redirect", OptionValue::Ignore),
        ("progress", OptionValue::Ignore),
        ("ca-certificate", OptionValue::Ignore),
        ("certificate", OptionValue::Ignore),
        ("private-key", OptionValue::Ignore),
        ("load-cookies", OptionValue::Ignore),
    ],
    long_equals: true,
    exact: &["-nv", "-nc", "-nd", "-nH"],
};

fn wget(args: &[String], cwd: Option<&str>, out: &mut ShellAnalysis) {
    let Some(parsed) = parse_options(&WGET, args) else {
        out.opaque();
        return;
    };
    apply_values(&parsed, "http", cwd, out);
    for url in &parsed.positionals {
        out.egress(url_effect(url, "http"));
    }
    // Without `-O` wget saves under a name taken from the URL or response.
    let names_output = parsed
        .values
        .iter()
        .any(|(role, _)| *role == OptionValue::Document);
    let spider = parsed.long_flags.iter().any(|flag| flag == "spider");
    if !names_output && !spider {
        out.unresolved();
    }
}

const HTTPIE: OptionSpec = OptionSpec {
    flags: "vhbfjFIq",
    valued: &[
        ('p', OptionValue::Ignore),
        ('s', OptionValue::Ignore),
        ('a', OptionValue::Ignore),
        ('A', OptionValue::Ignore),
        ('o', OptionValue::Write),
    ],
    long_flags: &[
        "json",
        "form",
        "multipart",
        "verbose",
        "headers",
        "body",
        "follow",
        "check-status",
        "ignore-stdin",
        "quiet",
        "offline",
        "all",
    ],
    long_valued: &[
        ("print", OptionValue::Ignore),
        ("pretty", OptionValue::Ignore),
        ("style", OptionValue::Ignore),
        ("auth", OptionValue::Ignore),
        ("auth-type", OptionValue::Ignore),
        ("timeout", OptionValue::Ignore),
        ("max-redirects", OptionValue::Ignore),
        ("verify", OptionValue::Ignore),
        ("cert", OptionValue::Ignore),
        ("cert-key", OptionValue::Ignore),
        ("output", OptionValue::Write),
    ],
    long_equals: true,
    exact: &[],
};

/// HTTPie (`http`, `https`) and xh: `[METHOD] URL [items]`, where a URL
/// may be `:port/path` for localhost and names no scheme by default.
fn httpie(program: &str, args: &[String], cwd: Option<&str>, out: &mut ShellAnalysis) {
    let Some(parsed) = parse_options(&HTTPIE, args) else {
        out.opaque();
        return;
    };
    apply_values(&parsed, "http", cwd, out);
    let mut positionals = parsed.positionals.iter();
    let mut url = positionals.next();
    if url.is_some_and(|word| !word.is_empty() && word.chars().all(|c| c.is_ascii_uppercase())) {
        url = positionals.next();
    }
    let Some(url) = url else {
        out.opaque();
        return;
    };
    let scheme = if matches!(program, "https" | "xhs") {
        "https"
    } else {
        "http"
    };
    let url = match url.strip_prefix(':') {
        Some(rest) => format!("localhost:{rest}"),
        None => url.clone(),
    };
    out.egress(url_effect(&url, scheme));
}

const NETCAT: OptionSpec = OptionSpec {
    flags: "vznu46CDdN",
    valued: &[
        ('w', OptionValue::Ignore),
        ('p', OptionValue::Ignore),
        ('s', OptionValue::Ignore),
        ('q', OptionValue::Ignore),
        ('i', OptionValue::Ignore),
    ],
    long_flags: &[],
    long_valued: &[],
    long_equals: false,
    exact: &[],
};

/// `nc host port`. Listening, proxies and `-e` are not understood.
fn netcat(args: &[String], out: &mut ShellAnalysis) {
    let target = parse_options(&NETCAT, args).and_then(|parsed| match &parsed.positionals[..] {
        [host, port] => egress_effect("", host, Some(port.parse::<u16>().ok()?)),
        _ => None,
    });
    out.egress(target);
}

const SSH: OptionSpec = OptionSpec {
    flags: "46AaCfGgKkMNnqsTtVvXxYy",
    valued: &[
        ('p', OptionValue::Port),
        ('i', OptionValue::Ignore),
        ('l', OptionValue::Ignore),
        ('L', OptionValue::Ignore),
        ('R', OptionValue::Ignore),
        ('D', OptionValue::Ignore),
        ('c', OptionValue::Ignore),
        ('m', OptionValue::Ignore),
        ('b', OptionValue::Ignore),
        ('e', OptionValue::Ignore),
        ('E', OptionValue::Write),
    ],
    long_flags: &[],
    long_valued: &[],
    long_equals: false,
    exact: &[],
};

fn port_option(parsed: &ParsedOptions) -> Result<Option<u16>, ()> {
    match parsed
        .values
        .iter()
        .rev()
        .find(|(role, _)| *role == OptionValue::Port)
    {
        Some((_, port)) => port.parse::<u16>().map(Some).map_err(|_| ()),
        None => Ok(None),
    }
}

/// `user@host` or `host` (brackets allowed for IPv6) as an ssh egress.
fn ssh_host_effect(destination: &str, port: Option<u16>) -> Option<EffectKey> {
    let host = destination
        .rsplit_once('@')
        .map_or(destination, |(_, host)| host);
    egress_effect("ssh", host, port)
}

/// `ssh [options] destination [command]`: the remote command runs on the
/// destination, so only the destination is an effect here.
fn ssh(args: &[String], cwd: Option<&str>, out: &mut ShellAnalysis) {
    let Some(parsed) = parse_options(&SSH, args) else {
        out.opaque();
        return;
    };
    apply_values(&parsed, "ssh", cwd, out);
    let Ok(port) = port_option(&parsed) else {
        out.opaque();
        return;
    };
    let key = parsed.positionals.first().and_then(|destination| {
        if destination.contains("://") {
            let url = reqwest::Url::parse(destination).ok()?;
            if url.scheme() != "ssh" {
                return None;
            }
            egress_effect("ssh", url.host_str()?, port.or(url.port()))
        } else {
            ssh_host_effect(destination, port)
        }
    });
    out.egress(key);
}

/// A remote operand of scp or rsync (`[user@]host:path`, `host::module`,
/// `scheme://`), as an egress on `scheme`. `Ok(None)` for a local path,
/// `Err` when it cannot be read.
fn remote_operand(operand: &str, scheme: &str, port: Option<u16>) -> Result<Option<EffectKey>, ()> {
    if operand.contains("://") {
        return url_effect(operand, scheme).map(Some).ok_or(());
    }
    if operand.starts_with(['/', '.', '~']) {
        return Ok(None);
    }
    let (user_host, rest) = match operand.find('[') {
        // `[user@][v6]:path`
        Some(open) => {
            let close = operand.find(']').ok_or(())?;
            if close < open || operand.as_bytes().get(close + 1) != Some(&b':') {
                return Err(());
            }
            (&operand[..=close], &operand[close + 1..])
        }
        None => match operand.split_once(':') {
            Some((before, _)) if !before.contains('/') => (before, &operand[before.len()..]),
            _ => return Ok(None),
        },
    };
    if user_host.is_empty() {
        return Err(());
    }
    // `host::module` is the rsync daemon.
    if rest.starts_with("::") && scheme == "rsync" {
        let host = user_host
            .rsplit_once('@')
            .map_or(user_host, |(_, host)| host);
        return egress_effect("rsync", host, port.or(Some(873)))
            .map(Some)
            .ok_or(());
    }
    ssh_host_effect(user_host, port).map(Some).ok_or(())
}

/// Sources and destination of scp/rsync: remote operands are egress, a
/// local destination is written. `Some(true)` when the destination is
/// local; `None` when the operands cannot be read.
fn transfer(
    positionals: &[String],
    scheme: &str,
    port: Option<u16>,
    cwd: Option<&str>,
    out: &mut ShellAnalysis,
) -> Option<bool> {
    let Some((destination, sources)) = positionals.split_last().filter(|(_, s)| !s.is_empty())
    else {
        out.opaque();
        return None;
    };
    for source in sources {
        match remote_operand(source, scheme, port) {
            Ok(Some(key)) => out.add(key),
            Ok(None) => {}
            Err(()) => out.opaque(),
        }
    }
    match remote_operand(destination, scheme, port) {
        Ok(Some(key)) => {
            out.add(key);
            Some(false)
        }
        Ok(None) => {
            out.path(EffectClass::Write, destination, cwd);
            Some(true)
        }
        Err(()) => {
            out.opaque();
            None
        }
    }
}

const SCP: OptionSpec = OptionSpec {
    flags: "346BCpqrTvA",
    valued: &[
        ('P', OptionValue::Port),
        ('i', OptionValue::Ignore),
        ('l', OptionValue::Ignore),
        ('c', OptionValue::Ignore),
    ],
    long_flags: &[],
    long_valued: &[],
    long_equals: false,
    exact: &[],
};

fn scp(args: &[String], cwd: Option<&str>, out: &mut ShellAnalysis) {
    let Some(parsed) = parse_options(&SCP, args) else {
        out.opaque();
        return;
    };
    let Ok(port) = port_option(&parsed) else {
        out.opaque();
        return;
    };
    transfer(&parsed.positionals, "ssh", port, cwd, out);
}

const RSYNC: OptionSpec = OptionSpec {
    flags: "avzrlptgoDhPnciuqHAXSWxRbIKLkmyO",
    valued: &[('T', OptionValue::Ignore), ('B', OptionValue::Ignore)],
    long_flags: &[
        "archive",
        "verbose",
        "compress",
        "recursive",
        "links",
        "perms",
        "times",
        "group",
        "owner",
        "devices",
        "specials",
        "human-readable",
        "progress",
        "partial",
        "dry-run",
        "checksum",
        "update",
        "quiet",
        "hard-links",
        "acls",
        "xattrs",
        "sparse",
        "whole-file",
        "one-file-system",
        "relative",
        "backup",
        "ignore-existing",
        "inplace",
        "append",
        "stats",
        "itemize-changes",
        "delete",
        "delete-before",
        "delete-during",
        "delete-after",
        "delete-delay",
        "delete-excluded",
        "remove-source-files",
        "prune-empty-dirs",
        "numeric-ids",
        "safe-links",
        "copy-links",
        "keep-dirlinks",
        "no-motd",
        "mkpath",
    ],
    long_valued: &[
        ("exclude", OptionValue::Ignore),
        ("include", OptionValue::Ignore),
        ("exclude-from", OptionValue::Ignore),
        ("include-from", OptionValue::Ignore),
        ("bwlimit", OptionValue::Ignore),
        ("timeout", OptionValue::Ignore),
        ("contimeout", OptionValue::Ignore),
        ("chmod", OptionValue::Ignore),
        ("max-size", OptionValue::Ignore),
        ("min-size", OptionValue::Ignore),
        ("temp-dir", OptionValue::Ignore),
        ("partial-dir", OptionValue::Ignore),
        ("backup-dir", OptionValue::Ignore),
        ("suffix", OptionValue::Ignore),
        ("password-file", OptionValue::Ignore),
        ("port", OptionValue::Port),
        ("log-file", OptionValue::Write),
    ],
    long_equals: true,
    exact: &[],
};

fn rsync(args: &[String], cwd: Option<&str>, out: &mut ShellAnalysis) {
    let Some(parsed) = parse_options(&RSYNC, args) else {
        out.opaque();
        return;
    };
    apply_values(&parsed, "rsync", cwd, out);
    let Ok(port) = port_option(&parsed) else {
        out.opaque();
        return;
    };
    let Some(local_destination) = transfer(&parsed.positionals, "rsync", port, cwd, out) else {
        return;
    };
    let flag = |name: &str| parsed.long_flags.iter().any(|flag| flag.starts_with(name));
    let Some((destination, sources)) = parsed.positionals.split_last() else {
        return;
    };
    // `--delete*` removes files below a local destination (a remote one is
    // already egress).
    if flag("delete") && local_destination {
        out.path(EffectClass::Delete, destination, cwd);
    }
    // `--remove-source-files` deletes what it sent.
    if flag("remove-source-files") {
        for source in sources {
            if matches!(remote_operand(source, "rsync", port), Ok(None)) {
                out.path(EffectClass::Delete, source, cwd);
            }
        }
    }
}

const COPY: OptionSpec = OptionSpec {
    flags: "rRfinpavLPHlsuxXTcbdD",
    valued: &[('t', OptionValue::TargetDir), ('S', OptionValue::Ignore)],
    long_flags: &[
        "archive",
        "recursive",
        "force",
        "interactive",
        "no-clobber",
        "verbose",
        "update",
        "parents",
        "no-target-directory",
        "dereference",
        "no-dereference",
        "one-file-system",
        "link",
        "symbolic-link",
        "preserve",
        "backup",
        "reflink",
        "strip-trailing-slashes",
    ],
    long_valued: &[
        ("target-directory", OptionValue::TargetDir),
        ("preserve", OptionValue::Ignore),
        ("no-preserve", OptionValue::Ignore),
        ("backup", OptionValue::Ignore),
        ("suffix", OptionValue::Ignore),
        ("sparse", OptionValue::Ignore),
        ("reflink", OptionValue::Ignore),
        ("update", OptionValue::Ignore),
    ],
    long_equals: true,
    exact: &[],
};

/// `cp`/`mv`: the destination (the last operand, or `-t DIR`) is written;
/// `mv` also removes its sources.
fn copied(program: &str, args: &[String], cwd: Option<&str>, out: &mut ShellAnalysis) {
    let Some(parsed) = parse_options(&COPY, args) else {
        out.unresolved();
        return;
    };
    let target_dir = parsed
        .values
        .iter()
        .rev()
        .find(|(role, _)| *role == OptionValue::TargetDir)
        .map(|(_, dir)| dir.clone());
    let (destination, sources) = match &target_dir {
        Some(dir) => (dir.as_str(), &parsed.positionals[..]),
        None => match parsed.positionals.split_last() {
            Some((destination, sources)) if !sources.is_empty() => (destination.as_str(), sources),
            _ => {
                out.unresolved();
                return;
            }
        },
    };
    out.path(EffectClass::Write, destination, cwd);
    if program == "mv" {
        for source in sources {
            out.path(EffectClass::Delete, source, cwd);
        }
    }
}

const CREATE: OptionSpec = OptionSpec {
    flags: "aipcmhf",
    valued: &[
        ('t', OptionValue::Ignore),
        ('d', OptionValue::Ignore),
        ('r', OptionValue::Ignore),
    ],
    long_flags: &[
        "append",
        "ignore-interrupts",
        "no-create",
        "no-dereference",
        "parents",
        "verbose",
    ],
    long_valued: &[
        ("output-error", OptionValue::Ignore),
        ("date", OptionValue::Ignore),
        ("reference", OptionValue::Ignore),
        ("time", OptionValue::Ignore),
        ("mode", OptionValue::Ignore),
    ],
    long_equals: true,
    exact: &[],
};

/// `tee`, `touch` and `mkdir` write each operand.
fn created(program: &str, args: &[String], cwd: Option<&str>, out: &mut ShellAnalysis) {
    // `mkdir -m MODE` takes a value where `touch -m` does not.
    let spec = if program == "mkdir" {
        OptionSpec {
            flags: "pv",
            valued: &[('m', OptionValue::Ignore)],
            ..CREATE
        }
    } else {
        CREATE
    };
    match parse_options(&spec, args) {
        Some(parsed) => {
            for operand in &parsed.positionals {
                if operand != "-" {
                    out.path(EffectClass::Write, operand, cwd);
                }
            }
        }
        None => out.unresolved(),
    }
}

const GIT_REMOTE: OptionSpec = OptionSpec {
    flags: "fuqvndptrhls",
    valued: &[
        ('b', OptionValue::Ignore),
        ('o', OptionValue::Ignore),
        ('j', OptionValue::Ignore),
    ],
    long_flags: &[
        "quiet",
        "verbose",
        "progress",
        "force",
        "tags",
        "no-tags",
        "prune",
        "dry-run",
        "atomic",
        "follow-tags",
        "set-upstream",
        "delete",
        "force-with-lease",
        "rebase",
        "no-rebase",
        "ff-only",
        "ff",
        "no-ff",
        "unshallow",
        "heads",
        "refs",
        "no-checkout",
        "bare",
        "mirror",
        "single-branch",
        "no-single-branch",
        "shallow-submodules",
        "no-hardlinks",
        "local",
        "shared",
        "sparse",
    ],
    long_valued: &[
        ("branch", OptionValue::Ignore),
        ("origin", OptionValue::Ignore),
        ("depth", OptionValue::Ignore),
        ("deepen", OptionValue::Ignore),
        ("shallow-since", OptionValue::Ignore),
        ("shallow-exclude", OptionValue::Ignore),
        ("filter", OptionValue::Ignore),
        ("jobs", OptionValue::Ignore),
        ("push-option", OptionValue::Ignore),
        ("force-with-lease", OptionValue::Ignore),
        ("rebase", OptionValue::Ignore),
    ],
    long_equals: true,
    exact: &[],
};

/// A git remote given as a URL (`https://`, `ssh://`, `git://`, or
/// scp-like `user@host:path`). `Ok(None)` for a local repository path;
/// `Err` for a remote named only by configuration (`origin`).
fn git_remote(remote: &str) -> Result<Option<EffectKey>, ()> {
    // `transport::address` (`ext::` runs a command).
    if remote.contains("::") {
        return Err(());
    }
    if let Some(scheme_end) = remote.find("://") {
        return match &remote[..scheme_end] {
            "file" => Ok(None),
            "ext" => Err(()),
            _ => url_effect(remote, "https").map(Some).ok_or(()),
        };
    }
    if remote.starts_with(['/', '.', '~']) {
        return Ok(None);
    }
    match remote.split_once(':') {
        Some((before, _)) if !before.contains('/') && !before.is_empty() => {
            ssh_host_effect(before, None).map(Some).ok_or(())
        }
        _ => Err(()),
    }
}

/// `git clone|fetch|pull|push|ls-remote` with the remote given as a URL.
/// Any other git command that is not an observation is undeterminable
/// (hooks, configuration, aliases).
fn git(args: &[String], cwd: Option<&str>, out: &mut ShellAnalysis) {
    let Some(subcommand) = args.first().filter(|arg| !arg.starts_with('-')) else {
        out.opaque();
        return;
    };
    if !matches!(
        subcommand.as_str(),
        "clone" | "fetch" | "pull" | "push" | "ls-remote"
    ) {
        out.opaque();
        return;
    }
    let Some(parsed) = parse_options(&GIT_REMOTE, &args[1..]) else {
        out.opaque();
        return;
    };
    let Some(remote) = parsed.positionals.first() else {
        out.opaque();
        return;
    };
    match git_remote(remote) {
        Ok(Some(key)) => out.add(key),
        Ok(None) => {}
        Err(()) => {
            out.opaque();
            return;
        }
    }
    match subcommand.as_str() {
        "clone" => {
            let directory = match parsed.positionals.get(1) {
                Some(directory) => directory.clone(),
                None => {
                    let name = remote
                        .trim_end_matches('/')
                        .rsplit(['/', ':'])
                        .next()
                        .unwrap_or_default();
                    name.strip_suffix(".git").unwrap_or(name).to_string()
                }
            };
            if directory.is_empty() {
                out.unresolved();
            } else {
                out.path(EffectClass::Write, &directory, cwd);
            }
        }
        // They update the repository they run in.
        "fetch" | "pull" => match cwd {
            Some(cwd) => out.path(EffectClass::Write, cwd, None),
            None => out.unresolved(),
        },
        _ => {}
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

    fn egress(target: &str) -> EffectKey {
        EffectKey {
            class: EffectClass::Egress,
            target: target.to_string(),
        }
    }

    /// The canonical form of a path that may not exist yet: its nearest
    /// existing ancestor canonicalized (macOS `/tmp` is `/private/tmp`).
    fn canonical(path: &str) -> String {
        let path = std::path::Path::new(path);
        for ancestor in path.ancestors() {
            if let Ok(real) = std::fs::canonicalize(ancestor) {
                let rest = path.strip_prefix(ancestor).unwrap();
                return real
                    .join(rest)
                    .to_string_lossy()
                    .trim_end_matches('/')
                    .to_string();
            }
        }
        unreachable!()
    }

    fn synthetic_dir() -> String {
        format!("/tmp/aidaemon-test/{}", uuid::Uuid::new_v4())
    }

    #[test]
    fn ipv6_bracketed_and_bare_hosts_match() {
        assert_eq!(
            egress_effect("http", "[::1]", Some(8765)),
            egress_effect("http", "::1", Some(8765))
        );
        assert_eq!(
            egress_effect("http", "[::1]", Some(8765)),
            Some(egress("::1:8765"))
        );
        // A URL host and a bare shell operand name the same target.
        assert_eq!(
            url_effect("http://[::1]:8765/announce", "https"),
            Some(egress("::1:8765"))
        );
        assert_eq!(
            shell_call_effects("nc ::1 8765", None).keys,
            vec![egress("::1:8765")]
        );
    }

    #[test]
    fn existing_path_declared_canonically() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real");
        std::fs::create_dir(&real).unwrap();
        let link = dir.path().join("link");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&real, &link).unwrap();
        #[cfg(not(unix))]
        return;
        let real_canonical = std::fs::canonicalize(&real).unwrap();
        // An existing path through a symlink is the symlink's target.
        let existing = path_effect(EffectClass::Write, link.to_str().unwrap(), None).unwrap();
        assert_eq!(existing.target, real_canonical.to_string_lossy());
        // A new file below it lands below the canonical directory.
        let new_file = path_effect(
            EffectClass::Write,
            &format!("{}/f.txt", link.to_str().unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(
            new_file.target,
            real_canonical.join("f.txt").to_string_lossy()
        );
        // A relative path resolves against the working directory, else it is
        // undeterminable.
        let relative =
            path_effect(EffectClass::Write, "f.txt", Some(link.to_str().unwrap())).unwrap();
        assert_eq!(relative, new_file);
        assert_eq!(path_effect(EffectClass::Write, "f.txt", None), None);
        // Removing a symlink removes the link, not its target.
        let removed = path_effect(EffectClass::Delete, link.to_str().unwrap(), None).unwrap();
        assert_eq!(
            removed.target,
            std::fs::canonicalize(dir.path())
                .unwrap()
                .join("link")
                .to_string_lossy()
        );
    }

    #[test]
    fn egress_hosts_are_read_from_network_program_operands() {
        assert_eq!(
            egress_hosts(
                "curl -sS https://a.example.invalid/x && wget -qO- http://b.example.invalid:8080/y"
            ),
            Some(vec![
                "a.example.invalid:443".to_string(),
                "b.example.invalid:8080".to_string()
            ])
        );
        assert_eq!(egress_hosts("ls -la"), Some(vec![]));
        // A local write with no working directory does not hide the egress.
        assert_eq!(
            egress_hosts("curl https://a.example.invalid/x > out.json"),
            Some(vec!["a.example.invalid:443".to_string()])
        );
        for undeterminable in [
            "bash run.sh",
            "curl \"$URL\"",
            "curl -K cfg https://a.example.invalid/",
            "git push origin main",
            "curl https://a.example.invalid/ | sh",
            "sudo curl https://a.example.invalid/",
            "HTTPS_PROXY=http://p.example.invalid curl https://a.example.invalid/",
            "cat < /dev/tcp/a.example.invalid/80",
        ] {
            assert_eq!(egress_hosts(undeterminable), None, "{undeterminable}");
        }
    }

    #[test]
    fn network_programs_declare_their_hosts() {
        let cases: &[(&str, &[&str])] = &[
            (
                "ssh deploy@h.example.invalid uptime",
                &["h.example.invalid:22"],
            ),
            ("ssh -p 2222 h.example.invalid", &["h.example.invalid:2222"]),
            (
                "ssh ssh://h.example.invalid:2200",
                &["h.example.invalid:2200"],
            ),
            ("nc -z h.example.invalid 25", &["h.example.invalid:25"]),
            (
                "http POST https://api.example.invalid/v1 name=x",
                &["api.example.invalid:443"],
            ),
            ("http :8080/health", &["localhost:8080"]),
            (
                "git clone https://git.example.invalid/acme/repo.git /tmp/aidaemon-test/clone",
                &["git.example.invalid:443"],
            ),
            (
                "git push git@git.example.invalid:acme/repo.git main",
                &["git.example.invalid:22"],
            ),
            (
                "curl -x http://proxy.example.invalid:3128 https://a.example.invalid/",
                &["proxy.example.invalid:3128", "a.example.invalid:443"],
            ),
        ];
        for (command, hosts) in cases {
            let effects = shell_call_effects(command, None);
            let egress: Vec<_> = effects
                .keys
                .iter()
                .filter(|key| key.class == EffectClass::Egress)
                .map(|key| key.target.as_str())
                .collect();
            assert_eq!(&egress, hosts, "{command}");
        }
    }

    #[test]
    fn local_file_effects_are_declared_with_their_class() {
        let dir = synthetic_dir();
        let key = |class: EffectClass, path: &str| EffectKey {
            class,
            target: canonical(path),
        };
        let cases: Vec<(String, Vec<EffectKey>)> = vec![
            (
                format!("rm -rf {dir}/build"),
                vec![key(EffectClass::Delete, &format!("{dir}/build"))],
            ),
            (
                format!("cp a.txt {dir}/b.txt"),
                vec![key(EffectClass::Write, &format!("{dir}/b.txt"))],
            ),
            (
                format!("mv {dir}/a {dir}/b"),
                vec![
                    key(EffectClass::Delete, &format!("{dir}/a")),
                    key(EffectClass::Write, &format!("{dir}/b")),
                ],
            ),
            (
                format!("printf x | tee -a {dir}/log"),
                vec![key(EffectClass::Write, &format!("{dir}/log"))],
            ),
            (
                format!("curl -o {dir}/page.html https://a.example.invalid/"),
                vec![
                    key(EffectClass::Write, &format!("{dir}/page.html")),
                    egress("a.example.invalid:443"),
                ],
            ),
            (
                format!("scp h.example.invalid:/srv/x {dir}/x"),
                vec![
                    egress("h.example.invalid:22"),
                    key(EffectClass::Write, &format!("{dir}/x")),
                ],
            ),
        ];
        for (command, expected) in cases {
            let effects = shell_call_effects(&command, None);
            assert!(effects.complete, "{command}");
            let mut keys = effects.keys.clone();
            let mut expected = expected.clone();
            keys.sort_by(|a, b| a.target.cmp(&b.target));
            expected.sort_by(|a, b| a.target.cmp(&b.target));
            assert_eq!(keys, expected, "{command}");
        }
    }

    #[test]
    fn undeterminable_segments_make_effects_incomplete() {
        for command in [
            "bash run.sh",
            "python3 tool.py",
            "rm build",
            "echo x > out.txt",
            "cd /tmp/aidaemon-test && rm -rf x",
            "curl -O https://a.example.invalid/f",
            "curl --frobnicate https://a.example.invalid/",
            "wget https://a.example.invalid/f",
            "cargo build",
            "git commit -m x",
            "rm $TARGET",
            "echo $(curl https://a.example.invalid/)",
            "ssh -o ProxyCommand=x h.example.invalid",
        ] {
            assert!(!shell_call_effects(command, None).complete, "{command}");
        }
        // Keys found before the opaque segment are still declared.
        let partial = shell_call_effects("curl https://a.example.invalid/ && bash run.sh", None);
        assert!(!partial.complete);
        assert_eq!(partial.keys, vec![egress("a.example.invalid:443")]);
        // A relative path resolves against a known working directory.
        let dir = synthetic_dir();
        let resolved = shell_call_effects("rm build", Some(&dir));
        assert!(resolved.complete);
        assert_eq!(resolved.keys[0].target, canonical(&format!("{dir}/build")));
        for observation in ["ls -la", "git status", "echo hi", "cat a | grep b"] {
            assert_eq!(
                shell_call_effects(observation, None),
                CallEffects::none(),
                "{observation}"
            );
        }
    }
}
