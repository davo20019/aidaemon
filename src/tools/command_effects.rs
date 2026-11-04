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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CommandEffects {
    /// Every executable is a known observer used in a known observing shape.
    pub observation_only: bool,
    /// Some command removes files (`rm`, `find -delete`, `git clean`, ...),
    /// including behind wrappers and groupings.
    pub deletes: bool,
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
    if command.contains("/dev/tcp/") || command.contains("/dev/udp/") {
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

pub(crate) fn command_effects(command: &str) -> CommandEffects {
    let deletes = expanded_commands(command).iter().any(|tokens| {
        let program = basename(&tokens[0]);
        let args = &tokens[1..];
        DELETE_PROGRAMS.contains(&program)
            || (program == "find" && args.iter().any(|arg| arg == "-delete"))
            || (program == "git" && args.iter().any(|arg| arg == "clean"))
    });
    CommandEffects {
        observation_only: is_observation_only(command),
        deletes,
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
            '$' if next == Some('(') => {
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
                if next == Some('(') || (!dup && redirect_target(&chars, j) != "/dev/null") {
                    unsafe_syntax = true;
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
    if scanned.unsafe_syntax || command.contains("/dev/tcp/") || command.contains("/dev/udp/") {
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
        "tree" => !any(&|a| a == "-o"),
        "uniq" => args.iter().filter(|arg| !arg.starts_with('-')).count() < 2,
        "date" => args.is_empty() || (args.len() == 1 && args[0].starts_with('+')),
        "hostname" => args.is_empty(),
        "file" => !any(&|a| {
            a == "--compile" || a.starts_with("--magic-file") || short_cluster_has(a, &['C', 'm'])
        }),
        "git" => {
            let mut index = 0;
            while let Some(arg) = args.get(index).filter(|arg| arg.starts_with('-')) {
                if arg == "-C" {
                    index += 2;
                    continue;
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
            "git -C d status",
            "date +%Y",
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
