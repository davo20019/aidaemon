use crate::traits::{ToolCallSemantics, ToolMutationEffects, ToolVerificationMode};

/// True when a shell path is an operating-system discard sink rather than a
/// durable filesystem target. All command-policy consumers use this shared
/// resource classification so mutation semantics and access-manifest
/// extraction cannot disagree about the same redirection.
pub(crate) fn is_discard_sink_path(path: &str) -> bool {
    path.trim() == "/dev/null"
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

/// Strip a leading `cd <dir> &&` or `cd <dir>;` prefix from a
/// command string.  Returns a `Cow` so we avoid allocating when there is
/// nothing to strip.
fn strip_leading_cd(cmd: &str) -> std::borrow::Cow<'_, str> {
    let trimmed = cmd.trim_start();
    if !trimmed.starts_with("cd ") {
        return std::borrow::Cow::Borrowed(cmd);
    }
    // Find the separator (`&&` or `;`) that ends the `cd` invocation.
    for sep in &[" && ", "; "] {
        if let Some(pos) = trimmed.find(sep) {
            let rest = trimmed[pos + sep.len()..].trim_start();
            if !rest.is_empty() {
                return std::borrow::Cow::Owned(rest.to_string());
            }
        }
    }
    // No separator found — the whole command is just `cd <dir>`.
    std::borrow::Cow::Borrowed(cmd)
}

#[derive(Debug)]
struct ShellStructure {
    segments: Vec<String>,
    has_output_redirection: bool,
}

fn parse_shell_structure(command: &str) -> Option<ShellStructure> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut chars = command.chars().peekable();
    let mut quote: Option<char> = None;
    let mut escaped = false;
    let mut has_output_redirection = false;

    while let Some(ch) = chars.next() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        if ch == '\\' && quote != Some('\'') {
            escaped = true;
            current.push(ch);
            continue;
        }
        if let Some(active_quote) = quote {
            current.push(ch);
            if ch == active_quote {
                quote = None;
            }
            continue;
        }
        if matches!(ch, '\'' | '"') {
            quote = Some(ch);
            current.push(ch);
            continue;
        }

        let is_separator = match ch {
            ';' => true,
            '&' if chars.peek() == Some(&'&') => {
                chars.next();
                true
            }
            '|' => {
                if chars.peek() == Some(&'|') {
                    chars.next();
                }
                true
            }
            '>' => {
                has_output_redirection = true;
                if matches!(chars.peek(), Some('>') | Some('|')) {
                    chars.next();
                }
                false
            }
            _ => false,
        };
        if is_separator {
            let segment = current.trim();
            if segment.is_empty() {
                return None;
            }
            segments.push(segment.to_string());
            current.clear();
        } else {
            current.push(ch);
        }
    }

    if escaped || quote.is_some() {
        return None;
    }
    let tail = current.trim();
    if !tail.is_empty() {
        segments.push(tail.to_string());
    }
    if segments.is_empty() {
        return None;
    }
    Some(ShellStructure {
        segments,
        has_output_redirection,
    })
}

/// Remove redirections that discard or duplicate streams without touching
/// the filesystem (`2>/dev/null`, `>/dev/null`, `&>/dev/null`, `2>&1`).
/// They are pure noise suppression, not mutations.
fn strip_non_mutating_redirections(command: &str) -> String {
    static NON_MUTATING_REDIRECT: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| {
            regex::Regex::new(r"(?:[0-9]+|&)?>{1,2}\s*/dev/null|[0-9]+>&[0-9]+").unwrap()
        });
    NON_MUTATING_REDIRECT.replace_all(command, " ").into_owned()
}

pub(crate) fn classify_shell_command(command: &str) -> ToolCallSemantics {
    classify_shell_command_with_depth(command, 0)
}

const MAX_NESTED_SHELL_CLASSIFICATION_DEPTH: usize = 8;

fn classify_shell_command_with_depth(command: &str, depth: usize) -> ToolCallSemantics {
    if depth > MAX_NESTED_SHELL_CLASSIFICATION_DEPTH {
        return ToolCallSemantics::mutation();
    }
    let command = strip_non_mutating_redirections(command);
    let Some(structure) = parse_shell_structure(&command) else {
        return ToolCallSemantics::default();
    };
    let mut observes = false;
    let mut mutates = structure.has_output_redirection;
    let mut unknown = false;
    let mut mutation_effects = if structure.has_output_redirection {
        ToolMutationEffects::LOCAL_SOURCE_WRITE
    } else {
        ToolMutationEffects::NONE
    };
    for segment in structure.segments {
        let semantics = classify_simple_shell_command(&segment, depth);
        observes |= semantics.observes_state();
        mutates |= semantics.mutates_state();
        unknown |= semantics.effect == crate::traits::ToolCallEffect::Unknown;
        mutation_effects = mutation_effects.union(semantics.mutation_effects);
    }
    match (observes, mutates) {
        (true, true) => ToolCallSemantics::observation_and_mutation_with(mutation_effects)
            .with_verification_mode(ToolVerificationMode::ResultContent),
        (true, false) => ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent),
        (false, true) => ToolCallSemantics::mutation_with(mutation_effects),
        (false, false) if unknown => ToolCallSemantics::default(),
        (false, false) => ToolCallSemantics::administrative(),
    }
}

/// Resolve static uncertainty from the capability boundary actually enforced
/// for one terminal call. An opaque program is observational only when a
/// native confinement profile is active and exposes neither writable task
/// targets nor network access. The runtime sandbox, rather than inspection of
/// embedded source text, enforces that claim.
pub(crate) fn classify_confined_shell_command(
    command: &str,
    confinement_active: bool,
    has_write_targets: bool,
) -> ToolCallSemantics {
    let semantics = classify_shell_command(command);
    if semantics.effect != crate::traits::ToolCallEffect::Unknown || !confinement_active {
        return semantics;
    }
    if has_write_targets {
        ToolCallSemantics::mutation()
    } else {
        ToolCallSemantics::observation().with_verification_mode(ToolVerificationMode::ResultContent)
    }
}

fn observation() -> ToolCallSemantics {
    ToolCallSemantics::observation().with_verification_mode(ToolVerificationMode::ResultContent)
}

fn observation_and_derived_mutation() -> ToolCallSemantics {
    ToolCallSemantics::observation_and_mutation_with(ToolMutationEffects::LOCAL_DERIVED_WRITE)
        .with_verification_mode(ToolVerificationMode::ResultContent)
}

fn mutation_with(effects: ToolMutationEffects) -> ToolCallSemantics {
    ToolCallSemantics::mutation_with(effects)
}

fn looks_like_env_assignment(token: &str) -> bool {
    let Some((name, _)) = token.split_once('=') else {
        return false;
    };
    !name.is_empty()
        && name.chars().enumerate().all(|(index, ch)| {
            ch == '_' || ch.is_ascii_alphanumeric() && (index > 0 || !ch.is_ascii_digit())
        })
}

fn executable_name(token: &str) -> &str {
    std::path::Path::new(token)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(token)
}

fn known_cli_supports_inert_introspection(executable: &str) -> bool {
    matches!(
        executable,
        "node"
            | "npm"
            | "python"
            | "python3"
            | "ruby"
            | "perl"
            | "php"
            | "java"
            | "javac"
            | "go"
            | "rustc"
            | "cargo"
            | "deno"
            | "bun"
            | "yarn"
            | "pnpm"
            | "corepack"
            | "wrangler"
            | "git"
            | "gh"
            | "docker"
            | "podman"
            | "kubectl"
            | "terraform"
            | "tofu"
            | "pulumi"
            | "cloudflared"
            | "make"
            | "cmake"
            | "gradle"
            | "mvn"
            | "aws"
            | "gcloud"
            | "az"
    )
}

fn is_inert_introspection(executable: &str, args: &[String]) -> bool {
    if !known_cli_supports_inert_introspection(executable) {
        return false;
    }
    // Introspection flags are only authoritative as the complete invocation.
    // A flag occurring after a subcommand or script belongs to that program:
    // `git commit -v`, `docker run -v ...`, and `node script.js --help` can all
    // mutate state and must never inherit read-only semantics.
    matches!(
        args,
        [only]
            if matches!(
                only.as_str(),
                "--version" | "-version" | "-v" | "--help" | "-h" | "version" | "help"
            )
    )
}

fn git_semantics(args: &[String]) -> ToolCallSemantics {
    let mut index = 0;
    while let Some(arg) = args.get(index) {
        if arg == "-c" {
            index = index.saturating_add(2);
        } else if arg.starts_with('-') {
            index += 1;
        } else {
            break;
        }
    }
    let Some(subcommand) = args.get(index).map(String::as_str) else {
        return mutation_with(ToolMutationEffects::UNSPECIFIED);
    };
    let rest = &args[index + 1..];
    match subcommand {
        "status" | "log" | "diff" | "show" | "rev-parse" | "shortlog" | "blame" | "grep"
        | "ls-files" | "ls-tree" | "describe" => observation(),
        "remote" => match rest.first().map(String::as_str) {
            None | Some("-v") | Some("--verbose") | Some("get-url") | Some("show") => observation(),
            _ => mutation_with(ToolMutationEffects::REPOSITORY_WRITE),
        },
        "branch" => {
            let mutating_flag = rest.iter().any(|arg| {
                matches!(
                    arg.as_str(),
                    "-d" | "-D"
                        | "-m"
                        | "-M"
                        | "-c"
                        | "-C"
                        | "--delete"
                        | "--move"
                        | "--copy"
                        | "--set-upstream-to"
                        | "--unset-upstream"
                )
            });
            let positional = rest.iter().any(|arg| !arg.starts_with('-'));
            if mutating_flag || positional {
                mutation_with(ToolMutationEffects::REPOSITORY_WRITE)
            } else {
                observation()
            }
        }
        "tag" => {
            let list_only = rest.is_empty()
                || rest.iter().all(|arg| {
                    arg.starts_with('-')
                        && !matches!(arg.as_str(), "-d" | "--delete" | "-a" | "--annotate")
                });
            if list_only {
                observation()
            } else {
                mutation_with(ToolMutationEffects::REPOSITORY_WRITE)
            }
        }
        "config" => {
            if rest.iter().any(|arg| {
                matches!(
                    arg.as_str(),
                    "--get" | "--get-all" | "--get-regexp" | "--list" | "-l" | "--show-origin"
                )
            }) {
                observation()
            } else {
                mutation_with(ToolMutationEffects::CONFIGURATION)
            }
        }
        "add" | "commit" | "notes" => mutation_with(ToolMutationEffects::REPOSITORY_WRITE),
        "push" => mutation_with(ToolMutationEffects::REMOTE_MUTATION),
        "fetch" => mutation_with(ToolMutationEffects::REPOSITORY_WRITE),
        "pull" => mutation_with(
            ToolMutationEffects::LOCAL_WORKSPACE_WRITE.union(ToolMutationEffects::REPOSITORY_WRITE),
        ),
        "checkout" | "switch" | "restore" | "reset" | "merge" | "rebase" | "cherry-pick"
        | "revert" | "stash" | "apply" | "am" => mutation_with(
            ToolMutationEffects::LOCAL_WORKSPACE_WRITE.union(ToolMutationEffects::REPOSITORY_WRITE),
        ),
        "clean" | "rm" => mutation_with(
            ToolMutationEffects::LOCAL_WORKSPACE_WRITE
                .union(ToolMutationEffects::REPOSITORY_WRITE)
                .union(ToolMutationEffects::DESTRUCTIVE),
        ),
        "mv" | "init" | "clone" => mutation_with(
            ToolMutationEffects::LOCAL_WORKSPACE_WRITE.union(ToolMutationEffects::REPOSITORY_WRITE),
        ),
        "gc" | "repack" => mutation_with(ToolMutationEffects::REPOSITORY_WRITE),
        _ => mutation_with(ToolMutationEffects::UNSPECIFIED),
    }
}

fn curl_semantics(args: &[String]) -> ToolCallSemantics {
    let mut effects = ToolMutationEffects::NONE;
    let mut index = 0;
    while let Some(arg) = args.get(index) {
        let flag = arg.as_str();
        if matches!(
            flag,
            "-d" | "--data"
                | "--data-ascii"
                | "--data-binary"
                | "--data-raw"
                | "--data-urlencode"
                | "--json"
                | "-F"
                | "--form"
                | "--form-string"
                | "-T"
                | "--upload-file"
        ) || flag.starts_with("--data=")
            || flag.starts_with("--json=")
            || flag.starts_with("--form=")
            || flag.starts_with("--upload-file=")
        {
            effects = effects.union(ToolMutationEffects::REMOTE_MUTATION);
        }
        if matches!(
            flag,
            "-o" | "--output" | "-O" | "--remote-name" | "--remote-header-name" | "--create-dirs"
        ) || flag.starts_with("--output=")
        {
            effects = effects.union(ToolMutationEffects::LOCAL_WORKSPACE_WRITE);
        }
        if matches!(flag, "-X" | "--request") {
            if args.get(index + 1).is_some_and(|method| {
                !matches!(
                    method.to_ascii_uppercase().as_str(),
                    "GET" | "HEAD" | "OPTIONS"
                )
            }) {
                effects = effects.union(ToolMutationEffects::REMOTE_MUTATION);
            }
            index += 1;
        } else if let Some(method) = flag.strip_prefix("-X") {
            if !method.is_empty()
                && !matches!(
                    method.to_ascii_uppercase().as_str(),
                    "GET" | "HEAD" | "OPTIONS"
                )
            {
                effects = effects.union(ToolMutationEffects::REMOTE_MUTATION);
            }
        }
        index += 1;
    }
    if !effects.is_empty() {
        mutation_with(effects)
    } else {
        observation()
    }
}

fn npm_semantics(args: &[String]) -> ToolCallSemantics {
    let Some(subcommand) = args.iter().find(|arg| !arg.starts_with('-')) else {
        return mutation_with(ToolMutationEffects::UNSPECIFIED);
    };
    match subcommand.as_str() {
        "audit" if args.iter().any(|arg| arg == "fix" || arg == "--fix") => mutation_with(
            ToolMutationEffects::LOCAL_SOURCE_WRITE
                .union(ToolMutationEffects::LOCAL_DERIVED_WRITE)
                .union(ToolMutationEffects::CONFIGURATION),
        ),
        "audit" | "outdated" | "ls" | "list" | "view" | "info" | "show" | "search" | "whoami"
        | "prefix" | "root" | "bin" | "fund" => observation(),
        "config"
            if args
                .get(1)
                .is_some_and(|arg| matches!(arg.as_str(), "get" | "list" | "ls")) =>
        {
            observation()
        }
        "pkg" if args.get(1).is_some_and(|arg| arg == "get") => observation(),
        "test" | "t" => observation_and_derived_mutation(),
        "install" | "i" | "ci" | "update" | "uninstall" | "remove" => mutation_with(
            ToolMutationEffects::LOCAL_WORKSPACE_WRITE
                .union(ToolMutationEffects::LOCAL_DERIVED_WRITE)
                .union(ToolMutationEffects::CONFIGURATION),
        ),
        "publish" | "unpublish" | "deprecate" | "access" | "owner" => {
            mutation_with(ToolMutationEffects::REMOTE_MUTATION)
        }
        "version" => mutation_with(
            ToolMutationEffects::LOCAL_SOURCE_WRITE.union(ToolMutationEffects::REPOSITORY_WRITE),
        ),
        "pkg" => mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE),
        "config" => mutation_with(ToolMutationEffects::CONFIGURATION),
        "cache" => mutation_with(ToolMutationEffects::LOCAL_DERIVED_WRITE),
        "run" | "run-script" => {
            let script = args
                .iter()
                .skip_while(|arg| *arg != subcommand)
                .skip(1)
                .find(|arg| !arg.starts_with('-'))
                .map(|arg| arg.to_ascii_lowercase())
                .unwrap_or_default();
            if contains_any(
                &script,
                &["test", "lint", "check", "typecheck", "audit", "verify"],
            ) {
                observation_and_derived_mutation()
            } else if contains_any(&script, &["build", "compile", "bundle", "generate"]) {
                mutation_with(ToolMutationEffects::LOCAL_DERIVED_WRITE)
            } else {
                mutation_with(ToolMutationEffects::UNSPECIFIED)
            }
        }
        _ => mutation_with(ToolMutationEffects::UNSPECIFIED),
    }
}

fn wrangler_semantics(args: &[String]) -> ToolCallSemantics {
    match args.first().map(String::as_str) {
        Some("whoami" | "tail") => observation(),
        Some("deploy") => mutation_with(ToolMutationEffects::REMOTE_DEPLOY),
        Some("pages") if args.get(1).is_some_and(|arg| arg == "deploy") => {
            mutation_with(ToolMutationEffects::REMOTE_DEPLOY)
        }
        Some("versions")
            if args
                .get(1)
                .is_some_and(|arg| matches!(arg.as_str(), "upload" | "deploy")) =>
        {
            mutation_with(ToolMutationEffects::REMOTE_DEPLOY)
        }
        Some("deployments" | "versions")
            if args
                .get(1)
                .is_some_and(|arg| matches!(arg.as_str(), "list" | "view" | "status")) =>
        {
            observation()
        }
        Some("pages")
            if args
                .get(1)
                .is_some_and(|arg| matches!(arg.as_str(), "project" | "deployment"))
                && args.get(2).is_some_and(|arg| arg == "list") =>
        {
            observation()
        }
        Some("delete") => mutation_with(
            ToolMutationEffects::REMOTE_MUTATION.union(ToolMutationEffects::DESTRUCTIVE),
        ),
        _ => mutation_with(ToolMutationEffects::REMOTE_MUTATION),
    }
}

fn classify_simple_shell_command(command: &str, depth: usize) -> ToolCallSemantics {
    let command = command.trim();
    if command.is_empty() {
        return ToolCallSemantics::administrative();
    }

    // Command substitutions can execute an arbitrary nested program even when
    // the outer command only prints. Fail closed instead of trusting the outer
    // executable name.
    if contains_any(command, &["$(", "`", "<(", ">("]) {
        return ToolCallSemantics::mutation();
    }

    // The compound-command parser normally splits this prefix first. Retain
    // this normalization for direct callers and malformed-but-recoverable input.
    let command = strip_leading_cd(command);
    let Ok(tokens) = shell_words::split(&command) else {
        return ToolCallSemantics::default();
    };
    let mut command_index = 0;
    while tokens
        .get(command_index)
        .is_some_and(|token| looks_like_env_assignment(token))
    {
        command_index += 1;
    }
    let Some(command_token) = tokens.get(command_index) else {
        return ToolCallSemantics::administrative();
    };
    let executable = executable_name(command_token).to_ascii_lowercase();
    let executable = executable.as_str();
    let args = &tokens[command_index + 1..];

    // Shell launchers are transport, not the operation being authorized. Parse
    // the script passed to `-c` using the same compound-command classifier so
    // sequencing, pipes, and redirections retain their real aggregate effects.
    // Unknown invocation shapes and excessive nesting remain fail-closed.
    if matches!(executable, "sh" | "bash" | "dash" | "ksh" | "zsh") {
        let script_index = args.iter().enumerate().find_map(|(index, arg)| {
            (arg == "-c" || arg.starts_with('-') && arg[1..].contains('c')).then_some(index + 1)
        });
        return script_index
            .and_then(|index| args.get(index))
            .map_or_else(ToolCallSemantics::mutation, |script| {
                classify_shell_command_with_depth(script, depth + 1)
            });
    }

    if executable == "env" {
        let nested = args
            .iter()
            .position(|arg| !arg.starts_with('-') && !looks_like_env_assignment(arg));
        return nested.map_or_else(observation, |index| {
            classify_simple_shell_command(&args[index..].join(" "), depth)
        });
    }
    if executable == "cd" {
        return ToolCallSemantics::administrative();
    }
    if matches!(executable, "npx" | "bunx") {
        let mut index = 0;
        while let Some(arg) = args.get(index) {
            if matches!(
                arg.as_str(),
                "-p" | "--package" | "-c" | "--call" | "--cache" | "--userconfig" | "--registry"
            ) {
                index = index.saturating_add(2);
            } else if arg.starts_with('-') {
                index += 1;
            } else {
                break;
            }
        }
        let nested = (index < args.len()).then_some(index);
        return nested.map_or_else(ToolCallSemantics::mutation, |index| {
            classify_simple_shell_command(&args[index..].join(" "), depth)
        });
    }
    if executable == "pnpm" && args.first().is_some_and(|arg| arg == "dlx") {
        return if args.len() > 1 {
            classify_simple_shell_command(&args[1..].join(" "), depth)
        } else {
            ToolCallSemantics::mutation()
        };
    }
    if is_inert_introspection(executable, args) {
        return observation();
    }

    // Document extractors may default to writing beside the source. Only an
    // explicit stdout target is a pure observation.
    if executable == "pdftotext" {
        return if args.last().is_some_and(|arg| arg == "-") {
            observation()
        } else {
            mutation_with(ToolMutationEffects::LOCAL_WORKSPACE_WRITE)
        };
    }

    if executable == "find" {
        let mutating_primary = args.iter().any(|arg| {
            matches!(
                arg.as_str(),
                "-delete"
                    | "-exec"
                    | "-execdir"
                    | "-ok"
                    | "-okdir"
                    | "-fls"
                    | "-fprint"
                    | "-fprint0"
                    | "-fprintf"
            )
        });
        return if mutating_primary {
            mutation_with(
                ToolMutationEffects::LOCAL_WORKSPACE_WRITE.union(ToolMutationEffects::DESTRUCTIVE),
            )
        } else {
            observation()
        };
    }

    if executable == "curl" {
        return curl_semantics(args);
    }
    if executable == "wget" {
        let stdout_only = args.iter().any(|arg| {
            matches!(arg.as_str(), "--spider" | "-qo-" | "-o-") || arg == "--output-document=-"
        }) || args
            .windows(2)
            .any(|pair| pair[0] == "-o" && pair[1] == "-");
        return if stdout_only {
            observation()
        } else {
            mutation_with(ToolMutationEffects::LOCAL_WORKSPACE_WRITE)
        };
    }
    if executable == "git" {
        return git_semantics(args);
    }
    if executable == "wrangler" {
        return wrangler_semantics(args);
    }
    if executable == "npm" {
        return npm_semantics(args);
    }

    if matches!(executable, "rm" | "rmdir" | "unlink" | "shred") {
        return mutation_with(
            ToolMutationEffects::LOCAL_WORKSPACE_WRITE.union(ToolMutationEffects::DESTRUCTIVE),
        );
    }
    if matches!(
        executable,
        "mkdir" | "touch" | "cp" | "mv" | "ln" | "truncate"
    ) {
        return mutation_with(ToolMutationEffects::LOCAL_WORKSPACE_WRITE);
    }
    if executable == "tee" {
        return mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE);
    }
    if matches!(executable, "chmod" | "chown" | "chgrp" | "xattr") {
        return mutation_with(ToolMutationEffects::CONFIGURATION);
    }
    if matches!(executable, "kill" | "killall" | "pkill") {
        return mutation_with(ToolMutationEffects::PROCESS_STATE);
    }
    if executable == "launchctl"
        && args.first().is_some_and(|arg| {
            matches!(
                arg.as_str(),
                "print" | "list" | "print-disabled" | "procinfo" | "blame"
            )
        })
    {
        return observation();
    }
    if executable == "systemctl"
        && args.first().is_some_and(|arg| {
            matches!(
                arg.as_str(),
                "status" | "show" | "is-active" | "is-enabled" | "list-units" | "list-unit-files"
            )
        })
    {
        return observation();
    }
    if executable == "service" && args.get(1).is_some_and(|arg| arg == "status") {
        return observation();
    }
    if matches!(executable, "launchctl" | "systemctl" | "service") {
        return mutation_with(
            ToolMutationEffects::PROCESS_STATE.union(ToolMutationEffects::CONFIGURATION),
        );
    }

    if matches!(
        executable,
        "ls" | "pwd"
            | "cat"
            | "head"
            | "tail"
            | "rg"
            | "grep"
            | "stat"
            | "wc"
            | "mdls"
            | "mdfind"
            | "locate"
            | "strings"
            | "date"
            | "uname"
            | "whoami"
            | "hostname"
            | "uptime"
            | "ps"
            | "printenv"
            | "echo"
            | "printf"
            | "true"
            | "false"
            | "sleep"
            | "test"
            | "["
            | "tree"
            | "du"
            | "df"
            | "file"
            | "diff"
            | "which"
            | "whereis"
    ) {
        return observation();
    }
    if executable == "sort" {
        return if args
            .iter()
            .any(|arg| arg == "-o" || arg.starts_with("--output="))
        {
            mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE)
        } else {
            observation()
        };
    }
    if executable == "uniq" {
        let positional_count = args.iter().filter(|arg| !arg.starts_with('-')).count();
        return if positional_count > 1 {
            mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE)
        } else {
            observation()
        };
    }
    if executable == "sed" {
        return if args
            .iter()
            .any(|arg| arg == "-i" || arg.starts_with("-i") || arg == "--in-place")
        {
            mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE)
        } else {
            observation()
        };
    }

    let subcommand = args
        .iter()
        .find(|arg| !arg.starts_with('-'))
        .map(String::as_str);
    match executable {
        "cargo" => match subcommand {
            Some("tree" | "metadata") => observation(),
            Some("test" | "check" | "clippy") => observation_and_derived_mutation(),
            Some("fmt") if args.iter().any(|arg| arg == "--check") => {
                observation_and_derived_mutation()
            }
            Some("build") => mutation_with(ToolMutationEffects::LOCAL_DERIVED_WRITE),
            Some("fmt") => mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE),
            Some("install" | "uninstall") => mutation_with(ToolMutationEffects::CONFIGURATION),
            _ => mutation_with(ToolMutationEffects::UNSPECIFIED),
        },
        "go" => match subcommand {
            Some("version" | "env") => observation(),
            Some("test" | "list" | "vet") => observation_and_derived_mutation(),
            Some("build") => mutation_with(ToolMutationEffects::LOCAL_DERIVED_WRITE),
            Some("fmt") => mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE),
            _ => mutation_with(ToolMutationEffects::UNSPECIFIED),
        },
        "pytest" | "jest" | "vitest" => observation_and_derived_mutation(),
        "python" | "python3" if args.windows(2).any(|pair| pair == ["-m", "pytest"]) => {
            observation_and_derived_mutation()
        }
        // Interpreter source is opaque executable input. Do not guess its
        // effects from code strings. The confined capability boundary resolves
        // this unknown to observation or mutation for the actual invocation.
        "python" | "python3" => ToolCallSemantics::default(),
        "yarn" | "pnpm" | "bun" => match subcommand {
            Some("test" | "lint" | "check" | "typecheck" | "audit") => {
                observation_and_derived_mutation()
            }
            Some("build") => mutation_with(ToolMutationEffects::LOCAL_DERIVED_WRITE),
            Some("install" | "add" | "remove" | "update") => mutation_with(
                ToolMutationEffects::LOCAL_WORKSPACE_WRITE
                    .union(ToolMutationEffects::LOCAL_DERIVED_WRITE)
                    .union(ToolMutationEffects::CONFIGURATION),
            ),
            _ => mutation_with(ToolMutationEffects::UNSPECIFIED),
        },
        _ => mutation_with(ToolMutationEffects::UNSPECIFIED),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_extractors_classify_as_observation() {
        // These are the exact commands read_file's PDF/Word stubs recommend —
        // they must classify as observation or the text-only gate blocks the
        // very command we told the model to run.
        for cmd in [
            "pdftotext -layout \"/Users/x/Downloads/Offer Letter (1).pdf\" -",
            "mdls -raw -name kMDItemTextContent \"/Users/x/file.pdf\"",
            "mdfind -name \"Offer Letter\"",
            "strings /tmp/blob.bin",
        ] {
            let semantics = classify_shell_command(cmd);
            assert!(
                semantics.observes_state() && !semantics.mutates_state(),
                "expected pure observation for: {cmd}"
            );
        }
    }

    #[test]
    fn pdftotext_without_stdout_target_still_mutates() {
        // `pdftotext report.pdf` writes report.txt next to the input.
        let semantics = classify_shell_command("pdftotext report.pdf");
        assert!(semantics.mutates_state());
    }

    #[test]
    fn shell_launcher_inherits_compound_script_effects() {
        let observation =
            classify_shell_command("/bin/sh -lc 'sleep 1; printf SYNTHETIC_OK\\n; /usr/bin/false'");
        assert!(observation.observes_state());
        assert!(!observation.mutates_state());

        let mutation = classify_shell_command("bash -lc 'printf x > synthetic-output.txt'");
        assert!(mutation.mutates_state());
    }

    #[test]
    fn shell_launcher_observation_survives_literal_escaped_quotes() {
        let command = r#"/bin/sh -c 'sleep 35; printf \'SYNTHETIC_OK\\n\''"#;
        let observation = classify_confined_shell_command(command, true, false);
        assert!(observation.observes_state());
        assert!(!observation.mutates_state());

        assert_eq!(
            classify_shell_command(command).effect,
            crate::traits::ToolCallEffect::Unknown
        );
        assert!(classify_confined_shell_command(command, true, true).mutates_state());
        assert_eq!(
            classify_confined_shell_command(command, false, false).effect,
            crate::traits::ToolCallEffect::Unknown
        );
    }

    #[test]
    fn dev_null_redirect_does_not_count_as_mutation() {
        // `find ... 2>/dev/null` is the canonical noise-suppressed lookup;
        // discarding output mutates nothing.
        let semantics = classify_shell_command("find ~ -name \"*Offer Letter*.pdf\" 2>/dev/null");
        assert!(semantics.observes_state());
        assert!(!semantics.mutates_state());
    }

    #[test]
    fn stderr_to_stdout_dup_does_not_count_as_mutation() {
        let semantics = classify_shell_command("ls -la /tmp 2>&1");
        assert!(semantics.observes_state());
        assert!(!semantics.mutates_state());
    }

    #[test]
    fn redirect_to_real_file_still_mutates() {
        let semantics = classify_shell_command("ls -la > files.txt");
        assert!(semantics.mutates_state());
    }

    #[test]
    fn append_to_dev_null_does_not_mutate() {
        let semantics = classify_shell_command("grep -r foo . >> /dev/null");
        assert!(!semantics.mutates_state());
    }

    #[test]
    fn test_strip_leading_cd() {
        // No cd prefix — returned as-is
        assert_eq!(
            strip_leading_cd("python3 foo.py").as_ref(),
            "python3 foo.py"
        );

        // cd with &&
        assert_eq!(
            strip_leading_cd("cd /home/user/project && python3 foo.py").as_ref(),
            "python3 foo.py"
        );

        // cd with ;
        assert_eq!(strip_leading_cd("cd /tmp; ls -la").as_ref(), "ls -la");

        // Just cd, no following command
        assert_eq!(strip_leading_cd("cd /home/user").as_ref(), "cd /home/user");

        // Nested cd (only strips first)
        assert_eq!(
            strip_leading_cd("cd /a && cd /b && echo hi").as_ref(),
            "cd /b && echo hi"
        );
    }

    #[test]
    fn test_classify_with_cd_prefix() {
        // Embedded interpreter source remains opaque until the enforced
        // capability boundary resolves it for a concrete invocation.
        let sem = classify_shell_command("python3 -c 'print(1)'");
        assert_eq!(sem.effect, crate::traits::ToolCallEffect::Unknown);
        let sem = classify_confined_shell_command(
            "cd /home/user/project && python3 -c 'print(1)'",
            true,
            false,
        );
        assert!(sem.observes_state());
        assert!(!sem.mutates_state());

        let sem = classify_shell_command("cd /tmp && ls -la");
        assert!(sem.observes_state(), "cd-prefixed ls should be observation");
        assert!(
            !sem.mutates_state(),
            "cd-prefixed ls should NOT be mutation"
        );

        let sem = classify_shell_command("cd /project && cargo test");
        assert!(
            sem.observes_state(),
            "cd-prefixed cargo test should observe"
        );

        let sem = classify_shell_command("cd /project && cargo build");
        assert!(
            sem.mutates_state(),
            "cd-prefixed cargo build should be mutation"
        );
    }

    #[test]
    fn output_redirection_is_state_mutating() {
        for command in [
            "echo value > file",
            "cat > file",
            "echo value >> file",
            "echo value 2> errors.log",
            "echo value &> all.log",
        ] {
            assert!(
                classify_shell_command(command).mutates_state(),
                "{command} must be mutating"
            );
        }
    }

    #[test]
    fn compound_and_pipeline_semantics_include_every_segment() {
        let mixed = classify_shell_command("ls && rm file");
        assert!(mixed.observes_state());
        assert!(mixed.mutates_state());

        let observation = classify_shell_command("rg pattern file | head");
        assert!(observation.observes_state());
        assert!(!observation.mutates_state());

        let tee = classify_shell_command("rg pattern file | tee output.txt");
        assert!(tee.observes_state());
        assert!(tee.mutates_state());
    }

    #[test]
    fn quoted_and_escaped_operators_are_not_shell_structure() {
        for command in [
            "echo 'value > file'",
            "echo \"a | b && c; d\"",
            r"echo value \> file",
        ] {
            let semantics = classify_shell_command(command);
            assert!(semantics.observes_state(), "{command} should observe");
            assert!(
                !semantics.mutates_state(),
                "{command} should not be classified as a mutation"
            );
        }
    }

    #[test]
    fn ambiguous_shell_expression_remains_typed_unknown() {
        let semantics = classify_shell_command("echo 'unterminated");
        assert_eq!(semantics.effect, crate::traits::ToolCallEffect::Unknown);
        assert!(!semantics.observes_state());
    }

    #[test]
    fn runtime_and_cloud_cli_introspection_are_observations() {
        for command in [
            "node --version",
            "npm --version",
            "wrangler --version",
            "wrangler whoami",
            "cd /Users/example/projects && node --version",
            "cd /Users/example/projects && npm --version",
            "cd /Users/example/projects && wrangler --version",
            "cd /Users/example/projects && wrangler whoami",
            "node --version && npm --version && wrangler --version && wrangler whoami",
            "launchctl print gui/501/ai.aidaemon",
            "systemctl is-active aidaemon",
            "service aidaemon status",
        ] {
            let semantics = classify_shell_command(command);
            assert!(semantics.observes_state(), "{command} should observe");
            assert!(
                !semantics.mutates_state(),
                "{command} must not request a mutation checkpoint"
            );
        }
    }

    #[test]
    fn similarly_named_mutations_do_not_inherit_read_only_semantics() {
        for command in [
            "node build.js",
            "node build.js --version",
            "npm install",
            "wrangler deploy",
            "npx wrangler deploy",
            "git commit -v",
            "docker run -v /tmp:/work image",
            "git branch feature/new-work",
            "git tag v1.2.3",
            "find . -delete",
            "curl -o response.json https://example.com",
            "curl -T artifact.zip https://example.com/upload",
            "wget https://example.com/archive.zip",
            "npm audit fix",
            "sort input.txt -o output.txt",
            "uniq input.txt output.txt",
            "echo $(touch changed.txt)",
        ] {
            assert!(
                classify_shell_command(command).mutates_state(),
                "{command} should remain mutating"
            );
        }
        assert_eq!(
            classify_shell_command("python mutate.py --help").effect,
            crate::traits::ToolCallEffect::Unknown,
            "opaque interpreter source must not receive effects from argv wording"
        );
    }

    #[test]
    fn typed_effects_separate_source_build_and_deployment_outcomes() {
        let tests = classify_shell_command("cargo test");
        assert!(tests
            .mutation_effects
            .contains(ToolMutationEffects::LOCAL_DERIVED_WRITE));
        assert!(!tests
            .mutation_effects
            .intersects(ToolMutationEffects::LOCAL_SOURCE_WRITE));

        let deploy = classify_shell_command("npx wrangler deploy");
        assert!(deploy
            .mutation_effects
            .contains(ToolMutationEffects::REMOTE_DEPLOY));

        let json_post =
            classify_shell_command("curl --json '{\"ok\":true}' https://example.com/api");
        assert!(json_post
            .mutation_effects
            .contains(ToolMutationEffects::REMOTE_MUTATION));

        let scaffold = classify_shell_command("mkdir -p site");
        assert!(!scaffold
            .mutation_effects
            .satisfies(ToolMutationEffects::LOCAL_SOURCE_WRITE));

        for command in [
            "echo content > index.html",
            "echo content | tee index.html",
            "sed -i '' 's/old/new/' index.html",
        ] {
            assert!(
                classify_shell_command(command)
                    .mutation_effects
                    .satisfies(ToolMutationEffects::LOCAL_SOURCE_WRITE),
                "{command} should record authored content"
            );
        }

        assert!(!ToolMutationEffects::UNSPECIFIED.satisfies(
            ToolMutationEffects::LOCAL_SOURCE_WRITE.union(ToolMutationEffects::REMOTE_DEPLOY)
        ));
        assert!(ToolMutationEffects::LOCAL_SOURCE_WRITE.satisfies(ToolMutationEffects::UNSPECIFIED));
    }

    #[test]
    fn structured_git_queries_stay_read_only() {
        for command in [
            "git status --short",
            "git remote -v",
            "git remote get-url origin",
            "git branch --show-current",
            "git tag --list",
            "git config --get remote.origin.url",
        ] {
            let semantics = classify_shell_command(command);
            assert!(semantics.observes_state(), "{command} should observe");
            assert!(!semantics.mutates_state(), "{command} should not mutate");
        }
    }
}
