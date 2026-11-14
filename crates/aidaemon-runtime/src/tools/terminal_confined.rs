//! The terminal's confined auto-approve tier: a `run` of an allowlisted
//! build, lint, read-only git or inspection command runs in the confined
//! sandbox without an owner card (what `run_command` used to do).
//!
//! The allowlist is a typed table of command prefixes, each with the access
//! its process gets: read-only, or write in its working directory (builds,
//! formatters and linters write derived output and caches there). Test
//! runners and other commands whose purpose is running repository code
//! (`cargo test`/`run`/`bench`, npm/yarn/bun scripts, `npx`, `make`,
//! `go generate`, Gradle/Maven) are not in it: they go through the owner.

use std::time::{Duration, Instant};

use aidaemon_tools::backend::SharedExecutionBackend;
use aidaemon_tools::daemon_guard::detect_daemonization_primitives;
use aidaemon_tools::fs_utils;

/// What a process in the confined tier may touch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfinedAccess {
    /// Read the working directory (and Git's metadata); write nothing.
    ReadOnly,
    /// Also write in the working directory (derived output, caches).
    WorkspaceWrite,
}

/// Prefixes whose process writes derived output in its working directory.
/// Build and lint commands can still run repository code inside the
/// sandbox (Cargo build scripts and proc macros, linter config and plugins).
const WORKSPACE_WRITE_PREFIXES: &[&str] = &[
    "cargo build",
    "cargo check",
    "cargo clippy",
    "cargo fmt",
    "cargo doc",
    "go build",
    "go vet",
    "go mod",
    "rustfmt",
    "black",
    "ruff",
    "isort",
    "flake8",
    "mypy",
    "pylint",
    "eslint",
    "prettier",
    "tsc",
    "biome",
];

/// Prefixes whose process only reads.
const READ_ONLY_PREFIXES: &[&str] = &[
    "cargo tree",
    "cargo metadata",
    "npm ls",
    "npm outdated",
    "npm audit",
    // Read-only git (`branch`, `tag`, `remote`: listing forms only).
    "git status",
    "git log",
    "git diff",
    "git show",
    "git branch",
    "git remote",
    "git stash list",
    "git tag",
    "git blame",
    "git shortlog",
    "git rev-parse",
    // File inspection
    "ls",
    "wc",
    "file",
    "du",
    "df",
    "stat",
    "head",
    "tail",
    "sort",
    "uniq",
    "diff",
    "tree",
    // Environment
    "which",
    "whoami",
    "uname",
    "hostname",
    "date",
    "uptime",
    "pwd",
    // Pure output/status observations. Absolute paths are listed so an
    // observational predicate such as `/usr/bin/false` is not mislabeled.
    "printf",
    "/usr/bin/printf",
    "/bin/printf",
    "echo",
    "/bin/echo",
    "true",
    "/usr/bin/true",
    "/bin/true",
    "false",
    "/usr/bin/false",
    "/bin/false",
];

/// The access an allowlisted command line runs with, or `None` when it is
/// not in the tier: not a listed prefix (or a mutating form of `git
/// branch`/`tag`/`remote`), a shell operator, or a daemonization primitive.
pub(crate) fn confined_access(command: &str) -> Option<ConfinedAccess> {
    let command = command.trim();
    if aidaemon_policies::contains_shell_operator(command)
        || !detect_daemonization_primitives(command).is_empty()
        || !git_ref_command_only_lists(command)
    {
        return None;
    }
    let matches = |prefixes: &[&str]| {
        prefixes.iter().any(|prefix| {
            command == *prefix
                || command.starts_with(&format!("{prefix} "))
                || command.starts_with(&format!("{prefix}\t"))
        })
    };
    if matches(WORKSPACE_WRITE_PREFIXES) {
        Some(ConfinedAccess::WorkspaceWrite)
    } else if matches(READ_ONLY_PREFIXES) {
        Some(ConfinedAccess::ReadOnly)
    } else {
        None
    }
}

/// `git branch`, `git tag` and `git remote` also create, rename and delete.
/// Only their listing forms are allowlisted: known listing flags, and a
/// positional argument only where a listing takes one (a `--list` pattern,
/// the commit after `--contains`, the remote name after `get-url`).
fn git_ref_command_only_lists(cmd: &str) -> bool {
    let words: Vec<&str> = cmd.split_whitespace().collect();
    let (subcommand, args) = match words.as_slice() {
        ["git", sub, args @ ..] => (*sub, args),
        _ => return true,
    };
    const VALUE_FLAGS: &[&str] = &[
        "--contains",
        "--no-contains",
        "--merged",
        "--no-merged",
        "--points-at",
    ];
    let listing_flag = |arg: &str| {
        VALUE_FLAGS.contains(&arg)
            || matches!(
                arg,
                "-a" | "--all"
                    | "-r"
                    | "--remotes"
                    | "-l"
                    | "--list"
                    | "-v"
                    | "-vv"
                    | "--verbose"
                    | "--show-current"
                    | "-i"
                    | "--ignore-case"
                    | "--no-color"
                    | "--no-column"
            )
            || ["--sort=", "--format=", "--color", "--column", "--abbrev"]
                .iter()
                .any(|prefix| arg.starts_with(prefix))
            || (subcommand == "tag" && arg.starts_with("-n"))
    };
    match subcommand {
        "branch" | "tag" => {
            let lists_patterns = args.iter().any(|arg| matches!(*arg, "-l" | "--list"))
                || (subcommand == "tag" && args.iter().any(|arg| arg.starts_with("-n")));
            let mut takes_value = false;
            for arg in args {
                if takes_value {
                    takes_value = false;
                    continue;
                }
                if arg.starts_with('-') {
                    if !listing_flag(arg) {
                        return false;
                    }
                    takes_value = VALUE_FLAGS.contains(arg);
                } else if !lists_patterns {
                    return false;
                }
            }
            true
        }
        "remote" => match args {
            [] | ["-v"] | ["--verbose"] => true,
            ["get-url", rest @ ..] => rest
                .iter()
                .all(|arg| matches!(*arg, "--all" | "--push") || !arg.starts_with('-')),
            _ => false,
        },
        _ => true,
    }
}

/// How long a confined-tier process may run.
pub(crate) const CONFINED_TIER_TIMEOUT: Duration = Duration::from_secs(300);

/// A confined-tier process that ran: its output, the directory it ran in,
/// and the paths it read beyond that (Git's metadata).
pub(crate) struct ConfinedRun {
    pub output: fs_utils::CommandOutput,
    pub dir: String,
    pub adapter_read_paths: Vec<String>,
}

/// Run `command` confined with `access`, in `working_dir` (default: the
/// backend's workspace root), able to read `read_paths` too.
pub(crate) async fn run_confined(
    backend: &SharedExecutionBackend,
    command: &str,
    working_dir: Option<&str>,
    read_paths: &[String],
    access: ConfinedAccess,
) -> anyhow::Result<ConfinedRun> {
    let command = command.trim();
    let dir = match working_dir {
        Some(dir) => backend.resolve_path(dir).await?,
        None => backend.workspace_root().clone(),
    };

    // Fail early with an actionable workspace error: a Git command run from
    // an accidental daemon launch directory (often `/` for a service) exits
    // 128 with a misleading message.
    let is_git_command = command == "git" || command.starts_with("git ");
    let mut reads = read_paths.to_vec();
    let mut adapter_read_paths = Vec::new();
    if is_git_command {
        let preflight =
            fs_utils::run_cmd_backend(&**backend, "git rev-parse --show-toplevel", Some(&dir), 5)
                .await?;
        if preflight.exit_code != 0 {
            anyhow::bail!(
                "Git preflight failed: execution directory '{}' is not inside a repository. \
                 Configure execution.workspace_root or pass working_dir explicitly.",
                dir
            );
        }
        let metadata = fs_utils::run_cmd_backend(
            &**backend,
            "git rev-parse --path-format=absolute --git-dir --git-common-dir",
            Some(&dir),
            5,
        )
        .await?;
        if metadata.exit_code != 0 {
            anyhow::bail!(
                "Git metadata discovery failed for execution directory '{}'.",
                dir
            );
        }
        for path in metadata.stdout.lines().map(str::trim) {
            if path.starts_with('/') && !adapter_read_paths.iter().any(|seen| seen == path) {
                adapter_read_paths.push(path.to_string());
            }
        }
        reads.extend(adapter_read_paths.iter().cloned());
    }

    // `/usr/bin/git` is an xcrun shim on macOS. Resolving the concrete
    // developer-tool executable before confinement keeps the shim from
    // attempting a cache write during a read-only observation.
    #[cfg(target_os = "macos")]
    let executable_command = if is_git_command {
        let resolved =
            fs_utils::run_cmd_backend(&**backend, "xcrun --find git", Some(&dir), 5).await?;
        if resolved.exit_code != 0 {
            anyhow::bail!("Could not resolve the macOS Git executable before confinement.");
        }
        let git_path = resolved.stdout.trim();
        if !git_path.starts_with('/') || git_path.contains(['\n', '\r']) {
            anyhow::bail!("macOS returned an invalid Git executable path.");
        }
        format!(
            "'{}'{}",
            git_path.replace('\'', "'\"'\"'"),
            command.strip_prefix("git").unwrap_or_default()
        )
    } else {
        command.to_string()
    };
    #[cfg(not(target_os = "macos"))]
    let executable_command = command.to_string();

    let writes = match access {
        ConfinedAccess::WorkspaceWrite => vec![dir.to_string()],
        ConfinedAccess::ReadOnly => Vec::new(),
    };
    let started = Instant::now();
    let mut request = super::terminal::confined_terminal_execution_request(
        backend,
        &executable_command,
        Some(dir.as_str()),
        &reads,
        &writes,
    )
    .await?;
    if access == ConfinedAccess::ReadOnly {
        // Git inspection refreshes its index opportunistically; disable
        // optional locks so it stays within the read-only sandbox.
        request
            .env
            .insert("GIT_OPTIONAL_LOCKS".to_string(), "0".to_string());
        if is_git_command {
            request
                .env
                .insert("GIT_CONFIG_GLOBAL".to_string(), "/dev/null".to_string());
            request
                .env
                .insert("GIT_CONFIG_NOSYSTEM".to_string(), "1".to_string());
        }
    }
    let execution = backend.execute(request, CONFINED_TIER_TIMEOUT).await?;
    if execution.timed_out {
        anyhow::bail!(
            "Command timed out after {}s",
            CONFINED_TIER_TIMEOUT.as_secs()
        );
    }
    Ok(ConfinedRun {
        output: fs_utils::CommandOutput {
            exit_code: execution.exit_code,
            stdout: execution.stdout_lossy(),
            stderr: execution.stderr_lossy(),
            duration_ms: started.elapsed().as_millis() as u64,
        },
        dir: dir.to_string(),
        adapter_read_paths,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The allowlist as a table: test runners (whose job is running repo
    /// code), installs, scripts and environment dumps are out; `git
    /// branch`/`tag`/`remote` run only in their listing forms; builds and
    /// linters write their working directory, everything else only reads.
    #[test]
    fn allowlist_table() {
        use ConfinedAccess::{ReadOnly, WorkspaceWrite};
        let cases: &[(&str, Option<ConfinedAccess>)] = &[
            ("cargo build", Some(WorkspaceWrite)),
            ("cargo build --release", Some(WorkspaceWrite)),
            ("cargo clippy", Some(WorkspaceWrite)),
            ("go vet ./...", Some(WorkspaceWrite)),
            ("npm ls", Some(ReadOnly)),
            ("git status", Some(ReadOnly)),
            ("git log --oneline", Some(ReadOnly)),
            ("ls -la", Some(ReadOnly)),
            ("printf synthetic-output", Some(ReadOnly)),
            ("/usr/bin/printf synthetic-output", Some(ReadOnly)),
            ("true", Some(ReadOnly)),
            ("/usr/bin/false", Some(ReadOnly)),
            ("cargo test", None),
            ("cargo test --release", None),
            ("npm test", None),
            ("yarn test", None),
            ("bun test", None),
            ("pytest tests/", None),
            ("python3 -m pytest", None),
            ("go test ./...", None),
            ("jest", None),
            ("vitest run", None),
            ("env", None),
            ("printenv", None),
            ("printenv HOME", None),
            ("rm -rf /", None),
            ("curl http://example.invalid", None),
            ("git push", None),
            ("git reset --hard", None),
            ("sudo apt install", None),
            ("chmod 777 /etc", None),
            ("cargo run", None),
            ("cargo run --release -- --evil", None),
            ("cargo bench", None),
            ("npm run build", None),
            ("npm install tailwindcss", None),
            ("npx some-package", None),
            ("yarn run dev", None),
            ("bun run start", None),
            ("go generate ./...", None),
            ("make", None),
            ("make install", None),
            ("cmake --build .", None),
            ("gradle build", None),
            ("mvn package", None),
            ("ls | grep foo", None),
            (r#"ls "$(id)""#, None),
            ("cargo check &", None),
            ("nohup ls", None),
            ("git branch", Some(ReadOnly)),
            ("git branch -a", Some(ReadOnly)),
            ("git branch -vv --sort=-committerdate", Some(ReadOnly)),
            ("git branch --show-current", Some(ReadOnly)),
            ("git branch --contains abc123", Some(ReadOnly)),
            ("git branch --list feature/*", Some(ReadOnly)),
            ("git branch new-feature", None),
            ("git branch -d old", None),
            ("git branch -D old", None),
            ("git branch -m old new", None),
            ("git branch --set-upstream-to=origin/main", None),
            ("git tag", Some(ReadOnly)),
            ("git tag -l v1.*", Some(ReadOnly)),
            ("git tag -n5", Some(ReadOnly)),
            ("git tag --points-at HEAD", Some(ReadOnly)),
            ("git tag v9.9.9", None),
            ("git tag -d v1.0.0", None),
            ("git tag -a v1 -m msg", None),
            ("git remote", Some(ReadOnly)),
            ("git remote -v", Some(ReadOnly)),
            ("git remote get-url origin", Some(ReadOnly)),
            ("git remote add evil https://example.invalid/x", None),
            ("git remote remove origin", None),
            ("git remote set-url origin https://example.invalid/x", None),
        ];
        for (command, access) in cases {
            assert_eq!(confined_access(command), *access, "{command}");
        }
    }

    #[tokio::test]
    async fn a_git_command_runs_in_the_selected_repository() {
        let dir = tempfile::tempdir().unwrap();
        let initialized = std::process::Command::new("git")
            .arg("init")
            .arg("--quiet")
            .arg(dir.path())
            .status()
            .unwrap();
        assert!(initialized.success());
        let run = run_confined(
            &aidaemon_tools::testing::active_execution_backend(),
            "git status --short --branch",
            dir.path().to_str(),
            &[],
            ConfinedAccess::ReadOnly,
        )
        .await
        .unwrap();
        assert_eq!(run.output.exit_code, 0, "{}", run.output.stderr);
        assert!(run.output.stdout.contains("## "), "{}", run.output.stdout);
    }

    #[tokio::test]
    async fn a_git_command_outside_a_repository_says_so() {
        let dir = tempfile::tempdir().unwrap();
        let error = run_confined(
            &aidaemon_tools::testing::active_execution_backend(),
            "git status",
            dir.path().to_str(),
            &[],
            ConfinedAccess::ReadOnly,
        )
        .await
        .err()
        .expect("not a repository")
        .to_string();
        assert!(error.contains("Git preflight failed"), "{error}");
        assert!(error.contains("execution.workspace_root"), "{error}");
    }
}
