//! Confined runs: the macOS Seatbelt policy, the Linux sandbox adapters,
//! runtime support grants and write-root preparation.

use super::*;

/// Elevated capabilities a confined run was granted beyond filesystem scope.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ConfinedCapabilities {
    pub(crate) network: bool,
    /// The run was not shown to the owner (the confined auto-approve tier):
    /// inside its write roots, the places whose content runs later (git
    /// hooks and configuration, `.husky`) stay unwritable.
    pub(crate) protect_persistence: bool,
}

/// Paths inside a write root that an unattended confined run may not write:
/// what is there runs later on its own (git hooks, git configuration that
/// names helpers and hooks, husky hooks). Seatbelt denies them by rule;
/// bubblewrap mounts them read-only over the writable root; the Codex
/// adapter gets them as read entries nested in the write grant.
pub(super) const PERSISTENCE_SUBPATHS: &[&str] = &[".git/hooks", ".git/config", ".husky"];

pub(crate) async fn confined_terminal_execution_request(
    backend: &SharedExecutionBackend,
    command: &str,
    working_dir: Option<&str>,
    read_paths: &[String],
    write_paths: &[String],
) -> anyhow::Result<ExecutionRequest> {
    confined_terminal_execution_request_inner(
        backend,
        command,
        false,
        working_dir,
        read_paths,
        write_paths,
        &[],
        ConfinedCapabilities {
            protect_persistence: true,
            ..ConfinedCapabilities::default()
        },
    )
    .await
}

pub(super) async fn confined_terminal_script_execution_request(
    backend: &SharedExecutionBackend,
    script: &str,
    working_dir: Option<&str>,
    read_paths: &[String],
    write_paths: &[String],
    write_roots: &[String],
    capabilities: ConfinedCapabilities,
) -> anyhow::Result<ExecutionRequest> {
    confined_terminal_execution_request_inner(
        backend,
        script,
        true,
        working_dir,
        read_paths,
        write_paths,
        write_roots,
        capabilities,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
#[allow(unreachable_code)]
pub(super) async fn confined_terminal_execution_request_inner(
    backend: &SharedExecutionBackend,
    shell_source: &str,
    script_via_stdin: bool,
    working_dir: Option<&str>,
    read_paths: &[String],
    write_paths: &[String],
    write_roots: &[String],
    capabilities: ConfinedCapabilities,
) -> anyhow::Result<ExecutionRequest> {
    let resolved_cwd = match working_dir {
        Some(path) => {
            let resolved = backend.resolve_path(path).await?;
            backend.canonicalize(&resolved).await.unwrap_or(resolved)
        }
        None => {
            let resolved = backend.workspace_root().clone();
            backend.canonicalize(&resolved).await.unwrap_or(resolved)
        }
    };
    let cwd = Some(resolved_cwd);
    let mut reads = Vec::new();
    for path in read_paths {
        let path = canonicalize_access_path(
            backend,
            resolve_access_path(backend, path, cwd.as_ref()).await?,
        )
        .await
        .to_string();
        if !reads.contains(&path) {
            reads.push(path);
        }
    }
    let mut resolved_write_paths = Vec::new();
    for path in write_paths {
        let path = canonicalize_access_path(
            backend,
            resolve_access_path(backend, path, cwd.as_ref()).await?,
        )
        .await
        .to_string();
        if !resolved_write_paths.contains(&path) {
            resolved_write_paths.push(path);
        }
    }
    let mut resolved_write_roots = Vec::new();
    for path in write_roots {
        let path = canonicalize_access_path(
            backend,
            resolve_access_path(backend, path, cwd.as_ref()).await?,
        )
        .await
        .to_string();
        if !resolved_write_roots.contains(&path) {
            resolved_write_roots.push(path);
        }
    }
    let mut writes = resolved_write_paths.clone();
    for path in &resolved_write_roots {
        if !writes.contains(path) {
            writes.push(path.clone());
        }
    }

    // Every confined command gets a private, per-invocation writable scratch
    // directory, even a read-only one. Build and package tools (npm, npx,
    // cargo, pip, wrangler, vite) must write a cache/temp somewhere to run at
    // all; without an authorized scratch they fail opaquely, and the agent's
    // own recovery move (setting a cache path) is then rejected by the scope
    // lock. This is least-privilege — a fresh dir under the daemon scratch
    // root, never the host's global /tmp — and it is exported as TMPDIR/cache
    // below so no command has to discover or declare it.
    let managed_scratch = provision_managed_scratch(backend).await;
    if let Some(scratch) = managed_scratch.as_ref()
        && !writes.contains(scratch)
    {
        writes.push(scratch.clone());
    }

    // Task data authority and executable runtime support are separate lanes.
    // A native sandbox still needs to execute owner-installed toolchains, but
    // granting their entire home directory would expose unrelated data and
    // credentials. Resolve exact machine invocations and add only registered,
    // read-only runtime roots/caches.
    let runtime_support = native_sandbox_runtime_support(backend, shell_source).await?;
    let mut runtime_support = runtime_support;
    add_manifest_runtime_environment(
        backend,
        &resolved_write_paths,
        &resolved_write_roots,
        managed_scratch.as_deref(),
        script_via_stdin,
        &mut runtime_support,
    )
    .await;

    // Linux confines with the Codex sandbox adapter when it is installed,
    // else with the SDK's scoped bubblewrap profile (`[terminal]
    // confinement = "sandbox"` must never fail just for a missing adapter).
    #[cfg(not(target_os = "macos"))]
    let adapter = {
        let codex = backend.resolve_executable("codex").await?;
        let bubblewrap = if codex.is_none() {
            usable_bubblewrap(backend).await?
        } else {
            None
        };
        select_linux_sandbox_adapter(codex, bubblewrap)?
    };
    // The Codex adapter is a Node.js entrypoint (`#!/usr/bin/env
    // node`). Its interpreter is not necessarily in a base system directory
    // on hosted runners, and the confined child PATH is intentionally reduced
    // to granted runtime prefixes. Grant the adapter's own executable prefix
    // so the outer adapter process can resolve its interpreter without
    // widening access to unrelated host paths. (bwrap is a native binary
    // that runs outside its own sandbox: it needs no grant.)
    #[cfg(not(target_os = "macos"))]
    if let LinuxSandboxAdapter::Codex(codex) = &adapter {
        if let Some(parent) = codex.parent() {
            runtime_support.add_read(parent.to_string());
            runtime_support.add_path_prefix(parent.to_string());
            runtime_support.add_executable(parent.to_string());
        }
        let canonical = backend
            .canonicalize(codex)
            .await
            .unwrap_or_else(|_| codex.clone());
        if let Some(parent) = canonical.parent() {
            runtime_support.add_read(parent.to_string());
            runtime_support.add_path_prefix(parent.to_string());
            runtime_support.add_executable(parent.to_string());
        }
    }
    for path in &runtime_support.read_paths {
        if !reads.contains(path) && !writes.contains(path) {
            reads.push(path.clone());
        }
    }

    let mut sandbox_environment = std::collections::BTreeMap::new();
    if !runtime_support.path_prefixes.is_empty() {
        sandbox_environment.insert(
            "PATH".to_string(),
            native_sandbox_search_path(
                &runtime_support.path_prefixes,
                &runtime_support.executable_paths,
            ),
        );
    }
    sandbox_environment.extend(runtime_support.environment);

    #[cfg(target_os = "macos")]
    {
        return macos_manifest_sandbox_request(
            shell_source,
            script_via_stdin,
            cwd,
            &reads,
            &writes,
            &runtime_support.executable_paths,
            sandbox_environment,
            capabilities,
        );
    }

    #[cfg(not(target_os = "macos"))]
    let sandbox_cwd = cwd
        .as_ref()
        .map(ToString::to_string)
        .unwrap_or_else(|| "/".to_string());
    // Landlock-backed Linux sandboxes cannot install a write rule for a path
    // that does not exist yet: the kernel needs the containing directory as a
    // writable anchor before the command can create the declared future leaf.
    // Keep the typed leaf/root grant above as the authority boundary, and add
    // only each missing target's immediate parent to the adapter profile. The
    // parent is already the command's working or runtime-created chain; this
    // projection makes the same future-target contract work on Linux as it
    // does with seatbelt on macOS.
    #[cfg(not(target_os = "macos"))]
    let mut policy_writes = writes.clone();
    #[cfg(not(target_os = "macos"))]
    for target in resolved_write_paths
        .iter()
        .chain(resolved_write_roots.iter())
    {
        let target_path = std::path::Path::new(target);
        if backend
            .metadata(&aidaemon_tools::BackendPath::new(target.clone()))
            .await
            .is_err()
            && let Some(parent) = target_path.parent()
        {
            let parent = parent.to_string_lossy().to_string();
            if !policy_writes.contains(&parent) {
                policy_writes.push(parent);
            }
        }
    }
    #[cfg(target_os = "macos")]
    let policy_writes = writes;
    #[cfg(not(target_os = "macos"))]
    {
        // The sandbox is a process boundary. Supplying these values only to
        // the outer adapter does not guarantee that its child keeps them, so
        // carry the registered runtime profile explicitly into the confined
        // process as well. These are runtime dependencies, not task
        // authority or ambient owner configuration.
        let mut confined = vec!["/usr/bin/env".to_string()];
        confined.extend(
            sandbox_environment
                .iter()
                .map(|(name, value)| format!("{name}={value}")),
        );
        if script_via_stdin {
            confined.extend(script_shell_argv());
        } else {
            confined.extend([
                "/bin/sh".to_string(),
                "-c".to_string(),
                shell_source.to_string(),
            ]);
        }
        let (program, args) = match &adapter {
            LinuxSandboxAdapter::Codex(codex) => {
                let sandbox_state =
                    codex_sandbox_state_json(&sandbox_cwd, &reads, &policy_writes, capabilities)?;
                let mut args = vec![
                    "sandbox".to_string(),
                    "--sandbox-state-json".to_string(),
                    sandbox_state,
                    "--".to_string(),
                ];
                args.extend(confined);
                (codex.to_string(), args)
            }
            LinuxSandboxAdapter::Bubblewrap(bwrap) => (
                bwrap.to_string(),
                bubblewrap_sandbox_args(
                    &sandbox_cwd,
                    &reads,
                    &policy_writes,
                    capabilities,
                    &confined,
                ),
            ),
        };
        let mut request = ExecutionRequest::argv(program, args);
        request.cwd = cwd;
        request.env.extend(sandbox_environment);
        if script_via_stdin {
            request.stdin = Some(shell_source.as_bytes().to_vec());
        }
        Ok(request)
    }
}

/// Resolve symlinked existing prefixes while preserving a not-yet-created
/// suffix. A policy for `/tmp/new-root` must protect the kernel path
/// `/private/tmp/new-root`, including before the requested create occurs.
pub(super) async fn canonicalize_access_path(
    backend: &SharedExecutionBackend,
    path: aidaemon_tools::BackendPath,
) -> aidaemon_tools::BackendPath {
    if let Ok(canonical) = backend.canonicalize(&path).await {
        return canonical;
    }
    let mut candidate = path.clone();
    let mut suffix = Vec::new();
    while let Some(name) = candidate.file_name().map(str::to_string) {
        suffix.push(name);
        let Some(parent) = candidate.parent() else {
            return path;
        };
        candidate = parent;
        if let Ok(mut canonical) = backend.canonicalize(&candidate).await {
            for component in suffix.iter().rev() {
                canonical = canonical.join(component);
            }
            return canonical;
        }
    }
    path
}

#[cfg(target_os = "macos")]
pub(super) fn validated_seatbelt_path(path: &str) -> anyhow::Result<String> {
    anyhow::ensure!(
        path.starts_with('/')
            && !path
                .chars()
                .any(|character| matches!(character, '\n' | '\r' | '\0')),
        "sandbox manifest paths must be absolute and single-line"
    );
    let escaped = path.replace('\\', "\\\\").replace('"', "\\\"");
    Ok(escaped)
}

#[cfg(target_os = "macos")]
pub(super) fn seatbelt_ancestor_clause(path: &str) -> Option<String> {
    // Seatbelt rejects `(path-ancestors "/")` while the root literal itself
    // is valid. Root has no ancestors, so omitting the empty relationship is
    // both the accurate capability model and valid policy syntax.
    (path != "/").then(|| format!("(path-ancestors \"{path}\")"))
}

/// `<ancestor>/node_modules` for every strict ancestor of `cwd`, in npm's
/// PATH order (nearest first). These are lookup targets only.
#[cfg(target_os = "macos")]
pub(super) fn npm_ancestor_bin_lookup_paths(cwd: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = std::path::Path::new(cwd).parent();
    while let Some(dir) = current {
        let name = dir.to_string_lossy();
        let name = name.trim_end_matches('/');
        out.push(format!("{name}/node_modules"));
        current = dir.parent();
    }
    out
}

#[cfg(target_os = "macos")]
pub(super) fn macos_manifest_sandbox_policy(
    traversal_cwd: Option<&str>,
    read_paths: &[String],
    write_paths: &[String],
    executable_paths: &[String],
    script_via_stdin: bool,
    capabilities: ConfinedCapabilities,
) -> anyhow::Result<String> {
    let mut policy = include_str!("macos_terminal_sandbox.sbpl").to_string();
    if capabilities.network {
        // Granted through the terminal approval ladder (or a persisted
        // capability grant), never by default. `system-socket` covers the
        // resolver's control socket; DNS itself goes through mDNSResponder,
        // which the base policy already allows a mach lookup for.
        policy.push_str(
            "\n;; Elevated capability granted for this run: outbound network.\n(allow network-outbound)\n(allow system-socket)\n",
        );
    }
    if let Some(path) = traversal_cwd {
        let path = validated_seatbelt_path(path)?;
        // Execution cwd is location, not implicit authority to read that tree.
        // Grant only the directory objects needed by getcwd/path traversal;
        // child file contents still require an explicit read/write manifest.
        let clauses = std::iter::once(format!("(literal \"{path}\")"))
            .chain(seatbelt_ancestor_clause(&path))
            .collect::<Vec<_>>()
            .join(" ");
        policy.push_str(&format!(
            "\n(allow file-read* file-test-existence\n  {clauses})\n"
        ));
        // npm's run-script prepends `<ancestor>/node_modules/.bin` for every
        // ancestor of the project to the child PATH and then spawns a bare
        // `sh`. execvp(3) keeps walking PATH on ENOENT/EACCES but aborts on
        // any other errno, and seatbelt reports an undeclared path as EPERM —
        // so every `npx`/`npm run` under $HOME died with `spawn EPERM` before
        // printing anything. Metadata-only grants on exactly those entries
        // turn the denial into ENOENT; sibling file contents stay unreadable.
        let bin_clauses = npm_ancestor_bin_lookup_paths(&path)
            .into_iter()
            .map(|dir| format!("(literal \"{dir}\") (subpath \"{dir}/.bin\")"))
            .collect::<Vec<_>>()
            .join("\n  ");
        if !bin_clauses.is_empty() {
            policy.push_str(&format!(
                "\n(allow file-read-metadata file-test-existence\n  {bin_clauses})\n"
            ));
        }
    }
    if !read_paths.is_empty() {
        let paths = read_paths
            .iter()
            .map(|path| validated_seatbelt_path(path))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let data_clauses = paths
            .iter()
            .map(|path| format!("(literal \"{path}\") (subpath \"{path}\")"))
            .collect::<Vec<_>>()
            .join("\n  ");
        let ancestor_clauses = paths
            .iter()
            .filter_map(|path| seatbelt_ancestor_clause(path))
            .collect::<Vec<_>>()
            .join("\n  ");
        // `getcwd(3)` and normal path lookup walk and read containing
        // directories. This exposes ancestor entry names, but never sibling
        // file contents; exact/subpath data access remains bound to the
        // declared target.
        policy.push_str(&format!(
            "\n(allow file-read* file-test-existence\n  {data_clauses})\n"
        ));
        if !ancestor_clauses.is_empty() {
            policy.push_str(&format!(
                "(allow file-read* file-test-existence\n  {ancestor_clauses})\n"
            ));
        }
    }
    if !write_paths.is_empty() {
        let paths = write_paths
            .iter()
            .map(|path| validated_seatbelt_path(path))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let data_clauses = paths
            .iter()
            .map(|path| format!("(literal \"{path}\") (subpath \"{path}\")"))
            .collect::<Vec<_>>()
            .join("\n  ");
        let ancestor_clauses = paths
            .iter()
            .filter_map(|path| seatbelt_ancestor_clause(path))
            .collect::<Vec<_>>()
            .join("\n  ");
        // A write grant also permits reading the exact output tree so one
        // process can create, validate, and clean it without widening scope.
        // Ancestors receive read/traversal access only. Seatbelt requires
        // directory lookup on the containing path to create a future child;
        // granting write operations to every ancestor would unnecessarily
        // permit mutation of the containing directory itself.
        policy.push_str(&format!(
            "\n(allow file-read* file-test-existence file-write*\n  {data_clauses})\n"
        ));
        if capabilities.protect_persistence {
            // Seatbelt applies the last matching rule, so this deny wins
            // over the write grant above.
            let deny_clauses = paths
                .iter()
                .flat_map(|path| {
                    PERSISTENCE_SUBPATHS.iter().map(move |sub| {
                        format!("(literal \"{path}/{sub}\") (subpath \"{path}/{sub}\")")
                    })
                })
                .collect::<Vec<_>>()
                .join("\n  ");
            policy.push_str(&format!("\n(deny file-write*\n  {deny_clauses})\n"));
        }
        if !ancestor_clauses.is_empty() {
            policy.push_str(&format!(
                "(allow file-read* file-test-existence\n  {ancestor_clauses})\n"
            ));
        }
    }
    if !executable_paths.is_empty() {
        let paths = executable_paths
            .iter()
            .map(|path| validated_seatbelt_path(path))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let clauses = paths
            .iter()
            .map(|path| format!("(literal \"{path}\") (subpath \"{path}\")"))
            .collect::<Vec<_>>()
            .join("\n  ");
        // Runtime mapping is a separate capability from task-data reads. This
        // lets Homebrew/Python/Rust toolchains load their immutable binaries
        // and frameworks without making an arbitrary user read target
        // executable or widening task mutation authority.
        policy.push_str(&format!("\n(allow file-map-executable\n  {clauses})\n"));
    }
    if script_via_stdin {
        // Darwin's /bin/sh (bash) materializes here-documents in the
        // platform temporary directory and does not consistently honor the
        // task's TMPDIR for that parser-internal file.  Permit only the
        // shell's transient `sh-thd.*` names in that one OS-managed directory
        // — never the directory as a whole — so a script can use normal shell
        // syntax without reopening ambient temporary write authority.
        let mut temp_dirs = vec![std::env::temp_dir()];
        if let Ok(canonical) = dunce::canonicalize(&temp_dirs[0])
            && !temp_dirs.iter().any(|path| path == &canonical)
        {
            temp_dirs.push(canonical);
        }
        // `/tmp` and `/var/tmp` are symlink/alias spellings on Darwin, and
        // shells launched by different service managers can inherit either
        // spelling. Keep the aliases in the same narrowly scoped transient
        // rule rather than granting a general temporary-tree write.
        for path in ["/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp"] {
            let path = PathBuf::from(path);
            if !temp_dirs.iter().any(|existing| existing == &path) {
                temp_dirs.push(path);
            }
        }
        // A canonical temp path may have a lexical `/private` counterpart (or
        // vice versa) when the daemon and shell resolve symlinks differently.
        // Add only that paired spelling; the transient filename remains the
        // final constrained component.
        let alias_dirs = temp_dirs
            .iter()
            .filter_map(|path| {
                let value = path.to_string_lossy();
                if let Some(rest) = value.strip_prefix("/private/") {
                    Some(PathBuf::from(format!("/{rest}")))
                } else if value.starts_with('/') {
                    Some(PathBuf::from(format!("/private{value}")))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        for path in alias_dirs {
            if !temp_dirs.iter().any(|existing| existing == &path) {
                temp_dirs.push(path);
            }
        }
        let patterns = temp_dirs
            .into_iter()
            .map(|path| {
                let escaped = regex::escape(path.to_string_lossy().trim_end_matches('/'));
                // Different /bin/sh builds place the parser temporary either
                // directly in TMPDIR or below one of a few implementation
                // directories. Enumerate a small structural depth while
                // constraining the leaf to the shell's transient prefix;
                // never grant the whole temporary tree.
                (0..=3)
                    .map(|depth| {
                        let components = if depth == 0 {
                            String::new()
                        } else {
                            "[^/]*/".repeat(depth)
                        };
                        format!("(regex #\"^{escaped}/{components}sh-thd.*$\")")
                    })
                    .collect::<Vec<_>>()
                    .join("\n  ")
            })
            .collect::<Vec<_>>();
        if !patterns.is_empty() {
            policy.push_str(&format!(
                "\n(allow file-read* file-test-existence file-write*\n  {})\n",
                patterns.join("\n  ")
            ));
        }
    }
    Ok(policy)
}

#[allow(clippy::too_many_arguments)]
#[cfg(target_os = "macos")]
pub(super) fn macos_manifest_sandbox_request(
    shell_source: &str,
    script_via_stdin: bool,
    cwd: Option<aidaemon_tools::BackendPath>,
    read_paths: &[String],
    write_paths: &[String],
    executable_paths: &[String],
    sandbox_environment: std::collections::BTreeMap<String, String>,
    capabilities: ConfinedCapabilities,
) -> anyhow::Result<ExecutionRequest> {
    let traversal_cwd = cwd.as_ref().map(aidaemon_tools::BackendPath::as_str);
    let policy = macos_manifest_sandbox_policy(
        traversal_cwd,
        read_paths,
        write_paths,
        executable_paths,
        script_via_stdin,
        capabilities,
    )?;
    let mut args = vec![
        "-p".to_string(),
        policy,
        "--".to_string(),
        "/usr/bin/env".to_string(),
    ];
    args.extend(
        sandbox_environment
            .iter()
            .map(|(name, value)| format!("{name}={value}")),
    );
    if script_via_stdin {
        args.extend(script_shell_argv());
    } else {
        args.extend([
            "/bin/sh".to_string(),
            "-c".to_string(),
            shell_source.to_string(),
        ]);
    }
    let mut request = ExecutionRequest::argv("/usr/bin/sandbox-exec", args);
    request.cwd = cwd;
    request.env.extend(sandbox_environment);
    if script_via_stdin {
        request.stdin = Some(shell_source.as_bytes().to_vec());
    }
    Ok(request)
}

#[derive(Debug, Default)]
pub(super) struct NativeSandboxRuntimeSupport {
    pub(super) read_paths: Vec<String>,
    /// Immutable runtime roots that may be mapped as executable by the native
    /// loader. This is deliberately separate from task-data reads: a user
    /// supplied read path must never become an executable mapping authority.
    pub(super) executable_paths: Vec<String>,
    pub(super) path_prefixes: Vec<String>,
    pub(super) environment: HashMap<String, String>,
    /// Python's macOS pip uses this explicit variable instead of
    /// XDG_CACHE_HOME. It is a capability discovered from the resolved
    /// executable, not a request/prose classifier.
    pub(super) python_cache: bool,
}

impl NativeSandboxRuntimeSupport {
    pub(super) fn add_read(&mut self, path: impl Into<String>) {
        let path = path.into();
        if !path.trim().is_empty() && !self.read_paths.contains(&path) {
            self.read_paths.push(path);
        }
    }

    pub(super) fn add_path_prefix(&mut self, path: impl Into<String>) {
        let path = path.into();
        if !path.trim().is_empty() && !self.path_prefixes.contains(&path) {
            self.path_prefixes.push(path);
        }
    }

    pub(super) fn add_executable(&mut self, path: impl Into<String>) {
        let path = path.into();
        if !path.trim().is_empty() && !self.executable_paths.contains(&path) {
            self.executable_paths.push(path);
        }
    }

    pub(super) fn prefer_path_prefix(&mut self, path: impl Into<String>) {
        let path = path.into();
        self.path_prefixes.retain(|existing| existing != &path);
        if !path.trim().is_empty() {
            self.path_prefixes.insert(0, path);
        }
    }
}

pub(super) fn parsed_command_programs(command: &str) -> Vec<String> {
    let mut programs = Vec::new();
    for invocation in aidaemon_tools::command_risk::structural_shell_invocations(command) {
        if !programs.contains(&invocation.program) {
            programs.push(invocation.program);
        }
    }
    programs
}

/// Return the immutable installation root that owns an executable managed by a
/// recognized toolchain/package layout. Runtime support is granted read-only
/// to this root; it never becomes task-data authority.
pub(super) fn managed_executable_runtime_root(
    executable: &aidaemon_tools::BackendPath,
) -> Option<String> {
    let path = PathBuf::from(executable.as_str());
    let components = path
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(value) => value.to_str().map(str::to_string),
            _ => None,
        })
        .collect::<Vec<_>>();

    let root_depth = components
        .windows(1)
        .position(|parts| parts[0] == "Cellar")
        .map(|index| index + 3)
        .or_else(|| {
            components
                .windows(2)
                .position(|parts| parts == [".rustup", "toolchains"])
                .map(|index| index + 3)
        })
        .or_else(|| {
            components
                .windows(3)
                .position(|parts| parts == [".nvm", "versions", "node"])
                .map(|index| index + 4)
        })
        .or_else(|| {
            components
                .windows(2)
                .position(|parts| parts == [".pyenv", "versions"])
                .map(|index| index + 3)
        })?;
    if components.len() <= root_depth {
        return None;
    }

    let mut root = PathBuf::from("/");
    for component in components.iter().take(root_depth) {
        root.push(component);
    }
    Some(root.to_string_lossy().into_owned())
}

pub(super) fn homebrew_stable_runtime_alias(
    executable: &aidaemon_tools::BackendPath,
) -> Option<String> {
    let path = PathBuf::from(executable.as_str());
    let components = path
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(value) => value.to_str().map(str::to_string),
            _ => None,
        })
        .collect::<Vec<_>>();
    let cellar_index = components.iter().position(|part| part == "Cellar")?;
    let formula = components.get(cellar_index + 1)?;
    let mut alias = PathBuf::from("/");
    for component in components.iter().take(cellar_index) {
        alias.push(component);
    }
    alias.push("opt");
    alias.push(formula);
    Some(alias.to_string_lossy().into_owned())
}

/// macOS sandbox path rules canonicalize a Homebrew formula symlink before
/// applying an exact grant. A process which later opens the stable
/// `/opt/homebrew/opt/<formula>` spelling therefore also needs read access to
/// the alias namespace directory, while the canonical Cellar root remains the
/// boundary that authorizes the formula's actual contents. Granting the
/// namespace alone does not expose another formula's Cellar files.
pub(super) fn homebrew_alias_namespace(executable: &aidaemon_tools::BackendPath) -> Option<String> {
    let alias = PathBuf::from(homebrew_stable_runtime_alias(executable)?);
    alias
        .parent()
        .map(|path| path.to_string_lossy().into_owned())
}

/// Ask a selected Python runtime for its own immutable import roots before it
/// enters confinement. This is capability discovery from the resolved
/// executable, not inference from shell prose: the adapter reports exactly
/// which standard-library trees its child process will need.
pub(super) async fn add_python_runtime_support(
    backend: &SharedExecutionBackend,
    executable: &aidaemon_tools::BackendPath,
    support: &mut NativeSandboxRuntimeSupport,
) {
    let probe = r#"import json,sys; print(json.dumps({"base_prefix":sys.base_prefix,"prefix":sys.prefix,"path":sys.path}))"#;
    let Ok(output) = backend
        .execute(
            ExecutionRequest::argv(
                executable.to_string(),
                vec!["-c".to_string(), probe.to_string()],
            ),
            Duration::from_secs(5),
        )
        .await
    else {
        return;
    };
    if output.exit_code != 0 {
        return;
    }
    let Some(profile) = output
        .stdout_lossy()
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<Value>(line).ok())
    else {
        return;
    };

    for path in profile
        .get("path")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|path| aidaemon_tools::BackendPath::is_absolute(path))
    {
        let path = aidaemon_tools::BackendPath::new(path.to_string());
        if backend.metadata(&path).await.is_ok() {
            support.add_read(path.to_string());
            support.add_executable(path.to_string());
        }
    }
    for prefix in ["base_prefix", "prefix"]
        .into_iter()
        .filter_map(|key| profile.get(key).and_then(Value::as_str))
        .filter(|path| aidaemon_tools::BackendPath::is_absolute(path))
    {
        let prefix = aidaemon_tools::BackendPath::new(prefix.to_string());
        for dependency in [prefix.join("lib"), prefix.join("pyvenv.cfg")] {
            if backend.metadata(&dependency).await.is_ok() {
                support.add_read(dependency.to_string());
                if dependency.file_name() == Some("lib") {
                    support.add_executable(dependency.to_string());
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
pub(super) fn macos_developer_root_from_sdk_root(sdk_root: &str) -> Option<String> {
    ["/Platforms/", "/SDKs/"]
        .into_iter()
        .find_map(|marker| {
            sdk_root
                .split_once(marker)
                .map(|(root, _)| root.to_string())
        })
        .filter(|root| !root.is_empty())
}

#[cfg(target_os = "macos")]
pub(super) async fn add_macos_developer_runtime_support(
    backend: &SharedExecutionBackend,
    support: &mut NativeSandboxRuntimeSupport,
) {
    let mut sdk_root = std::env::var("SDKROOT")
        .ok()
        .filter(|path| aidaemon_tools::BackendPath::is_absolute(path));
    if sdk_root.is_none() {
        let xcrun = aidaemon_tools::BackendPath::new("/usr/bin/xcrun");
        if backend.metadata(&xcrun).await.is_ok()
            && let Ok(output) = backend
                .execute(
                    ExecutionRequest::argv(
                        xcrun.to_string(),
                        vec![
                            "--sdk".to_string(),
                            "macosx".to_string(),
                            "--show-sdk-path".to_string(),
                        ],
                    ),
                    Duration::from_secs(5),
                )
                .await
            && output.exit_code == 0
        {
            sdk_root = output
                .stdout_lossy()
                .lines()
                .map(str::trim)
                .find(|path| aidaemon_tools::BackendPath::is_absolute(path))
                .map(str::to_string);
        }
    }
    if sdk_root.is_none() {
        for candidate in [
            "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
        ] {
            if backend
                .metadata(&aidaemon_tools::BackendPath::new(candidate))
                .await
                .is_ok()
            {
                sdk_root = Some(candidate.to_string());
                break;
            }
        }
    }
    if let Some(sdk_root) = sdk_root {
        let sdk_path = aidaemon_tools::BackendPath::new(sdk_root.clone());
        if backend.metadata(&sdk_path).await.is_ok() {
            if let Some(developer_root) = macos_developer_root_from_sdk_root(&sdk_root) {
                support.add_read(developer_root.clone());
                support
                    .environment
                    .insert("DEVELOPER_DIR".to_string(), developer_root);
            }
            support.add_read(sdk_root.clone());
            support.environment.insert("SDKROOT".to_string(), sdk_root);
        }
    }
}

pub(super) async fn native_sandbox_runtime_support(
    backend: &SharedExecutionBackend,
    command: &str,
) -> anyhow::Result<NativeSandboxRuntimeSupport> {
    let mut support = NativeSandboxRuntimeSupport::default();
    let mut programs = parsed_command_programs(command);
    // JavaScript package runners delegate to `node` transitively (their own
    // resolved path is a script under node_modules or a separate native
    // binary), so node's runtime roots would never be granted from the
    // top-level command alone. Resolve the interpreter they run as well.
    let runs_node = programs.iter().any(|program| {
        matches!(
            std::path::Path::new(program)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(program.as_str()),
            "npm" | "npx" | "yarn" | "pnpm" | "bun" | "corepack" | "tsx" | "ts-node" | "vite"
        )
    });
    if runs_node && !programs.iter().any(|program| program == "node") {
        programs.push("node".to_string());
    }
    for program in programs {
        let Some(resolved) = backend.resolve_executable(&program).await? else {
            continue;
        };
        if let Some(parent) = resolved.parent() {
            support.add_read(parent.to_string());
            support.add_path_prefix(parent.to_string());
            support.add_executable(parent.to_string());
        }
        let canonical = backend
            .canonicalize(&resolved)
            .await
            .unwrap_or(resolved.clone());
        if let Some(parent) = canonical.parent() {
            support.add_read(parent.to_string());
            support.add_path_prefix(parent.to_string());
            support.add_executable(parent.to_string());
        }
        if let Some(runtime_root) = managed_executable_runtime_root(&canonical) {
            support.add_read(runtime_root.clone());
            support.add_executable(runtime_root);
        }
        // A conda/miniforge environment is an immutable toolchain prefix: its
        // executables read prefix-relative configuration at startup (for
        // example `<prefix>/ssl/openssl.cnf` for node/python) and load
        // `<prefix>/lib`. Recognize it by its `conda-meta` marker directory
        // and grant the prefix read-only, exactly like a Homebrew Cellar root.
        for location in [resolved.as_str(), canonical.as_str()] {
            if let Some(prefix) = std::path::Path::new(location)
                .parent()
                .and_then(|bin| bin.parent())
                && prefix.join("conda-meta").is_dir()
            {
                let prefix = prefix.to_string_lossy().to_string();
                support.add_read(prefix.clone());
                support.add_executable(prefix);
            }
        }
        // Standard prefix layout: an executable under `<prefix>/bin` loads
        // shared libraries from the sibling `<prefix>/lib` (and helpers from
        // `<prefix>/libexec`). Without them a relocatable runtime such as a
        // conda/miniforge `node` aborts with a missing `@rpath` dylib before
        // it can run. Read-only, library lanes only; no home-directory grant.
        for location in [resolved.as_str(), canonical.as_str()] {
            let location = std::path::Path::new(location);
            if location
                .parent()
                .and_then(|dir| dir.file_name())
                .and_then(|name| name.to_str())
                != Some("bin")
            {
                continue;
            }
            if let Some(prefix) = location.parent().and_then(|dir| dir.parent()) {
                for lane in ["lib", "libexec"] {
                    let path = prefix.join(lane);
                    if path.is_dir() {
                        let path = path.to_string_lossy().to_string();
                        support.add_read(path.clone());
                        support.add_executable(path);
                    }
                }
            }
        }
        if let Some(runtime_alias) = homebrew_stable_runtime_alias(&canonical) {
            let alias_path = aidaemon_tools::BackendPath::new(runtime_alias.clone());
            if backend.metadata(&alias_path).await.is_ok() {
                support.add_read(runtime_alias);
                support.add_executable(alias_path.to_string());
                if let Some(namespace) = homebrew_alias_namespace(&canonical) {
                    support.add_read(namespace);
                }
            }
        }

        let requested_name = std::path::Path::new(&program)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(program.as_str());
        let canonical_name = std::path::Path::new(canonical.as_str())
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if matches!(requested_name, "python" | "python3" | "pip" | "pip3")
            || matches!(canonical_name, "python" | "python3" | "pip" | "pip3")
        {
            support.python_cache = true;
            add_python_runtime_support(backend, &canonical, &mut support).await;
        }
        if canonical_name == "rustup" && requested_name != "rustup" {
            let output = backend
                .execute(
                    ExecutionRequest::argv(
                        canonical.to_string(),
                        vec!["which".to_string(), requested_name.to_string()],
                    ),
                    Duration::from_secs(5),
                )
                .await?;
            if output.exit_code == 0
                && let Some(actual) = output
                    .stdout_lossy()
                    .lines()
                    .map(str::trim)
                    .find(|line| aidaemon_tools::BackendPath::is_absolute(line))
            {
                let actual = aidaemon_tools::BackendPath::new(actual.to_string());
                if let Some(bin_dir) = actual.parent() {
                    support.prefer_path_prefix(bin_dir.to_string());
                    // Rust compiler tools load libraries and target data
                    // from the sibling toolchain tree, not just `bin/`.
                    if let Some(toolchain_root) = bin_dir.parent() {
                        support.add_read(toolchain_root.to_string());
                        support.add_executable(toolchain_root.to_string());
                    } else {
                        support.add_read(bin_dir.to_string());
                        support.add_executable(bin_dir.to_string());
                    }
                }
            }
        }

        // Git aborts (exit 128) when its ordinary configuration files exist
        // but cannot be opened, so a confined `git status` in an authorized
        // workspace fails for a reason unrelated to the task's data authority.
        // Grant only the exact existing config files as runtime support of the
        // selected executable; credentials, hooks, and the home directory stay
        // outside the profile.
        if requested_name == "git" || canonical_name == "git" {
            for config in [
                backend.home_hint().join(".gitconfig"),
                backend.home_hint().join(".config/git/config"),
                backend.home_hint().join(".config/git/ignore"),
                backend.home_hint().join(".config/git/attributes"),
                aidaemon_tools::BackendPath::new("/etc/gitconfig".to_string()),
            ] {
                if backend.metadata(&config).await.is_ok() {
                    support.add_read(config.to_string());
                }
            }
            // Homebrew git consults its own prefix-level gitconfig.
            if let Some(prefix) = std::path::Path::new(canonical.as_str())
                .ancestors()
                .find(|ancestor| ancestor.join("etc/gitconfig").is_file())
            {
                support.add_read(prefix.join("etc/gitconfig").to_string_lossy().to_string());
            }
        }

        // Cargo requires dependency source caches to compile an already
        // resolved project. These roots deliberately exclude credentials and
        // broad home access; network remains denied by the permission profile.
        if requested_name == "cargo" {
            let cargo_home = backend.home_hint().join(".cargo");
            for relative in ["registry", "git", ".global-cache", ".rustc_info.json"] {
                let path = cargo_home.join(relative);
                if backend.metadata(&path).await.is_ok() {
                    support.add_read(path.to_string());
                }
            }

            // Cargo embeds libgit2 and consults Git's ordinary configuration
            // even for local project initialization. The task did not ask to
            // read this file as evidence, but it is an exact, read-only
            // runtime dependency of the selected executable. Keep it in the
            // runtime-support lane and never broaden the grant to the home
            // directory or Cargo credentials.
            for config in [
                backend.home_hint().join(".gitconfig"),
                backend.home_hint().join(".config/git/config"),
            ] {
                if backend.metadata(&config).await.is_ok() {
                    support.add_read(config.to_string());
                }
            }
        }
    }
    // Compiler drivers and system language launchers can resolve through the
    // selected Apple developer tree even when the top-level command is not
    // Cargo. Register it once as platform runtime support for every confined
    // command instead of maintaining per-language exceptions.
    #[cfg(target_os = "macos")]
    add_macos_developer_runtime_support(backend, &mut support).await;
    Ok(support)
}

/// Create a private, per-invocation writable scratch directory under the
/// daemon scratch root and prune abandoned siblings. Returned as a canonical
/// path so the sandbox policy and the child environment agree. Best-effort:
/// returns `None` if the directory cannot be created, in which case the
/// caller falls back to any declared write root.
pub(super) async fn provision_managed_scratch(backend: &SharedExecutionBackend) -> Option<String> {
    let parent = std::env::temp_dir().join("aidaemon-scratch");
    let parent_path = aidaemon_tools::BackendPath::new(parent.to_string_lossy().to_string());
    backend.create_dir_all(&parent_path).await.ok()?;
    // Prune scratch dirs older than 6 hours so a long-lived daemon does not
    // accumulate them. Best-effort; a busy directory is simply skipped.
    if let Ok(entries) = std::fs::read_dir(&parent) {
        let cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(6 * 3600);
        for entry in entries.flatten() {
            if entry
                .metadata()
                .and_then(|meta| meta.modified())
                .is_ok_and(|modified| modified < cutoff)
            {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }
    let dir = parent.join(uuid::Uuid::new_v4().to_string());
    let dir_path = aidaemon_tools::BackendPath::new(dir.to_string_lossy().to_string());
    backend.create_dir_all(&dir_path).await.ok()?;
    Some(
        backend
            .canonicalize(&dir_path)
            .await
            .unwrap_or(dir_path)
            .to_string(),
    )
}

/// Give package/build processes a task-scoped scratch location without
/// reopening the host's global `/tmp`. The location is selected only from an
/// already-declared writable directory in the access manifest; read-only and
/// exact-file calls receive no ambient temporary write authority.
pub(super) async fn add_manifest_runtime_environment(
    backend: &SharedExecutionBackend,
    write_paths: &[String],
    write_roots: &[String],
    managed_scratch: Option<&str>,
    script_via_stdin: bool,
    support: &mut NativeSandboxRuntimeSupport,
) {
    // A root nested under a declared *future* grant is a leaf the command
    // owns; it is neither materialized nor eligible as a scratch lane.
    let topmost_roots = materialized_write_roots(backend, write_roots, write_paths, None).await;
    let mut declared_roots = topmost_roots.iter().collect::<Vec<_>>();
    // The narrowest declared directory capability is the least-privilege
    // scratch lane. A broad existing cwd such as /tmp must not steal TMPDIR
    // from a more specific future output root merely because it already
    // exists before execution.
    declared_roots
        .sort_by_key(|path| std::cmp::Reverse(std::path::Path::new(path).components().count()));
    let preferred_root = declared_roots.first().map(|path| (*path).clone());
    let preferred_is_directory = if let Some(path) = preferred_root.as_ref() {
        backend
            .metadata(&aidaemon_tools::BackendPath::new(path.clone()))
            .await
            .ok()
            .is_some_and(|metadata| metadata.is_dir())
    } else {
        false
    };
    let future_scratch_root =
        preferred_root.is_some() && script_via_stdin && !preferred_is_directory;
    // A declared write root that does not exist yet is the command's future
    // OUTPUT directory (e.g. a heredoc target). Create it before execution so
    // the confined process can write into it. This is independent of which
    // directory backs TMPDIR/caches below.
    if future_scratch_root
        && let Some(path) = preferred_root.as_ref()
        && let Err(error) = backend
            .create_dir_all(&aidaemon_tools::BackendPath::new(path.clone()))
            .await
    {
        tracing::warn!(path = %path, %error, "Failed to prepare declared future write root");
    }
    let mut scratch_root = (preferred_is_directory || future_scratch_root)
        .then(|| preferred_root.clone())
        .flatten();
    if scratch_root.is_none() {
        for path in declared_roots.iter().skip(1) {
            let candidate = aidaemon_tools::BackendPath::new((*path).clone());
            if backend
                .metadata(&candidate)
                .await
                .ok()
                .is_some_and(|metadata| metadata.is_dir())
            {
                scratch_root = Some(candidate.to_string());
                break;
            }
        }
    }
    // Exact write paths may be used only when they are already directories;
    // never create one as a scratch root because it could be a future file.
    let scratch_root = if scratch_root.is_some() {
        scratch_root
    } else {
        let mut existing_paths = write_paths.iter().collect::<Vec<_>>();
        existing_paths
            .sort_by_key(|path| std::cmp::Reverse(std::path::Path::new(path).components().count()));
        let mut selected = None;
        for path in existing_paths {
            let candidate = aidaemon_tools::BackendPath::new(path.clone());
            if backend
                .metadata(&candidate)
                .await
                .ok()
                .is_some_and(|metadata| metadata.is_dir())
            {
                selected = Some(candidate.to_string());
                break;
            }
        }
        selected
    };
    // The private per-invocation managed scratch is preferred for TMPDIR and
    // caches. A declared write lane is the command's OUTPUT directory (e.g.
    // `dist`), which build tools routinely empty — putting TMPDIR or the npm
    // config there means a `vite build`/`rm -rf dist` wipes them mid-run. Only
    // fall back to a declared directory when no managed scratch exists.
    let scratch_root = managed_scratch.map(str::to_string).or(scratch_root);
    let Some(scratch_root) = scratch_root else {
        return;
    };

    // The scratch root is either the managed dir (already created) or an
    // existing declared directory; a declared future root was created above.
    if !std::path::Path::new(&scratch_root).exists() {
        let path = aidaemon_tools::BackendPath::new(scratch_root.clone());
        if let Err(error) = backend.create_dir_all(&path).await {
            tracing::warn!(path = %scratch_root, %error, "Failed to prepare runtime scratch root");
            return;
        }
    }
    // The sandbox policy and the child environment must use the same
    // canonical identity. Temporary roots on macOS can pass through `/var`
    // and `/private/var` aliases; resolving once here prevents the shell from
    // receiving an environment path that differs from the policy path.
    let scratch_path = aidaemon_tools::BackendPath::new(scratch_root);
    let scratch_root = backend
        .canonicalize(&scratch_path)
        .await
        .unwrap_or(scratch_path)
        .to_string();

    for variable in ["TMPDIR", "TMP", "TEMP"] {
        support
            .environment
            .insert(variable.to_string(), scratch_root.clone());
    }
    // XDG-aware build/package tools keep their cache inside the same declared
    // lane instead of falling back to the owner's home directory. This is a
    // capability projection, not a package-specific command rewrite.
    support
        .environment
        .insert("XDG_CACHE_HOME".to_string(), scratch_root.clone());
    // npm/npx (and wrangler, which shells out through npx) use this explicit
    // variable rather than XDG. Keep their cache in the authorized scratch so
    // a package tool never fails trying to write the owner's home cache.
    support
        .environment
        .insert("NPM_CONFIG_CACHE".to_string(), scratch_root.clone());
    // Point npm/npx at a neutral per-run user config inside the scratch dir
    // instead of the owner's `~/.npmrc`. npm reads its user config at startup
    // and aborts (EPERM) when the sandbox denies the home file; more
    // importantly `~/.npmrc` can hold a private-registry auth token that an
    // untrusted build must never read. A neutral empty config removes both the
    // failure and the exposure. Offline builds need no registry auth.
    let neutral_npmrc = format!("{scratch_root}/.aidaemon-npmrc");
    if backend
        .write(
            &aidaemon_tools::BackendPath::new(neutral_npmrc.clone()),
            b"",
            aidaemon_tools::WriteMode::Overwrite,
            true,
        )
        .await
        .is_ok()
    {
        support
            .environment
            .insert("NPM_CONFIG_USERCONFIG".to_string(), neutral_npmrc);
    }
    if support.python_cache {
        support
            .environment
            .insert("PIP_CACHE_DIR".to_string(), scratch_root);
    }
}

/// Run one confined command to completion and return (exit_code, stdout,
/// stderr). Used by the self-heal retry: it rebuilds the sandbox request with
/// additional authorized reads and re-executes, without touching the tracked
/// background-process machinery.
#[allow(clippy::too_many_arguments)]
pub(super) async fn run_confined_once(
    backend: &SharedExecutionBackend,
    command: &str,
    script_via_stdin: bool,
    working_dir: Option<&str>,
    read_paths: &[String],
    write_paths: &[String],
    write_roots: &[String],
    capabilities: ConfinedCapabilities,
    timeout: Duration,
) -> anyhow::Result<(Option<i32>, String, String)> {
    let request = if script_via_stdin {
        confined_terminal_script_execution_request(
            backend,
            command,
            working_dir,
            read_paths,
            write_paths,
            write_roots,
            capabilities,
        )
        .await?
    } else {
        confined_terminal_execution_request_inner(
            backend,
            command,
            false,
            working_dir,
            read_paths,
            write_paths,
            write_roots,
            capabilities,
        )
        .await?
    };
    let output = backend.execute(request, timeout).await?;
    Ok((
        Some(output.exit_code),
        output.stdout_lossy(),
        output.stderr_lossy(),
    ))
}

pub(super) fn native_sandbox_search_path(prefixes: &[String], readable_roots: &[String]) -> String {
    let current = std::env::var("PATH").unwrap_or_default();
    native_sandbox_search_path_from(
        prefixes,
        readable_roots,
        dirs::home_dir().as_deref(),
        &current,
    )
}

/// Compose the confined child's PATH. The daemon's own PATH is inherited only
/// where the sandbox can actually read it: an entry under the owner's home
/// that no runtime/task grant covers contributes nothing but an EPERM that
/// aborts execvp(3)-style lookups (see `npm_ancestor_bin_lookup_paths`).
pub(super) fn native_sandbox_search_path_from(
    prefixes: &[String],
    readable_roots: &[String],
    home: Option<&std::path::Path>,
    inherited_path: &str,
) -> String {
    // Base system directories the seatbelt policy always grants read /
    // test-existence on (see macos_terminal_sandbox.sbpl). These are the only
    // non-manifest directories a confined child may safely resolve executables
    // from. `/opt/homebrew` and `/usr/local` are deliberately NOT here: the
    // base policy does not grant them, so a binary living there cannot be
    // exec'd confined anyway — and keeping them in PATH actively breaks
    // unrelated lookups (see below).
    const BASE_SYSTEM_BIN_DIRS: [&str; 5] =
        ["/usr/bin", "/bin", "/usr/sbin", "/sbin", "/usr/libexec"];

    let mut paths = prefixes
        .iter()
        .map(PathBuf::from)
        .collect::<Vec<std::path::PathBuf>>();
    let readable = readable_roots
        .iter()
        .chain(prefixes.iter())
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let is_base_system_dir = |path: &std::path::Path| {
        BASE_SYSTEM_BIN_DIRS
            .iter()
            .any(|dir| path == std::path::Path::new(dir))
    };
    let _ = home;
    for path in std::env::split_paths(inherited_path) {
        if !path.is_absolute() || paths.contains(&path) {
            continue;
        }
        // Bare-name resolution (execvp(3), as libuv/`sh` do) probes each PATH
        // directory in order. On seatbelt, probing a directory the sandbox does
        // NOT grant returns EPERM, and that EPERM aborts the entire lookup
        // before it can reach a grantable binary later in PATH — so a confined
        // `sh -c` that spawns a bare `sh`/`node`/`cc` fails with `spawn EPERM`
        // and an opaque exit 255, even though `/bin/sh` is readable. The
        // confined PATH must therefore contain ONLY directories the policy
        // grants: a per-invocation read/exec root, or a base system dir.
        // Everything else (ungranted home dirs AND ungranted non-home dirs like
        // /opt/homebrew, /opt/*, /System/Cryptexes, plugin caches) is dropped.
        let granted = readable.iter().any(|root| path.starts_with(root));
        if !granted && !is_base_system_dir(&path) {
            continue;
        }
        paths.push(path);
    }
    for dir in BASE_SYSTEM_BIN_DIRS {
        let path = PathBuf::from(dir);
        if !paths.contains(&path) {
            paths.push(path);
        }
    }
    std::env::join_paths(paths)
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

/// The program that confines a Linux `terminal` command.
#[cfg(any(test, not(target_os = "macos")))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum LinuxSandboxAdapter {
    /// The Codex CLI's `codex sandbox` (its manifest-driven Landlock
    /// adapter): preferred when installed.
    Codex(aidaemon_tools::BackendPath),
    /// `bwrap` with the SDK's scoped profile
    /// ([`aidaemon_tools::BubblewrapSandbox::scoped_args`]): the fallback.
    Bubblewrap(aidaemon_tools::BackendPath),
}

/// Pick the Linux sandbox: the Codex adapter when present, else a usable
/// bubblewrap, else an error naming both (never a silent run on the host).
#[cfg(any(test, not(target_os = "macos")))]
pub(super) fn select_linux_sandbox_adapter(
    codex: Option<aidaemon_tools::BackendPath>,
    bubblewrap: Option<aidaemon_tools::BackendPath>,
) -> anyhow::Result<LinuxSandboxAdapter> {
    match (codex, bubblewrap) {
        (Some(codex), _) => Ok(LinuxSandboxAdapter::Codex(codex)),
        (None, Some(bwrap)) => {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                info!(
                    bwrap = %bwrap,
                    "Codex sandbox adapter not found; [terminal] confinement = \"sandbox\" \
                     confines commands with bubblewrap instead"
                )
            });
            Ok(LinuxSandboxAdapter::Bubblewrap(bwrap))
        }
        (None, None) => anyhow::bail!(
            "confined terminal execution needs a sandbox, and none is available: install the \
             Codex CLI (its `codex sandbox` adapter) or bubblewrap (`bwrap`, with \
             unprivileged user namespaces enabled)"
        ),
    }
}

/// `bwrap`, when it is installed where the backend runs and, on this host,
/// can actually create the namespaces the profile uses (bwrap may be present
/// while user namespaces are disabled or restricted by AppArmor).
#[cfg(not(target_os = "macos"))]
pub(super) async fn usable_bubblewrap(
    backend: &SharedExecutionBackend,
) -> anyhow::Result<Option<aidaemon_tools::BackendPath>> {
    let Some(bwrap) = backend.resolve_executable("bwrap").await? else {
        return Ok(None);
    };
    if backend.kind() == BackendKind::Local
        && !aidaemon_tools::Sandbox::is_available(&aidaemon_tools::BubblewrapSandbox::new())
    {
        return Ok(None);
    }
    Ok(Some(bwrap))
}

/// The scoped bubblewrap argv for the same manifest the Codex adapter gets.
#[cfg(any(test, not(target_os = "macos")))]
pub(super) fn bubblewrap_sandbox_args(
    cwd: &str,
    read_paths: &[String],
    write_paths: &[String],
    capabilities: ConfinedCapabilities,
    confined: &[String],
) -> Vec<String> {
    let mut args = aidaemon_tools::BubblewrapSandbox::scoped_args(
        &aidaemon_tools::BubblewrapScope {
            cwd: cwd.to_string(),
            read_paths: read_paths.to_vec(),
            write_paths: write_paths.to_vec(),
            allow_network: capabilities.network,
        },
        confined,
    );
    if capabilities.protect_persistence {
        // bwrap applies mounts in order, so a read-only bind of each
        // persistence path placed after the writable roots (before the
        // namespace flags) wins over the write grant, like seatbelt's
        // trailing deny. `-try` skips a path that does not exist.
        let protected = persistence_paths(write_paths)
            .into_iter()
            .flat_map(|path| ["--ro-bind-try".to_string(), path.clone(), path]);
        // `--unshare-pid` is the first flag scoped_args emits after its
        // mounts (an exact argv token, not prose).
        let at = args
            .iter()
            .position(|arg| arg == "--unshare-pid")
            .unwrap_or(args.len());
        args.splice(at..at, protected);
    }
    args
}

/// The persistence paths ([`PERSISTENCE_SUBPATHS`]) inside each write root.
#[cfg(any(test, not(target_os = "macos")))]
pub(super) fn persistence_paths(write_paths: &[String]) -> Vec<String> {
    write_paths
        .iter()
        .flat_map(|root| {
            let root = root.trim_end_matches('/');
            PERSISTENCE_SUBPATHS
                .iter()
                .map(move |sub| format!("{root}/{sub}"))
        })
        .collect()
}

#[cfg(any(test, not(target_os = "macos")))]
pub(super) fn codex_sandbox_state_json(
    cwd: &str,
    read_paths: &[String],
    write_paths: &[String],
    capabilities: ConfinedCapabilities,
) -> anyhow::Result<String> {
    let mut filesystem_entries = vec![json!({
        "path": {
            "type": "special",
            "value": { "kind": "minimal" }
        },
        "access": "read"
    })];
    filesystem_entries.extend(read_paths.iter().map(|path| {
        json!({
            "path": { "type": "path", "path": path },
            "access": "read"
        })
    }));
    filesystem_entries.extend(write_paths.iter().map(|path| {
        json!({
            "path": { "type": "path", "path": path },
            "access": "write"
        })
    }));
    if capabilities.protect_persistence {
        // A more specific entry overrides the write grant that contains it.
        // Only paths that exist: Codex's bubblewrap cannot mount under a
        // write root the command has yet to create (like bwrap's `-try`).
        filesystem_entries.extend(
            persistence_paths(write_paths)
                .into_iter()
                .filter(|path| std::path::Path::new(path).exists())
                .map(|path| {
                    json!({
                        "path": { "type": "path", "path": path },
                        "access": "read"
                    })
                }),
        );
    }
    let sandbox_cwd = reqwest::Url::from_file_path(cwd)
        .map_err(|_| anyhow::anyhow!("sandbox working directory must be an absolute local path"))?;
    Ok(serde_json::to_string(&json!({
        "permissionProfile": {
            "type": "managed",
            "file_system": {
                "type": "restricted",
                "entries": filesystem_entries
            },
            "network": if capabilities.network { "enabled" } else { "restricted" }
        },
        "codexLinuxSandboxExe": Value::Null,
        "sandboxCwd": sandbox_cwd.as_str(),
        "useLegacyLandlock": false
    }))?)
}

/// Declared write roots that must be materialized as directories before
/// launch. A root is skipped when it is a strict descendant of another
/// declared write grant (root or exact path) that does not exist yet, or when
/// the same future path is also declared as an exact write path. Such an
/// entry carries no capability information — the future ancestor's grant
/// already covers it — so it can only be a *target* the command owns, and
/// creating it as a directory would silently change the command's semantics
/// (a declared `.../result.txt` would become an unwritable directory). A
/// future root under an *existing* directory (a project's `dist`) is still
/// prepared: that entry is the only way to request the output directory.
/// `on_disk` reports `None` for a missing path, `Some(true)` for a directory
/// and `Some(false)` for any other existing entry.
pub(super) fn select_materialized_write_roots(
    write_roots: &[String],
    write_paths: &[String],
    on_disk: impl Fn(&str) -> Option<bool>,
) -> Vec<String> {
    let exists = |path: &str| on_disk(path).is_some();
    fn normalized(path: &str) -> std::path::PathBuf {
        let trimmed = path.trim();
        let trimmed = trimmed
            .strip_suffix('/')
            .filter(|p| !p.is_empty())
            .unwrap_or(trimmed);
        std::path::PathBuf::from(trimmed)
    }
    let future_grants: Vec<std::path::PathBuf> = write_roots
        .iter()
        .chain(write_paths.iter())
        .filter(|path| !exists(path))
        .map(|path| normalized(path))
        .collect();
    let future_exact_paths: Vec<std::path::PathBuf> = write_paths
        .iter()
        .filter(|path| !exists(path))
        .map(|path| normalized(path))
        .collect();
    let mut kept = Vec::new();
    for root in write_roots {
        let candidate = normalized(root);
        // Nested under a future grant: the ancestor covers it.
        let nested_under_future_grant = future_grants
            .iter()
            .any(|grant| candidate != *grant && candidate.starts_with(grant));
        // Also declared as a future *exact* path: the two declarations
        // contradict each other and the exact form is the more specific
        // one, so the entry is an exact file/dir target the command creates.
        let declared_as_exact_target = future_exact_paths.contains(&candidate);
        // Already on disk as a file (or other non-directory): it is an exact
        // target whatever role it was declared in; there is nothing to
        // prepare and `create_dir_all` would only fail with EEXIST.
        let existing_non_directory = on_disk(root) == Some(false);
        if !nested_under_future_grant
            && !declared_as_exact_target
            && !existing_non_directory
            && !kept.contains(root)
        {
            kept.push(root.clone());
        }
    }
    kept
}

/// Backend-aware form of [`select_materialized_write_roots`]: resolves each
/// declared grant against `cwd` to test existence.
pub(super) async fn materialized_write_roots(
    backend: &SharedExecutionBackend,
    write_roots: &[String],
    write_paths: &[String],
    cwd: Option<&aidaemon_tools::BackendPath>,
) -> Vec<String> {
    let mut on_disk = std::collections::HashMap::new();
    for path in write_roots.iter().chain(write_paths.iter()) {
        let Ok(resolved) = resolve_access_path(backend, path, cwd).await else {
            continue;
        };
        if let Ok(metadata) = backend.metadata(&resolved).await {
            on_disk.insert(path.clone(), metadata.is_dir());
        }
    }
    select_materialized_write_roots(write_roots, write_paths, |path| on_disk.get(path).copied())
}

pub(super) async fn resolve_access_path(
    backend: &SharedExecutionBackend,
    path: &str,
    cwd: Option<&aidaemon_tools::BackendPath>,
) -> anyhow::Result<aidaemon_tools::BackendPath> {
    let path = path.trim();
    anyhow::ensure!(!path.is_empty(), "execution access path cannot be empty");
    let prepared = if aidaemon_tools::BackendPath::is_absolute(path)
        || path == "~"
        || path.starts_with("~/")
    {
        path.to_string()
    } else if let Some(cwd) = cwd {
        cwd.join(path).to_string()
    } else {
        path.to_string()
    };
    backend.resolve_path(&prepared).await
}
