//! Workspace-bound execution backends: where agent-visible files and
//! processes live. The local backend is [`LocalBackend`]; this module adds
//! the SSH backend and, behind the `docker` feature, a persistent Docker
//! container, plus the `[execution]` settings that pick one.
//!
//! A host's control plane (configuration, state, channels, keychain, UI
//! automation) stays on the host. Agent-visible files and processes go
//! through the backend so a tool can never read one filesystem and execute
//! in another.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

use async_trait::async_trait;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::backend::{SharedExecutionBackend, normalize_posix_absolute};
use crate::local_backend::{LocalBackend, expand_local_path};
use crate::process_control::{
    configure_command_for_process_group, send_sigkill, send_sigterm, terminate_process_tree,
};
use crate::{
    BackendDirEntry, BackendFileType, BackendKind, BackendMetadata, BackendPath, ExecutionBackend,
    ExecutionOutput, ExecutionRequest, ProcessHandle, SpawnedProcess, WriteMode, shell_quote,
};

/// Agent-visible execution workspace configuration.
///
/// Daemon control-plane state (configuration, SQLite, channel inbox transport,
/// keychain access, and browser/computer-use sessions) always remains local.
/// Only tools that inspect or mutate the agent's working environment use this
/// backend.
#[derive(Debug, Deserialize, Clone)]
pub struct ExecutionConfig {
    /// Selected backend: `local`, `docker`, or `ssh`.
    #[serde(default = "default_execution_backend")]
    pub backend: String,
    /// Default working directory for relative agent-visible paths. For local
    /// execution this defaults to the daemon working directory.
    #[serde(default)]
    pub workspace_root: Option<String>,
    /// Allow path-based tools and explicit command working directories outside
    /// `workspace_root`. This defaults to true only for the local backend to
    /// preserve historical behavior. Shell commands retain the authority of
    /// the selected OS account/container and are governed by terminal approval.
    #[serde(default)]
    pub allow_outside_workspace: Option<bool>,
    /// Extra host variable names local commands inherit, on top of
    /// [`crate::local_backend::LOCAL_ENV_ALLOWLIST`] (a trailing `*` is a
    /// prefix). The daemon's other variables never reach a command.
    #[serde(default)]
    pub env_allowlist: Vec<String>,
    #[cfg_attr(not(feature = "docker"), allow(dead_code))]
    #[serde(default)]
    pub docker: DockerExecutionConfig,
    #[serde(default)]
    pub ssh: SshExecutionConfig,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            backend: default_execution_backend(),
            workspace_root: None,
            allow_outside_workspace: None,
            env_allowlist: Vec::new(),
            docker: DockerExecutionConfig::default(),
            ssh: SshExecutionConfig::default(),
        }
    }
}

impl ExecutionConfig {
    pub fn normalized_backend(&self) -> anyhow::Result<String> {
        let backend = self.backend.trim().to_ascii_lowercase();
        match backend.as_str() {
            "local" | "docker" | "ssh" => Ok(backend),
            _ => anyhow::bail!(
                "execution.backend must be one of: local, docker, ssh (got {:?})",
                self.backend
            ),
        }
    }

    pub fn effective_allow_outside_workspace(&self) -> bool {
        self.allow_outside_workspace
            .unwrap_or_else(|| self.backend.trim().eq_ignore_ascii_case("local"))
    }
}

fn default_execution_backend() -> String {
    "local".to_string()
}

#[cfg_attr(not(feature = "docker"), allow(dead_code))]
#[derive(Debug, Deserialize, Clone)]
pub struct DockerExecutionConfig {
    /// Persistent container name used for every tool call.
    #[serde(default = "default_docker_container")]
    pub container: String,
    /// Image used when `create_if_missing` creates the persistent container.
    #[serde(default = "default_docker_image")]
    pub image: String,
    /// Agent-visible workspace path inside the container.
    #[serde(default = "default_remote_workspace_root")]
    pub workspace_root: String,
    /// Host directory bind-mounted at `workspace_root`. Relative paths resolve
    /// next to config.toml. If omitted, the daemon working directory is used.
    #[serde(default)]
    pub host_workspace: Option<String>,
    /// Create and start the persistent container when it does not exist.
    #[serde(default = "default_true")]
    pub create_if_missing: bool,
    /// Docker network mode. Defaults to `none`; CLI agents that call cloud APIs
    /// need an explicitly configured network such as `bridge`.
    #[serde(default = "default_docker_network")]
    pub network: String,
    /// Optional Docker user (`uid:gid`, name, etc.) for `docker exec`.
    #[serde(default)]
    pub user: Option<String>,
    /// Additional environment variable names copied from the daemon into
    /// container executions. Values are never exposed in tool arguments.
    #[serde(default)]
    pub env_allowlist: Vec<String>,
}

impl Default for DockerExecutionConfig {
    fn default() -> Self {
        Self {
            container: default_docker_container(),
            image: default_docker_image(),
            workspace_root: default_remote_workspace_root(),
            host_workspace: None,
            create_if_missing: true,
            network: default_docker_network(),
            user: None,
            env_allowlist: Vec::new(),
        }
    }
}

fn default_docker_container() -> String {
    "aidaemon-execution".to_string()
}

fn default_docker_image() -> String {
    "ubuntu:24.04".to_string()
}

fn default_remote_workspace_root() -> String {
    "/workspace".to_string()
}

fn default_docker_network() -> String {
    "none".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct SshExecutionConfig {
    /// OpenSSH host or alias. Existing ~/.ssh/config, ssh-agent, ProxyJump,
    /// known_hosts, and hardware-token settings are honored.
    #[serde(default)]
    pub host: String,
    /// Persistent agent-visible directory on the SSH host.
    #[serde(default = "default_ssh_workspace_root")]
    pub workspace_root: String,
    #[serde(default)]
    pub port: Option<u16>,
    /// Optional local identity file path. Prefer an OpenSSH alias/ssh-agent.
    #[serde(default)]
    pub identity_file: Option<String>,
    /// Additional OpenSSH arguments, for example `["-J", "bastion"]`.
    #[serde(default)]
    pub extra_args: Vec<String>,
    #[serde(default = "default_ssh_connect_timeout")]
    pub connect_timeout_secs: u64,
    /// Create workspace_root on startup when it is missing.
    #[serde(default = "default_true")]
    pub create_workspace: bool,
    /// Environment variable names explicitly forwarded to remote executions.
    #[serde(default)]
    pub env_allowlist: Vec<String>,
}

impl Default for SshExecutionConfig {
    fn default() -> Self {
        Self {
            host: String::new(),
            workspace_root: default_ssh_workspace_root(),
            port: None,
            identity_file: None,
            extra_args: Vec::new(),
            connect_timeout_secs: default_ssh_connect_timeout(),
            create_workspace: true,
            env_allowlist: Vec::new(),
        }
    }
}

fn default_ssh_workspace_root() -> String {
    "~/aidaemon-workspace".to_string()
}

fn default_ssh_connect_timeout() -> u64 {
    10
}

fn default_true() -> bool {
    true
}

/// The local backend `[execution]` describes.
pub async fn local_backend(config: &ExecutionConfig) -> anyhow::Result<LocalBackend> {
    Ok(LocalBackend::new(
        config.workspace_root.as_deref(),
        config.effective_allow_outside_workspace(),
    )
    .await?
    .with_env_allowlist(config.env_allowlist.iter().cloned()))
}

pub async fn build_execution_backend(
    config: &ExecutionConfig,
    config_path: &Path,
) -> anyhow::Result<SharedExecutionBackend> {
    match config.normalized_backend()?.as_str() {
        "local" => Ok(Arc::new(local_backend(config).await?)),
        "ssh" => Ok(Arc::new(
            SshBackend::new(
                &config.ssh,
                config.workspace_root.as_deref(),
                config.effective_allow_outside_workspace(),
            )
            .await?,
        )),
        "docker" => {
            #[cfg(feature = "docker")]
            {
                Ok(Arc::new(
                    DockerBackend::new(
                        &config.docker,
                        config.workspace_root.as_deref(),
                        config.effective_allow_outside_workspace(),
                        config_path,
                    )
                    .await?,
                ))
            }
            #[cfg(not(feature = "docker"))]
            {
                let _ = config_path;
                anyhow::bail!(
                    "execution.backend=\"docker\" requires aidaemon built with \
                     --features execution-docker; the default binary remains Docker-free"
                )
            }
        }
        _ => unreachable!("validated execution backend"),
    }
}

#[derive(Clone)]
enum RemoteTransport {
    Ssh {
        host: String,
        args: Vec<String>,
    },
    #[cfg(feature = "docker")]
    Docker {
        container: String,
        user: Option<String>,
    },
}

#[derive(Clone)]
struct RemoteBackendCore {
    kind: BackendKind,
    id: String,
    approval_scope: String,
    workspace_root: BackendPath,
    canonical_workspace_root: String,
    home: BackendPath,
    allow_outside_workspace: bool,
    env_allowlist: Vec<String>,
    transport: RemoteTransport,
}

impl RemoteBackendCore {
    async fn run_transport(
        &self,
        script: &str,
        stdin: Option<Vec<u8>>,
        timeout: Duration,
    ) -> anyhow::Result<ExecutionOutput> {
        let mut child = self.spawn_transport(script, stdin).await?;
        let handle = child.handle().clone();
        let stdout = child
            .take_stdout()
            .ok_or_else(|| anyhow::anyhow!("remote transport stdout unavailable"))?;
        let stderr = child
            .take_stderr()
            .ok_or_else(|| anyhow::anyhow!("remote transport stderr unavailable"))?;
        let mut process = child.into_child();
        let started = Instant::now();
        let stdout_task = tokio::spawn(async move {
            let mut reader = stdout;
            let mut bytes = Vec::new();
            tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut bytes).await?;
            Ok::<_, std::io::Error>(bytes)
        });
        let stderr_task = tokio::spawn(async move {
            let mut reader = stderr;
            let mut bytes = Vec::new();
            tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut bytes).await?;
            Ok::<_, std::io::Error>(bytes)
        });
        let (status, timed_out) = match tokio::time::timeout(timeout, process.wait()).await {
            Ok(status) => (Some(status?), false),
            Err(_) => {
                if let Some(pid) = handle.local_pid() {
                    terminate_process_tree(pid, &mut process, Duration::from_secs(1)).await;
                }
                (None, true)
            }
        };
        Ok(ExecutionOutput {
            exit_code: status.and_then(|status| status.code()).unwrap_or(-1),
            stdout: stdout_task
                .await
                .ok()
                .and_then(Result::ok)
                .unwrap_or_default(),
            stderr: stderr_task
                .await
                .ok()
                .and_then(Result::ok)
                .unwrap_or_default(),
            duration_ms: started.elapsed().as_millis() as u64,
            timed_out,
        })
    }

    async fn spawn_transport(
        &self,
        script: &str,
        stdin: Option<Vec<u8>>,
    ) -> anyhow::Result<SpawnedProcess> {
        let mut command = match &self.transport {
            RemoteTransport::Ssh { host, args } => {
                let mut command = Command::new("ssh");
                // OpenSSH sends one remote command string rather than an argv
                // vector, so quote the script as one literal `sh -c` argument.
                command
                    .args(args)
                    .arg(host)
                    .arg(format!("sh -c {}", shell_quote(script)));
                command
            }
            #[cfg(feature = "docker")]
            RemoteTransport::Docker { container, user } => {
                let mut command = Command::new("docker");
                command.arg("exec").arg("-i");
                if let Some(user) = user {
                    command.arg("--user").arg(user);
                }
                command.arg(container).arg("sh").arg("-c").arg(script);
                command
            }
        };
        command
            .stdin(if stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        configure_command_for_process_group(&mut command);
        let mut child = command.spawn()?;
        if let (Some(stdin), Some(mut pipe)) = (stdin, child.stdin.take()) {
            tokio::spawn(async move {
                let _ = pipe.write_all(&stdin).await;
            });
        }
        let pid = child.id().unwrap_or(0);
        Ok(SpawnedProcess::new(
            ProcessHandle::new(pid, Some(pid), None),
            child,
        ))
    }

    async fn checked_path(&self, path: &BackendPath) -> anyhow::Result<()> {
        if self.allow_outside_workspace {
            return Ok(());
        }
        let path = normalize_posix_absolute(path.as_str())?;
        ensure_posix_descendant(&path, self.workspace_root.as_str())?;
        let script = format!(
            "p={}; root={}; \
             if [ -L \"$p\" ]; then printf 'symlink target refused: %s' \"$p\" >&2; exit 126; fi; \
             probe=\"$p\"; \
             while [ ! -e \"$probe\" ] && [ ! -L \"$probe\" ]; do \
               next=$(dirname -- \"$probe\"); [ \"$next\" = \"$probe\" ] && break; probe=\"$next\"; \
             done; \
             if [ -d \"$probe\" ]; then c=$(cd -P -- \"$probe\" && pwd); \
             else d=$(dirname -- \"$probe\"); c=$(cd -P -- \"$d\" && printf '%s/%s' \"$PWD\" \"$(basename -- \"$probe\")\"); fi; \
             case \"$c\" in \"$root\"|\"$root\"/*) exit 0;; *) printf '%s' \"$c\" >&2; exit 126;; esac",
            shell_quote(&path),
            shell_quote(&self.canonical_workspace_root),
        );
        let output = self
            .run_transport(&script, None, Duration::from_secs(10))
            .await?;
        anyhow::ensure!(
            output.exit_code == 0,
            "Path escapes execution workspace {}: {} ({})",
            self.workspace_root,
            path,
            output.stderr_lossy().trim()
        );
        Ok(())
    }

    async fn resolve_path(&self, path: &str) -> anyhow::Result<BackendPath> {
        let expanded = if path == "~" {
            self.home.as_str().to_string()
        } else if let Some(rest) = path.strip_prefix("~/") {
            format!("{}/{}", self.home.as_str().trim_end_matches('/'), rest)
        } else if path.starts_with('/') {
            path.to_string()
        } else {
            format!(
                "{}/{}",
                self.workspace_root.as_str().trim_end_matches('/'),
                path
            )
        };
        let normalized = normalize_posix_absolute(&expanded)?;
        let result = BackendPath::new(normalized);
        self.checked_path(&result).await?;
        Ok(result)
    }

    async fn canonicalize(&self, path: &BackendPath) -> anyhow::Result<BackendPath> {
        self.checked_path(path).await?;
        let script = format!(
            "p={}; if [ -d \"$p\" ]; then cd -P -- \"$p\" && pwd; \
             else d=$(dirname -- \"$p\"); b=$(basename -- \"$p\"); cd -P -- \"$d\" && printf '%s/%s\\n' \"$PWD\" \"$b\"; fi",
            shell_quote(path.as_str())
        );
        let output = self
            .run_transport(&script, None, Duration::from_secs(10))
            .await?;
        remote_success("canonicalize", output).map(|text| BackendPath::new(text.trim()))
    }

    async fn metadata(&self, path: &BackendPath) -> anyhow::Result<BackendMetadata> {
        self.checked_path(path).await?;
        let script = format!(
            "p={}; \
             if [ -L \"$p\" ]; then t=l; elif [ -f \"$p\" ]; then t=f; \
             elif [ -d \"$p\" ]; then t=d; elif [ -e \"$p\" ]; then t=o; else exit 44; fi; \
             n=$(wc -c < \"$p\" 2>/dev/null || printf 0); \
             m=$(stat -c %Y \"$p\" 2>/dev/null || stat -f %m \"$p\" 2>/dev/null || printf 0); \
             printf '%s\\t%s\\t%s\\n' \"$t\" \"$n\" \"$m\"",
            shell_quote(path.as_str())
        );
        let output = self
            .run_transport(&script, None, Duration::from_secs(10))
            .await?;
        let text = remote_success("metadata", output)?;
        parse_remote_metadata(text.trim())
    }

    async fn read(&self, path: &BackendPath) -> anyhow::Result<Vec<u8>> {
        self.checked_path(path).await?;
        let output = self
            .run_transport(
                &format!("cat -- {}", shell_quote(path.as_str())),
                None,
                Duration::from_secs(300),
            )
            .await?;
        if output.exit_code != 0 {
            anyhow::bail!("remote read failed: {}", output.stderr_lossy().trim());
        }
        Ok(output.stdout)
    }

    async fn write(
        &self,
        path: &BackendPath,
        content: &[u8],
        mode: WriteMode,
        create_parents: bool,
    ) -> anyhow::Result<()> {
        self.checked_path(path).await?;
        let parent = path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Cannot write path without parent: {path}"))?;
        self.checked_path(&parent).await?;
        let mkdir = if create_parents {
            format!("mkdir -p -- {} && ", shell_quote(parent.as_str()))
        } else {
            String::new()
        };
        let script = match mode {
            WriteMode::Append => {
                format!("{mkdir}cat >> {}", shell_quote(path.as_str()))
            }
            WriteMode::Overwrite => {
                let temp = format!("{}.aidaemon-write-{}", path.as_str(), uuid::Uuid::new_v4());
                format!(
                    "{mkdir}cat > {} && mv -f -- {} {}",
                    shell_quote(&temp),
                    shell_quote(&temp),
                    shell_quote(path.as_str())
                )
            }
        };
        let output = self
            .run_transport(&script, Some(content.to_vec()), Duration::from_secs(300))
            .await?;
        remote_success("write", output)?;
        Ok(())
    }

    async fn create_dir_all(&self, path: &BackendPath) -> anyhow::Result<()> {
        self.checked_path(path).await?;
        let output = self
            .run_transport(
                &format!("mkdir -p -- {}", shell_quote(path.as_str())),
                None,
                Duration::from_secs(30),
            )
            .await?;
        remote_success("create directory", output)?;
        Ok(())
    }

    async fn copy(&self, source: &BackendPath, destination: &BackendPath) -> anyhow::Result<()> {
        self.checked_path(source).await?;
        self.checked_path(destination).await?;
        let output = self
            .run_transport(
                &format!(
                    "cp -p -- {} {}",
                    shell_quote(source.as_str()),
                    shell_quote(destination.as_str())
                ),
                None,
                Duration::from_secs(60),
            )
            .await?;
        remote_success("copy", output)?;
        Ok(())
    }

    async fn rename(&self, source: &BackendPath, destination: &BackendPath) -> anyhow::Result<()> {
        self.checked_path(source).await?;
        self.checked_path(destination).await?;
        let output = self
            .run_transport(
                &format!(
                    "mv -f -- {} {}",
                    shell_quote(source.as_str()),
                    shell_quote(destination.as_str())
                ),
                None,
                Duration::from_secs(60),
            )
            .await?;
        remote_success("rename", output)?;
        Ok(())
    }

    async fn remove_file(&self, path: &BackendPath) -> anyhow::Result<()> {
        self.checked_path(path).await?;
        let output = self
            .run_transport(
                &format!("rm -f -- {}", shell_quote(path.as_str())),
                None,
                Duration::from_secs(30),
            )
            .await?;
        remote_success("remove file", output)?;
        Ok(())
    }

    async fn remove_empty_dir(&self, path: &BackendPath) -> anyhow::Result<()> {
        self.checked_path(path).await?;
        let output = self
            .run_transport(
                &format!("rmdir -- {}", shell_quote(path.as_str())),
                None,
                Duration::from_secs(30),
            )
            .await?;
        remote_success("remove empty directory", output)?;
        Ok(())
    }

    async fn read_dir(&self, path: &BackendPath) -> anyhow::Result<Vec<BackendDirEntry>> {
        self.checked_path(path).await?;
        let output = self
            .run_transport(
                &format!(
                    "find {} -mindepth 1 -maxdepth 1 -print0",
                    shell_quote(path.as_str())
                ),
                None,
                Duration::from_secs(30),
            )
            .await?;
        if output.exit_code != 0 {
            anyhow::bail!(
                "remote directory listing failed: {}",
                output.stderr_lossy().trim()
            );
        }
        let mut entries = Vec::new();
        for raw in output.stdout.split(|byte| *byte == 0) {
            if raw.is_empty() {
                continue;
            }
            let path = BackendPath::new(String::from_utf8_lossy(raw).into_owned());
            entries.push(BackendDirEntry {
                metadata: self.metadata(&path).await?,
                path,
            });
        }
        Ok(entries)
    }

    async fn spawn(&self, request: ExecutionRequest) -> anyhow::Result<SpawnedProcess> {
        let cwd = request.cwd.as_ref().unwrap_or(&self.workspace_root).clone();
        self.checked_path(&cwd).await?;
        let token = uuid::Uuid::new_v4().simple().to_string();
        let pid_file = format!("/tmp/aidaemon-exec-{token}.pid");
        let mut request_env = request.env;
        let removed_env = request
            .env_remove
            .into_iter()
            .filter(|name| valid_env_name(name))
            .collect::<BTreeSet<_>>();
        for name in &removed_env {
            request_env.remove(name);
        }
        let env = self.forwarded_environment(&request_env, &removed_env);
        let unset = removed_env
            .iter()
            .map(|name| format!("unset {name}; "))
            .collect::<String>();
        let command = request.command.as_posix_shell();
        let script = format!(
            "cd -- {} || exit $?; {} \
             if command -v setsid >/dev/null 2>&1; then \
               {}setsid sh -c {} & child=$!; target=-$child; \
             else {}sh -c {} & child=$!; target=$child; fi; \
             printf '%s' \"$target\" > {}; \
             wait \"$child\"; status=$?; rm -f -- {}; exit \"$status\"",
            shell_quote(cwd.as_str()),
            unset,
            env,
            shell_quote(&command),
            env,
            shell_quote(&command),
            shell_quote(&pid_file),
            shell_quote(&pid_file),
        );
        let mut spawned = self.spawn_transport(&script, request.stdin).await?;
        spawned.set_remote_token(token);
        Ok(spawned)
    }

    fn forwarded_environment(
        &self,
        request_env: &BTreeMap<String, String>,
        removed_env: &BTreeSet<String>,
    ) -> String {
        let mut values = BTreeMap::new();
        for name in &self.env_allowlist {
            if !removed_env.contains(name)
                && let Ok(value) = std::env::var(name)
            {
                values.insert(name.clone(), value);
            }
        }
        values.extend(request_env.clone());
        values
            .into_iter()
            .filter(|(name, _)| valid_env_name(name))
            .map(|(name, value)| format!("{name}={} ", shell_quote(&value)))
            .collect()
    }

    async fn terminate(&self, handle: &ProcessHandle, grace: Duration) -> anyhow::Result<()> {
        if let Some(token) = handle.remote_token() {
            let pid_file = format!("/tmp/aidaemon-exec-{token}.pid");
            let term = format!(
                "p={}; if [ -f \"$p\" ]; then target=$(cat \"$p\"); \
                 kill -TERM -- \"$target\" 2>/dev/null || \
                 kill -TERM \"$target\" 2>/dev/null || true; fi",
                shell_quote(&pid_file)
            );
            let _ = self
                .run_transport(&term, None, Duration::from_secs(5))
                .await;
            tokio::time::sleep(grace).await;
            let kill = format!(
                "p={}; if [ -f \"$p\" ]; then target=$(cat \"$p\"); \
                 kill -KILL -- \"$target\" 2>/dev/null || \
                 kill -KILL \"$target\" 2>/dev/null || true; rm -f \"$p\"; fi",
                shell_quote(&pid_file)
            );
            let _ = self
                .run_transport(&kill, None, Duration::from_secs(5))
                .await;
        }
        if let Some(pid) = handle.local_pid() {
            send_sigterm(pid);
            send_sigkill(pid);
        }
        Ok(())
    }
}

pub struct SshBackend {
    core: RemoteBackendCore,
}

impl SshBackend {
    async fn new(
        config: &SshExecutionConfig,
        workspace_override: Option<&str>,
        allow_outside_workspace: bool,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !config.host.trim().is_empty(),
            "execution.ssh.host is required when execution.backend=\"ssh\""
        );
        ensure_local_executable("ssh").await?;
        let mut args = vec![
            "-o".to_string(),
            "BatchMode=yes".to_string(),
            "-o".to_string(),
            format!("ConnectTimeout={}", config.connect_timeout_secs.max(1)),
        ];
        if let Some(port) = config.port {
            args.push("-p".to_string());
            args.push(port.to_string());
        }
        if let Some(identity_file) = &config.identity_file {
            args.push("-i".to_string());
            args.push(
                expand_local_path(identity_file)?
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        args.extend(config.extra_args.clone());

        let provisional = RemoteBackendCore {
            kind: BackendKind::Ssh,
            id: format!("ssh:{}", config.host),
            approval_scope: format!("ssh:{}", config.host),
            workspace_root: BackendPath::new(workspace_override.unwrap_or(&config.workspace_root)),
            canonical_workspace_root: String::new(),
            home: BackendPath::new("/"),
            allow_outside_workspace: true,
            env_allowlist: config.env_allowlist.clone(),
            transport: RemoteTransport::Ssh {
                host: config.host.clone(),
                args,
            },
        };
        let home_output = provisional
            .run_transport(
                "printf '%s\\n' \"$HOME\"; \
                 fingerprint=$(cat /etc/machine-id 2>/dev/null || \
                   hostname 2>/dev/null || printf unknown); \
                 set -- ${SSH_CONNECTION:-}; \
                 printf '%s|%s:%s\\n' \"$fingerprint\" \"${3:-unknown}\" \"${4:-unknown}\"",
                None,
                Duration::from_secs(config.connect_timeout_secs.max(1) + 5),
            )
            .await?;
        let identity_text = remote_success("SSH connectivity check", home_output)?;
        let mut identity_lines = identity_text.lines();
        let home = identity_lines.next().unwrap_or_default().trim().to_string();
        let remote_fingerprint = identity_lines.next().unwrap_or("unknown").trim();
        anyhow::ensure!(!home.is_empty(), "SSH host returned an empty HOME");
        let fingerprint_hash = hex::encode(Sha256::digest(remote_fingerprint.as_bytes()));

        let workspace = workspace_override.unwrap_or(&config.workspace_root);
        let workspace = if workspace == "~" {
            home.clone()
        } else if let Some(rest) = workspace.strip_prefix("~/") {
            format!("{}/{}", home.trim_end_matches('/'), rest)
        } else {
            workspace.to_string()
        };
        let workspace = normalize_posix_absolute(&workspace)?;
        if config.create_workspace {
            let output = provisional
                .run_transport(
                    &format!("mkdir -p -- {}", shell_quote(&workspace)),
                    None,
                    Duration::from_secs(30),
                )
                .await?;
            remote_success("create SSH workspace", output)?;
        }
        let canonical = provisional
            .run_transport(
                &format!("cd -P -- {} && pwd", shell_quote(&workspace)),
                None,
                Duration::from_secs(30),
            )
            .await?;
        let canonical = remote_success("resolve SSH workspace", canonical)?
            .trim()
            .to_string();
        let target_identity = format!(
            "ssh:{}:{}:{}:{}",
            config.host,
            config.port.unwrap_or(22),
            &fingerprint_hash[..16],
            canonical
        );

        Ok(Self {
            core: RemoteBackendCore {
                id: target_identity.clone(),
                approval_scope: target_identity,
                workspace_root: BackendPath::new(canonical.clone()),
                canonical_workspace_root: canonical,
                home: BackendPath::new(home),
                allow_outside_workspace,
                ..provisional
            },
        })
    }
}

macro_rules! delegate_remote_backend {
    ($backend:ty) => {
        #[async_trait]
        impl ExecutionBackend for $backend {
            fn kind(&self) -> BackendKind {
                self.core.kind
            }
            fn id(&self) -> &str {
                &self.core.id
            }
            fn approval_scope(&self) -> &str {
                &self.core.approval_scope
            }
            fn workspace_root(&self) -> &BackendPath {
                &self.core.workspace_root
            }
            fn home_hint(&self) -> &BackendPath {
                &self.core.home
            }
            fn allows_outside_workspace(&self) -> bool {
                self.core.allow_outside_workspace
            }
            async fn home_dir(&self) -> anyhow::Result<BackendPath> {
                Ok(self.core.home.clone())
            }
            async fn resolve_path(&self, path: &str) -> anyhow::Result<BackendPath> {
                self.core.resolve_path(path).await
            }
            async fn canonicalize(&self, path: &BackendPath) -> anyhow::Result<BackendPath> {
                self.core.canonicalize(path).await
            }
            async fn metadata(&self, path: &BackendPath) -> anyhow::Result<BackendMetadata> {
                self.core.metadata(path).await
            }
            async fn read(&self, path: &BackendPath) -> anyhow::Result<Vec<u8>> {
                self.core.read(path).await
            }
            async fn write(
                &self,
                path: &BackendPath,
                content: &[u8],
                mode: WriteMode,
                create_parents: bool,
            ) -> anyhow::Result<()> {
                self.core.write(path, content, mode, create_parents).await
            }
            async fn create_dir_all(&self, path: &BackendPath) -> anyhow::Result<()> {
                self.core.create_dir_all(path).await
            }
            async fn copy(
                &self,
                source: &BackendPath,
                destination: &BackendPath,
            ) -> anyhow::Result<()> {
                self.core.copy(source, destination).await
            }
            async fn rename(
                &self,
                source: &BackendPath,
                destination: &BackendPath,
            ) -> anyhow::Result<()> {
                self.core.rename(source, destination).await
            }
            async fn remove_file(&self, path: &BackendPath) -> anyhow::Result<()> {
                self.core.remove_file(path).await
            }
            async fn remove_empty_dir(&self, path: &BackendPath) -> anyhow::Result<()> {
                self.core.remove_empty_dir(path).await
            }
            async fn read_dir(&self, path: &BackendPath) -> anyhow::Result<Vec<BackendDirEntry>> {
                self.core.read_dir(path).await
            }
            async fn spawn(&self, request: ExecutionRequest) -> anyhow::Result<SpawnedProcess> {
                self.core.spawn(request).await
            }
            async fn terminate(
                &self,
                handle: &ProcessHandle,
                grace: Duration,
            ) -> anyhow::Result<()> {
                self.core.terminate(handle, grace).await
            }
        }
    };
}

delegate_remote_backend!(SshBackend);

#[cfg(feature = "docker")]
pub struct DockerBackend {
    core: RemoteBackendCore,
}

#[cfg(feature = "docker")]
impl DockerBackend {
    async fn new(
        config: &DockerExecutionConfig,
        workspace_override: Option<&str>,
        allow_outside_workspace: bool,
        config_path: &Path,
    ) -> anyhow::Result<Self> {
        ensure_local_executable("docker").await?;
        anyhow::ensure!(
            !config.container.trim().is_empty(),
            "execution.docker.container cannot be empty"
        );
        let inspect = local_command_output(
            "docker",
            &["inspect", "--type", "container", &config.container],
            Duration::from_secs(15),
        )
        .await?;
        if inspect.exit_code != 0 {
            anyhow::ensure!(
                config.create_if_missing,
                "Docker execution container {:?} does not exist and create_if_missing=false",
                config.container
            );
            let host_workspace = match &config.host_workspace {
                Some(path) => {
                    let expanded = expand_local_path(path)?;
                    if expanded.is_absolute() {
                        expanded
                    } else {
                        config_path
                            .parent()
                            .unwrap_or_else(|| Path::new("."))
                            .join(expanded)
                    }
                }
                None => std::env::current_dir()?,
            };
            tokio::fs::create_dir_all(&host_workspace).await?;
            let host_workspace = tokio::fs::canonicalize(host_workspace).await?;
            let workspace = workspace_override.unwrap_or(&config.workspace_root);
            let mut args = vec![
                "create".to_string(),
                "--name".to_string(),
                config.container.clone(),
                "--init".to_string(),
                "--workdir".to_string(),
                workspace.to_string(),
                "--network".to_string(),
                config.network.clone(),
                "--cap-drop".to_string(),
                "ALL".to_string(),
                "--security-opt".to_string(),
                "no-new-privileges".to_string(),
                "--mount".to_string(),
                format!(
                    "type=bind,src={},dst={}",
                    host_workspace.display(),
                    workspace
                ),
            ];
            if let Some(user) = &config.user {
                args.push("--user".to_string());
                args.push(user.clone());
            }
            args.extend([
                config.image.clone(),
                "sh".to_string(),
                "-c".to_string(),
                "trap 'exit 0' TERM INT; while :; do sleep 3600; done".to_string(),
            ]);
            let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
            let created = local_command_output("docker", &refs, Duration::from_secs(120)).await?;
            anyhow::ensure!(
                created.exit_code == 0,
                "Failed to create Docker execution container: {}",
                created.stderr_lossy().trim()
            );
        }
        let start = local_command_output(
            "docker",
            &["start", &config.container],
            Duration::from_secs(60),
        )
        .await?;
        anyhow::ensure!(
            start.exit_code == 0,
            "Failed to start Docker execution container: {}",
            start.stderr_lossy().trim()
        );
        let identity = local_command_output(
            "docker",
            &["inspect", "--format", "{{.Id}}", &config.container],
            Duration::from_secs(15),
        )
        .await?;
        anyhow::ensure!(
            identity.exit_code == 0,
            "Failed to identify Docker execution container: {}",
            identity.stderr_lossy().trim()
        );
        let container_id = identity.stdout_lossy().trim().to_string();
        anyhow::ensure!(
            !container_id.is_empty(),
            "Docker returned an empty container identity"
        );

        let workspace =
            normalize_posix_absolute(workspace_override.unwrap_or(&config.workspace_root))?;
        let provisional = RemoteBackendCore {
            kind: BackendKind::Docker,
            id: format!("docker:{}", config.container),
            approval_scope: format!("docker:{}", config.container),
            workspace_root: BackendPath::new(workspace.clone()),
            canonical_workspace_root: workspace.clone(),
            home: BackendPath::new("/root"),
            allow_outside_workspace: true,
            env_allowlist: config.env_allowlist.clone(),
            transport: RemoteTransport::Docker {
                container: config.container.clone(),
                user: config.user.clone(),
            },
        };
        let ready = provisional
            .run_transport(
                &format!(
                    "mkdir -p -- {} && cd -P -- {} && pwd && printf '%s\\n' \"$HOME\"",
                    shell_quote(&workspace),
                    shell_quote(&workspace)
                ),
                None,
                Duration::from_secs(30),
            )
            .await?;
        let text = remote_success("initialize Docker execution workspace", ready)?;
        let mut lines = text.lines();
        let canonical = lines.next().unwrap_or(&workspace).trim().to_string();
        let home = lines
            .next()
            .map(str::trim)
            .filter(|home| !home.is_empty())
            .unwrap_or(&canonical)
            .to_string();
        let target_identity = format!("docker:{container_id}:{canonical}");
        Ok(Self {
            core: RemoteBackendCore {
                id: target_identity.clone(),
                approval_scope: target_identity,
                workspace_root: BackendPath::new(canonical.clone()),
                canonical_workspace_root: canonical,
                home: BackendPath::new(home),
                allow_outside_workspace,
                ..provisional
            },
        })
    }
}

#[cfg(feature = "docker")]
delegate_remote_backend!(DockerBackend);

fn parse_remote_metadata(value: &str) -> anyhow::Result<BackendMetadata> {
    let mut parts = value.split('\t');
    let kind = match parts.next() {
        Some("f") => BackendFileType::File,
        Some("d") => BackendFileType::Directory,
        Some("l") => BackendFileType::Symlink,
        Some("o") => BackendFileType::Other,
        _ => anyhow::bail!("remote backend returned invalid metadata: {value:?}"),
    };
    let len = parts
        .next()
        .unwrap_or("0")
        .trim()
        .parse::<u64>()
        .unwrap_or(0);
    let seconds = parts
        .next()
        .unwrap_or("0")
        .trim()
        .parse::<u64>()
        .unwrap_or(0);
    Ok(BackendMetadata {
        file_type: kind,
        len,
        modified: (seconds > 0).then(|| UNIX_EPOCH + Duration::from_secs(seconds)),
    })
}

fn remote_success(operation: &str, output: ExecutionOutput) -> anyhow::Result<String> {
    anyhow::ensure!(
        !output.timed_out,
        "Remote {operation} timed out after {}ms",
        output.duration_ms
    );
    anyhow::ensure!(
        output.exit_code == 0,
        "Remote {operation} failed (exit {}): {}",
        output.exit_code,
        output.stderr_lossy().trim()
    );
    Ok(output.stdout_lossy())
}

fn valid_env_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|first| first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn ensure_posix_descendant(path: &str, root: &str) -> anyhow::Result<()> {
    let root = root.trim_end_matches('/');
    anyhow::ensure!(
        path == root || path.starts_with(&format!("{root}/")),
        "Path is outside execution workspace {root}: {path}"
    );
    Ok(())
}

async fn ensure_local_executable(executable: &str) -> anyhow::Result<()> {
    let output = local_command_output(
        "sh",
        &["-c", &format!("command -v {executable}")],
        Duration::from_secs(5),
    )
    .await?;
    anyhow::ensure!(
        output.exit_code == 0,
        "Required executable {:?} was not found on the daemon host",
        executable
    );
    Ok(())
}

async fn local_command_output(
    program: &str,
    args: &[&str],
    timeout: Duration,
) -> anyhow::Result<ExecutionOutput> {
    let started = Instant::now();
    let mut command = Command::new(program);
    command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    configure_command_for_process_group(&mut command);
    let mut child = command.spawn()?;
    let pid = child.id().unwrap_or(0);
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("stdout unavailable"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("stderr unavailable"))?;
    let stdout_task = tokio::spawn(async move {
        let mut reader = stdout;
        let mut bytes = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut bytes).await?;
        Ok::<_, std::io::Error>(bytes)
    });
    let stderr_task = tokio::spawn(async move {
        let mut reader = stderr;
        let mut bytes = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut bytes).await?;
        Ok::<_, std::io::Error>(bytes)
    });
    let (status, timed_out) = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(status) => (Some(status?), false),
        Err(_) => {
            terminate_process_tree(pid, &mut child, Duration::from_secs(1)).await;
            (None, true)
        }
    };
    Ok(ExecutionOutput {
        exit_code: status.and_then(|status| status.code()).unwrap_or(-1),
        stdout: stdout_task
            .await
            .ok()
            .and_then(Result::ok)
            .unwrap_or_default(),
        stderr: stderr_task
            .await
            .ok()
            .and_then(Result::ok)
            .unwrap_or_default(),
        duration_ms: started.elapsed().as_millis() as u64,
        timed_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LexicalPaths;
    use crate::backend::normalize_path_lexically;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn local_backend_is_shared_as_the_one_trait() {
        let local: Arc<dyn ExecutionBackend> = Arc::new(LocalBackend::current_unrestricted());
        let shared: SharedExecutionBackend = local.clone();
        assert_eq!(shared.kind(), BackendKind::Local);
        assert_eq!(local.id(), shared.id());
    }

    #[test]
    fn local_execution_path_includes_user_tool_roots() {
        let home = BackendPath::new("/synthetic/home");
        let path = crate::local_backend::local_execution_search_path(&home);
        let directories = std::env::split_paths(std::ffi::OsStr::new(&path)).collect::<Vec<_>>();
        assert!(directories.contains(&PathBuf::from("/synthetic/home/.cargo/bin")));
        assert!(directories.contains(&PathBuf::from("/synthetic/home/.local/bin")));
        assert!(directories.contains(&PathBuf::from("/usr/bin")));
    }

    async fn backend_contract(backend: &dyn ExecutionBackend) -> anyhow::Result<()> {
        let path = backend.resolve_path("nested/value.txt").await?;
        backend
            .write(&path, b"from file api\n", WriteMode::Overwrite, true)
            .await?;

        let output = backend
            .execute(
                ExecutionRequest::shell(
                    "cat nested/value.txt && printf 'from command\\n' >> nested/value.txt",
                ),
                Duration::from_secs(5),
            )
            .await?;
        anyhow::ensure!(output.exit_code == 0, "shell command failed");
        anyhow::ensure!(output.stdout_lossy() == "from file api\n");
        anyhow::ensure!(
            String::from_utf8(backend.read(&path).await?)? == "from file api\nfrom command\n"
        );

        let mut argv = ExecutionRequest::argv(
            "sh",
            vec![
                "-c".to_string(),
                "printf '%s' \"$1\"".to_string(),
                "sh".to_string(),
                "literal value; $(not-executed) 'quoted'".to_string(),
            ],
        );
        argv.env.insert(
            "AIDAEMON_CONTRACT_ENV".to_string(),
            "visible value".to_string(),
        );
        argv.env.insert(
            "AIDAEMON_CONTRACT_REMOVED".to_string(),
            "must-not-leak".to_string(),
        );
        argv.env_remove
            .push("AIDAEMON_CONTRACT_REMOVED".to_string());
        let output = backend.execute(argv, Duration::from_secs(5)).await?;
        anyhow::ensure!(output.exit_code == 0, "argv command failed");
        anyhow::ensure!(
            output.stdout_lossy() == "literal value; $(not-executed) 'quoted'",
            "argv quoting changed the literal argument"
        );

        let mut environment = ExecutionRequest::shell(
            "test \"$AIDAEMON_CONTRACT_ENV\" = 'visible value' && \
             test -z \"${AIDAEMON_CONTRACT_REMOVED+x}\"",
        );
        environment.env.insert(
            "AIDAEMON_CONTRACT_ENV".to_string(),
            "visible value".to_string(),
        );
        environment.env.insert(
            "AIDAEMON_CONTRACT_REMOVED".to_string(),
            "must-not-leak".to_string(),
        );
        environment
            .env_remove
            .push("AIDAEMON_CONTRACT_REMOVED".to_string());
        let output = backend.execute(environment, Duration::from_secs(5)).await?;
        anyhow::ensure!(output.exit_code == 0, "environment contract failed");

        let nested = backend.resolve_path("nested").await?;
        let entries = backend.read_dir(&nested).await?;
        anyhow::ensure!(
            entries
                .iter()
                .any(|entry| entry.path.file_name() == Some("value.txt"))
        );
        anyhow::ensure!(backend.metadata(&path).await?.is_file());

        let copy = backend.resolve_path("nested/copied.txt").await?;
        let renamed = backend.resolve_path("nested/renamed.txt").await?;
        backend.copy(&path, &copy).await?;
        backend.rename(&copy, &renamed).await?;
        anyhow::ensure!(backend.metadata(&renamed).await?.is_file());
        backend.remove_file(&renamed).await?;
        anyhow::ensure!(backend.metadata(&renamed).await.is_err());

        let host_temp = TempDir::new()?;
        let host_source = host_temp.path().join("attachment.txt");
        tokio::fs::write(&host_source, b"attachment bytes").await?;
        let imported_destination = backend.resolve_path("inbox/attachment.txt").await?;
        let imported = backend
            .import_local_file(&host_source, &imported_destination)
            .await?;
        anyhow::ensure!(backend.read(&imported).await? == b"attachment bytes");
        let host_export = host_temp.path().join("exported.txt");
        backend.export_local_file(&imported, &host_export).await?;
        anyhow::ensure!(tokio::fs::read(host_export).await? == b"attachment bytes");

        anyhow::ensure!(
            backend.resolve_path("../outside-workspace").await.is_err(),
            "confined backend accepted a workspace escape"
        );
        Ok(())
    }

    #[tokio::test]
    async fn local_backend_round_trip_and_command_share_workspace() {
        let temp = TempDir::new().unwrap();
        let config = ExecutionConfig {
            workspace_root: Some(temp.path().to_string_lossy().into_owned()),
            allow_outside_workspace: Some(false),
            ..ExecutionConfig::default()
        };
        let backend = local_backend(&config).await.unwrap();
        backend_contract(&backend).await.unwrap();
    }

    #[tokio::test]
    async fn local_backend_can_confine_workspace() {
        let temp = TempDir::new().unwrap();
        let config = ExecutionConfig {
            workspace_root: Some(temp.path().to_string_lossy().into_owned()),
            allow_outside_workspace: Some(false),
            ..ExecutionConfig::default()
        };
        let backend = local_backend(&config).await.unwrap();
        assert!(backend.resolve_path("ok.txt").await.is_ok());
        assert!(backend.resolve_path("/etc/passwd").await.is_err());
    }

    /// A daemon secret in the process environment never reaches a command
    /// on the local backend; an allowlisted variable does, and so does one
    /// the request sets itself.
    #[tokio::test]
    async fn local_commands_do_not_inherit_daemon_secrets() {
        // SAFETY: names unique to this test; nothing else reads them.
        unsafe {
            std::env::set_var("AIDAEMON_SYNTHETIC_SECRET_KEY", "synthetic-secret-value");
            std::env::set_var("AIDAEMON_SYNTHETIC_ALLOWED", "synthetic-allowed-value");
        }
        let backend =
            LocalBackend::current_unrestricted().with_env_allowlist(["AIDAEMON_SYNTHETIC_ALLOWED"]);
        let mut request = ExecutionRequest::shell("env");
        request.env.insert(
            "AIDAEMON_REQUEST_VAR".to_string(),
            "set-by-request".to_string(),
        );
        let output = backend
            .execute(request, std::time::Duration::from_secs(10))
            .await
            .unwrap();
        let env = output.stdout_lossy();
        assert!(!env.contains("synthetic-secret-value"), "{env}");
        assert!(env.contains("AIDAEMON_SYNTHETIC_ALLOWED=synthetic-allowed-value"));
        assert!(env.contains("AIDAEMON_REQUEST_VAR=set-by-request"));
        assert!(env.contains("PATH="), "{env}");
        assert!(env.contains("HOME="), "{env}");
    }

    #[test]
    fn lexical_paths_normalize_exactly_as_their_backend() {
        let backend = LocalBackend::current_unrestricted();
        let paths = LexicalPaths::of(&backend);
        for path in ["relative/file.txt", "~/notes.md", "~", "/tmp/a/../b"] {
            assert_eq!(
                paths.normalize(path).unwrap(),
                normalize_path_lexically(&backend, path).unwrap(),
                "{path}"
            );
        }
    }

    #[test]
    fn posix_normalization_rejects_root_escape() {
        assert_eq!(
            normalize_posix_absolute("/workspace/a/../b").unwrap(),
            "/workspace/b"
        );
        assert!(normalize_posix_absolute("/../etc/passwd").is_err());
    }

    #[test]
    fn shell_quoting_is_literal() {
        assert_eq!(shell_quote("a'b"), "'a'\"'\"'b'");
    }

    #[test]
    fn backend_path_keeps_posix_separators_and_extensions() {
        let path = BackendPath::new("/workspace/nested");
        let file = path.join("value.test.txt");
        assert_eq!(file.as_str(), "/workspace/nested/value.test.txt");
        assert_eq!(file.parent().unwrap().as_str(), "/workspace/nested");
        assert_eq!(file.file_name(), Some("value.test.txt"));
        assert_eq!(file.extension(), Some("txt"));
        assert_eq!(
            file.with_extension("bak").as_str(),
            "/workspace/nested/value.test.bak"
        );
    }

    #[test]
    fn ssh_default_workspace_is_user_writable() {
        assert_eq!(
            SshExecutionConfig::default().workspace_root,
            "~/aidaemon-workspace"
        );
    }

    #[cfg(feature = "docker")]
    #[tokio::test]
    #[ignore = "requires a running Docker daemon; opt in with AIDAEMON_RUN_DOCKER_TEST=1"]
    async fn docker_backend_matches_execution_contract() {
        if std::env::var("AIDAEMON_RUN_DOCKER_TEST").as_deref() != Ok("1") {
            return;
        }
        let temp = TempDir::new().unwrap();
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let container = format!("aidaemon-contract-{}", &suffix[..12]);
        let config = DockerExecutionConfig {
            container: container.clone(),
            image: std::env::var("AIDAEMON_TEST_DOCKER_IMAGE")
                .unwrap_or_else(|_| "alpine:3.20".to_string()),
            host_workspace: Some(temp.path().to_string_lossy().into_owned()),
            ..DockerExecutionConfig::default()
        };
        let config_path = temp.path().join("config.toml");

        let result = async {
            let backend = DockerBackend::new(&config, None, false, &config_path).await?;
            backend_contract(&backend).await
        }
        .await;
        let _ = local_command_output("docker", &["rm", "-f", &container], Duration::from_secs(30))
            .await;
        result.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires AIDAEMON_TEST_SSH_HOST pointing to an isolated test account"]
    async fn ssh_backend_matches_execution_contract() {
        let Ok(host) = std::env::var("AIDAEMON_TEST_SSH_HOST") else {
            return;
        };
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let workspace = format!("/tmp/aidaemon-contract-{}", &suffix[..12]);
        let config = SshExecutionConfig {
            host,
            workspace_root: workspace.clone(),
            port: std::env::var("AIDAEMON_TEST_SSH_PORT")
                .ok()
                .and_then(|port| port.parse().ok()),
            identity_file: std::env::var("AIDAEMON_TEST_SSH_IDENTITY_FILE").ok(),
            extra_args: std::env::var("AIDAEMON_TEST_SSH_EXTRA_ARGS")
                .ok()
                .and_then(|args| shell_words::split(&args).ok())
                .unwrap_or_default(),
            ..SshExecutionConfig::default()
        };
        let result = async {
            let backend = SshBackend::new(&config, None, false).await?;
            backend_contract(&backend).await?;
            let cleanup = backend
                .execute(
                    ExecutionRequest::shell(format!("rm -rf -- {}", shell_quote(&workspace))),
                    Duration::from_secs(30),
                )
                .await?;
            anyhow::ensure!(cleanup.exit_code == 0, "SSH test workspace cleanup failed");
            Ok::<_, anyhow::Error>(())
        }
        .await;
        result.unwrap();
    }
}
