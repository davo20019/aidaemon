//! The helper every process adapter runs its program with: a scrubbed
//! environment (no daemon secrets), argv only (no shell), stderr kept in a
//! small ring for the `Down` reason, kill-on-drop, and a pid file so a child
//! left behind by a crash or an `exec` restart is stopped before the next
//! one starts. The child runs in the foreground of the daemon: it lives and
//! dies with it.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{oneshot, watch};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::{Exposure, ExposureError, ExposureHealth};

/// Environment variables a tunnel program may need; everything else
/// (provider keys, tokens) is left out.
const KEPT_ENV: &[&str] = &[
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CONFIG_HOME",
    "XDG_RUNTIME_DIR",
    "SystemRoot",
    "SYSTEMROOT",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "ComSpec",
    "PATHEXT",
];

/// Stderr lines kept for a failure reason.
const RING: usize = 4;

/// A program to run: an absolute path and its arguments.
#[derive(Debug, Clone)]
pub(crate) struct ProcessSpec {
    pub program: PathBuf,
    pub args: Vec<String>,
}

impl ProcessSpec {
    fn label(&self) -> String {
        self.program
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "program".to_string())
    }

    fn command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.args).env_clear();
        for key in KEPT_ENV {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        command
    }

    /// Run to completion with a timeout; stdout on success.
    pub async fn output(&self, within: Duration) -> Result<String, String> {
        let mut command = self.command();
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let output = tokio::time::timeout(within, command.output())
            .await
            .map_err(|_| {
                format!(
                    "{} did not answer within {}s",
                    self.label(),
                    within.as_secs()
                )
            })?
            .map_err(|error| format!("could not run {}: {error}", self.label()))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "{} failed ({}): {}",
                self.label(),
                output.status,
                stderr.lines().last().unwrap_or("").trim()
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

/// `program` as given when it is a path, else found on `PATH`.
pub(crate) fn find_program(program: &str) -> Option<PathBuf> {
    let program = program.trim();
    if program.is_empty() {
        return None;
    }
    let as_path = Path::new(program);
    if as_path.components().count() > 1 || as_path.is_absolute() {
        return as_path.is_file().then(|| as_path.to_path_buf());
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(program);
        if candidate.is_file() {
            return Some(candidate);
        }
        #[cfg(windows)]
        {
            let exe = dir.join(format!("{program}.exe"));
            if exe.is_file() {
                return Some(exe);
            }
        }
    }
    None
}

/// `program`, or `NotInstalled` with how to install it.
pub(crate) fn require_program(program: &str, hint: &str) -> Result<PathBuf, ExposureError> {
    find_program(program).ok_or_else(|| ExposureError::NotInstalled {
        program: program.to_string(),
        hint: hint.to_string(),
    })
}

/// Which stream a line came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stream {
    Stdout,
    Stderr,
}

/// Reads the child's output and says when its state changes.
pub(crate) trait Watcher: Send + 'static {
    fn on_line(&mut self, stream: Stream, line: &str) -> Option<ExposureHealth>;
}

pub(crate) struct ProcessOptions {
    /// `Down` when the child reports nothing `Up` within this.
    pub startup_timeout: Duration,
    /// When the child is still running after this without having said it is
    /// up, report this state (for programs that print nothing when ready).
    pub assume_up_after: Option<(Duration, ExposureHealth)>,
    /// Where the child's pid is kept.
    pub pidfile: Option<PathBuf>,
    /// Run after the child stopped (best effort).
    pub cleanup: Option<ProcessSpec>,
}

/// A running child as an [`Exposure`].
pub(crate) struct ProcessExposure {
    rx: watch::Receiver<ExposureHealth>,
    stop: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
}

#[async_trait]
impl Exposure for ProcessExposure {
    fn watch(&self) -> watch::Receiver<ExposureHealth> {
        self.rx.clone()
    }

    async fn stop(mut self: Box<Self>) -> Result<(), ExposureError> {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(task) = self.task.take() {
            task.await
                .map_err(|error| ExposureError::Failed(format!("exposure task: {error}")))?;
        }
        Ok(())
    }
}

/// Stop a child left by an earlier daemon (crash or `exec` restart): the pid
/// in `pidfile`, only when that process still runs `program`.
async fn reap_stale(pidfile: &Path, program: &Path) {
    let Ok(text) = tokio::fs::read_to_string(pidfile).await else {
        return;
    };
    let _ = tokio::fs::remove_file(pidfile).await;
    #[cfg(unix)]
    {
        let Some(pid) = text
            .lines()
            .next()
            .and_then(|l| l.trim().parse::<u32>().ok())
        else {
            return;
        };
        let Some(name) = program
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
        else {
            return;
        };
        let comm = Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "comm="])
            .output()
            .await;
        let running_it = comm
            .ok()
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
            .is_some_and(|comm| !comm.is_empty() && comm.ends_with(&name));
        if running_it {
            warn!(pid, program = %name, "Stopping a tunnel left by an earlier daemon");
            let _ = Command::new("kill")
                .args(["-TERM", &pid.to_string()])
                .status()
                .await;
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (text, program);
    }
}

/// SIGTERM, then SIGKILL after 5 s (Windows: kill).
async fn terminate(child: &mut Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        let _ = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status()
            .await;
        if tokio::time::timeout(Duration::from_secs(5), child.wait())
            .await
            .is_ok()
        {
            return;
        }
    }
    let _ = child.kill().await;
}

fn with_stderr(reason: String, ring: &VecDeque<String>) -> String {
    if ring.is_empty() {
        return reason;
    }
    let tail: Vec<&str> = ring.iter().map(String::as_str).collect();
    format!("{reason}: {}", tail.join(" | "))
}

enum Event {
    Stop,
    Line(Stream, Option<String>),
    Exited(String),
    Timeout,
    AssumeUp,
}

/// Start `spec` and watch it until it exits or is stopped.
pub(crate) async fn spawn(
    spec: ProcessSpec,
    mut watcher: impl Watcher,
    options: ProcessOptions,
) -> Result<ProcessExposure, ExposureError> {
    if let Some(pidfile) = &options.pidfile {
        reap_stale(pidfile, &spec.program).await;
    }
    let label = spec.label();
    let mut command = spec.command();
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command
        .spawn()
        .map_err(|error| ExposureError::Failed(format!("could not start {label}: {error}")))?;
    if let (Some(pidfile), Some(pid)) = (&options.pidfile, child.id()) {
        if let Some(dir) = pidfile.parent() {
            let _ = tokio::fs::create_dir_all(dir).await;
        }
        let _ = tokio::fs::write(pidfile, format!("{pid}\n")).await;
    }
    let mut stdout = BufReader::new(child.stdout.take().expect("piped")).lines();
    let mut stderr = BufReader::new(child.stderr.take().expect("piped")).lines();
    let (tx, rx) = watch::channel(ExposureHealth::Starting);
    let (stop_tx, mut stop_rx) = oneshot::channel::<()>();

    let task = tokio::spawn(async move {
        let mut ring: VecDeque<String> = VecDeque::new();
        let mut up = false;
        let mut stdout_open = true;
        let mut stderr_open = true;
        let timeout = tokio::time::sleep(options.startup_timeout);
        tokio::pin!(timeout);
        let (assume_after, assumed) = match options.assume_up_after {
            Some((after, health)) => (after, Some(health)),
            None => (Duration::from_secs(u64::MAX / 4), None),
        };
        let assume = tokio::time::sleep(assume_after);
        tokio::pin!(assume);
        let final_health = loop {
            let event = tokio::select! {
                _ = &mut stop_rx => Event::Stop,
                line = stdout.next_line(), if stdout_open => Event::Line(Stream::Stdout, line.ok().flatten()),
                line = stderr.next_line(), if stderr_open => Event::Line(Stream::Stderr, line.ok().flatten()),
                status = child.wait() => Event::Exited(match status {
                    Ok(status) => status.to_string(),
                    Err(error) => error.to_string(),
                }),
                _ = &mut timeout, if !up => Event::Timeout,
                _ = &mut assume, if !up && assumed.is_some() => Event::AssumeUp,
            };
            match event {
                Event::Stop => {
                    terminate(&mut child).await;
                    break ExposureHealth::Down {
                        reason: "stopped".to_string(),
                    };
                }
                Event::Line(stream, None) => match stream {
                    Stream::Stdout => stdout_open = false,
                    Stream::Stderr => stderr_open = false,
                },
                Event::Line(stream, Some(line)) => {
                    if stream == Stream::Stderr {
                        debug!(program = %label, line = %line.chars().take(200).collect::<String>(), "tunnel stderr");
                        if ring.len() == RING {
                            ring.pop_front();
                        }
                        ring.push_back(line.chars().take(160).collect());
                    }
                    match watcher.on_line(stream, &line) {
                        Some(ExposureHealth::Down { reason }) => {
                            terminate(&mut child).await;
                            break ExposureHealth::Down { reason };
                        }
                        Some(health) => {
                            up |= matches!(
                                health,
                                ExposureHealth::Up(_) | ExposureHealth::Degraded { .. }
                            );
                            tx.send_replace(health);
                        }
                        None => {}
                    }
                }
                Event::Exited(status) => {
                    break ExposureHealth::Down {
                        reason: with_stderr(format!("{label} exited ({status})"), &ring),
                    };
                }
                Event::Timeout => {
                    terminate(&mut child).await;
                    break ExposureHealth::Down {
                        reason: with_stderr(
                            format!(
                                "{label} did not come up within {}s",
                                options.startup_timeout.as_secs()
                            ),
                            &ring,
                        ),
                    };
                }
                Event::AssumeUp => {
                    if let Some(health) = assumed.clone() {
                        up = true;
                        tx.send_replace(health);
                    }
                }
            }
        };
        if let Some(cleanup) = &options.cleanup {
            if let Err(error) = cleanup.output(Duration::from_secs(10)).await {
                warn!(program = %label, %error, "Tunnel cleanup failed");
            }
        }
        if let Some(pidfile) = &options.pidfile {
            let _ = tokio::fs::remove_file(pidfile).await;
        }
        tx.send_replace(final_health);
    });

    Ok(ProcessExposure {
        rx,
        stop: Some(stop_tx),
        task: Some(task),
    })
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::{OriginStability, PublicEndpoint};

    struct FirstLineUp;
    impl Watcher for FirstLineUp {
        fn on_line(&mut self, stream: Stream, line: &str) -> Option<ExposureHealth> {
            (stream == Stream::Stdout).then(|| {
                ExposureHealth::Up(PublicEndpoint {
                    origin: line.trim().to_string(),
                    stability: OriginStability::Stable,
                })
            })
        }
    }

    fn sh(script: &str) -> ProcessSpec {
        ProcessSpec {
            program: find_program("sh").unwrap(),
            args: vec!["-c".into(), script.into()],
        }
    }

    fn options(pidfile: Option<PathBuf>) -> ProcessOptions {
        ProcessOptions {
            startup_timeout: Duration::from_secs(5),
            assume_up_after: None,
            pidfile,
            cleanup: None,
        }
    }

    async fn wait_for(
        rx: &mut watch::Receiver<ExposureHealth>,
        check: impl Fn(&ExposureHealth) -> bool,
    ) -> ExposureHealth {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let health = rx.borrow_and_update().clone();
                if check(&health) {
                    return health;
                }
                if rx.changed().await.is_err() {
                    return rx.borrow().clone();
                }
            }
        })
        .await
        .expect("state reached")
    }

    #[tokio::test]
    async fn reports_up_then_down_with_stderr_on_exit() {
        let exposure = spawn(
            sh("echo https://a.example.com; echo boom >&2; sleep 0.2; exit 3"),
            FirstLineUp,
            options(None),
        )
        .await
        .unwrap();
        let mut rx = exposure.watch();
        wait_for(&mut rx, |h| matches!(h, ExposureHealth::Up(_))).await;
        let down = wait_for(&mut rx, |h| matches!(h, ExposureHealth::Down { .. })).await;
        let ExposureHealth::Down { reason } = down else {
            unreachable!()
        };
        assert!(
            reason.contains("exited") && reason.contains("boom"),
            "{reason}"
        );
    }

    #[tokio::test]
    async fn scrubs_the_environment() {
        std::env::set_var("AIDAEMON_SYNTHETIC_SECRET", "s3cr3t");
        let exposure = spawn(
            sh("echo \"x${AIDAEMON_SYNTHETIC_SECRET}x\"; sleep 5"),
            FirstLineUp,
            options(None),
        )
        .await
        .unwrap();
        let mut rx = exposure.watch();
        let up = wait_for(&mut rx, |h| matches!(h, ExposureHealth::Up(_))).await;
        assert_eq!(
            up,
            ExposureHealth::Up(PublicEndpoint {
                origin: "xx".into(),
                stability: OriginStability::Stable
            })
        );
        Box::new(exposure).stop().await.unwrap();
    }

    #[tokio::test]
    async fn stop_kills_the_child_and_removes_the_pidfile() {
        let dir = tempfile::tempdir().unwrap();
        let pidfile = dir.path().join("svc.pid");
        let exposure = spawn(
            sh("echo https://a.example.com; exec sleep 60"),
            FirstLineUp,
            options(Some(pidfile.clone())),
        )
        .await
        .unwrap();
        let mut rx = exposure.watch();
        wait_for(&mut rx, |h| matches!(h, ExposureHealth::Up(_))).await;
        let pid: u32 = std::fs::read_to_string(&pidfile)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        Box::new(exposure).stop().await.unwrap();
        assert!(!pidfile.exists());
        let alive = std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .unwrap()
            .success();
        assert!(!alive, "child {pid} still running");
    }

    #[tokio::test]
    async fn times_out_when_never_up() {
        let mut opts = options(None);
        opts.startup_timeout = Duration::from_millis(200);
        let exposure = spawn(sh("echo waiting >&2; sleep 30"), FirstLineUp, opts)
            .await
            .unwrap();
        let mut rx = exposure.watch();
        let down = wait_for(&mut rx, |h| matches!(h, ExposureHealth::Down { .. })).await;
        assert!(format!("{down:?}").contains("did not come up"));
    }

    #[tokio::test]
    async fn missing_program_is_not_installed() {
        let error = require_program("aidaemon-no-such-tunnel", "install it").unwrap_err();
        assert!(matches!(error, ExposureError::NotInstalled { .. }));
    }
}
