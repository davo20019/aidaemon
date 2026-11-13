//! The bubblewrap backend, run for real (Linux only).
//!
//! Each test skips with a message when `bwrap` cannot run here (missing, or
//! unprivileged user namespaces disabled). CI sets `AIDAEMON_REQUIRE_BWRAP=1`,
//! which turns that skip into a failure so the suite cannot pass vacuously.
//! The tests compile on every platform and are ignored off Linux.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use aidaemon_core::{CancellationToken, Tool, ToolContext};
use aidaemon_tools::{BubblewrapSandbox, CommandOutput, Sandbox, SandboxCommand, Shell, Workspace};
use serde_json::json;

/// Sets an environment variable that no other test uses, and removes it
/// again when dropped.
struct ScopedVar(&'static str);

impl ScopedVar {
    fn set(name: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        // SAFETY: test-only, and the name is private to one test. std
        // serializes its own environment reads and writes; foreign code
        // calling getenv concurrently is a risk the test binary accepts.
        unsafe { std::env::set_var(name, value) };
        Self(name)
    }
}

impl Drop for ScopedVar {
    fn drop(&mut self) {
        // SAFETY: as in `ScopedVar::set`.
        unsafe { std::env::remove_var(self.0) };
    }
}

fn bwrap() -> Option<BubblewrapSandbox> {
    let backend = BubblewrapSandbox::new();
    if backend.is_available() {
        return Some(backend);
    }
    let reason = "bubblewrap cannot run here: `bwrap` is missing or unprivileged user \
                  namespaces are disabled (on Ubuntu 24.04: sudo sysctl -w \
                  kernel.apparmor_restrict_unprivileged_userns=0)";
    if std::env::var("AIDAEMON_REQUIRE_BWRAP").is_ok_and(|v| v == "1") {
        panic!("AIDAEMON_REQUIRE_BWRAP=1 but {reason}");
    }
    eprintln!("skipping: {reason}");
    None
}

struct Fixture {
    _dir: tempfile::TempDir,
    root: PathBuf,
}

fn workspace() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(dir.path()).unwrap();
    Fixture { _dir: dir, root }
}

/// A directory outside the workspace and outside `/tmp` (which is a private
/// tmpfs inside the sandbox, so a write there would vanish rather than fail).
fn outside_dir() -> tempfile::TempDir {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/".into());
    tempfile::tempdir_in(home)
        .or_else(|_| tempfile::tempdir_in(std::env::current_dir().unwrap()))
        .unwrap()
}

fn spec(command: impl Into<String>, root: &Path) -> SandboxCommand {
    let mut s = SandboxCommand::new(command, root);
    s.env = vec![
        ("PATH".into(), "/usr/local/bin:/usr/bin:/bin".into()),
        ("LANG".into(), "C".into()),
    ];
    s.timeout = Duration::from_secs(30);
    s
}

async fn run(backend: &BubblewrapSandbox, s: &SandboxCommand) -> CommandOutput {
    backend.run(s).await.expect("bwrap failed to start")
}

/// Interfaces other than loopback listed in the sandbox's own netns.
fn non_loopback_interfaces(proc_net_dev: &str) -> Vec<String> {
    proc_net_dev
        .lines()
        .filter_map(|line| {
            line.split_once(':')
                .map(|(name, _)| name.trim().to_string())
        })
        .filter(|name| !name.is_empty() && name != "lo" && !name.contains(' '))
        .collect()
}

#[tokio::test]
#[cfg_attr(not(target_os = "linux"), ignore = "bubblewrap is Linux-only")]
async fn exit_codes_and_output_come_back() {
    let Some(backend) = bwrap() else { return };
    let ws = workspace();
    let out = run(&backend, &spec("echo out; echo err >&2; exit 7", &ws.root)).await;
    assert_eq!(out.exit_code, Some(7), "{}", out.output);
    assert!(
        out.output.contains("out") && out.output.contains("err"),
        "{}",
        out.output
    );
    assert!(!out.timed_out && !out.cancelled);

    let out = run(&backend, &spec("printf ok", &ws.root)).await;
    assert_eq!(out.exit_code, Some(0), "{}", out.output);
    assert_eq!(out.output, "ok");
}

#[tokio::test]
#[cfg_attr(not(target_os = "linux"), ignore = "bubblewrap is Linux-only")]
async fn workspace_is_writable_and_cwd_is_honoured() {
    let Some(backend) = bwrap() else { return };
    let ws = workspace();
    std::fs::create_dir(ws.root.join("sub")).unwrap();
    let mut s = spec(
        "pwd; echo written > made.txt; mkdir -p deep/er && touch deep/er/f",
        &ws.root,
    );
    s.cwd = ws.root.join("sub");
    let out = run(&backend, &s).await;
    assert_eq!(out.exit_code, Some(0), "{}", out.output);
    assert!(
        out.output
            .contains(&ws.root.join("sub").display().to_string()),
        "{}",
        out.output
    );
    assert_eq!(
        std::fs::read_to_string(ws.root.join("sub/made.txt")).unwrap(),
        "written\n"
    );
    assert!(ws.root.join("sub/deep/er/f").exists());
}

#[tokio::test]
#[cfg_attr(not(target_os = "linux"), ignore = "bubblewrap is Linux-only")]
async fn writes_outside_the_workspace_fail() {
    let Some(backend) = bwrap() else { return };
    let ws = workspace();
    let outside = outside_dir();
    let target = outside.path().join("escape.txt");
    let out = run(
        &backend,
        &spec(format!("echo bad > '{}'", target.display()), &ws.root),
    )
    .await;
    assert_ne!(
        out.exit_code,
        Some(0),
        "write outside succeeded: {}",
        out.output
    );
    assert!(!target.exists());

    // /tmp is a private tmpfs: a write there succeeds inside but never
    // reaches the host.
    let marker = format!("aidaemon-bwrap-{}", std::process::id());
    let out = run(
        &backend,
        &spec(
            format!("echo x > /tmp/{marker} && cat /tmp/{marker}"),
            &ws.root,
        ),
    )
    .await;
    assert_eq!(out.exit_code, Some(0), "{}", out.output);
    assert!(!Path::new("/tmp").join(&marker).exists());

    // A workspace under the host's /tmp is still the real directory.
    let out = run(&backend, &spec("touch here", &ws.root)).await;
    assert_eq!(out.exit_code, Some(0), "{}", out.output);
    assert!(ws.root.join("here").exists());
}

#[tokio::test]
#[cfg_attr(not(target_os = "linux"), ignore = "bubblewrap is Linux-only")]
async fn reads_of_denied_paths_fail() {
    let Some(backend) = bwrap() else { return };
    let ws = workspace();
    let secrets = outside_dir();
    std::fs::write(secrets.path().join("key"), "synthetic-secret").unwrap();
    let lone = secrets.path().join("lone");
    std::fs::write(&lone, "lone-secret").unwrap();
    // Inside the workspace too: the deny applies after the workspace bind.
    std::fs::create_dir(ws.root.join("private")).unwrap();
    std::fs::write(ws.root.join("private/p"), "ws-secret").unwrap();

    // Readable without the deny, so the denial below is the sandbox's doing.
    let open = spec(format!("cat '{}/key'", secrets.path().display()), &ws.root);
    let out = run(&backend, &open).await;
    assert!(out.output.contains("synthetic-secret"), "{}", out.output);

    let mut s = spec(format!("cat '{}/key'", secrets.path().display()), &ws.root);
    s.deny_read = vec![secrets.path().to_path_buf()];
    let out = run(&backend, &s).await;
    assert_ne!(out.exit_code, Some(0), "{}", out.output);
    assert!(!out.output.contains("synthetic-secret"), "{}", out.output);

    // A denied file reads as empty.
    let mut s = spec(format!("cat '{}'", lone.display()), &ws.root);
    s.deny_read = vec![lone.clone()];
    let out = run(&backend, &s).await;
    assert!(!out.output.contains("lone-secret"), "{}", out.output);

    let mut s = spec("cat private/p", &ws.root);
    s.deny_read = vec![ws.root.join("private")];
    let out = run(&backend, &s).await;
    assert_ne!(out.exit_code, Some(0), "{}", out.output);
    assert!(!out.output.contains("ws-secret"), "{}", out.output);
    // Hidden, not deleted.
    assert!(ws.root.join("private/p").exists());
}

#[tokio::test]
#[cfg_attr(not(target_os = "linux"), ignore = "bubblewrap is Linux-only")]
async fn network_is_off_unless_allowed() {
    let Some(backend) = bwrap() else { return };
    let ws = workspace();
    // /proc is mounted inside the sandbox, so /proc/net/dev lists the
    // sandbox's own network namespace.
    let out = run(&backend, &spec("cat /proc/net/dev", &ws.root)).await;
    assert_eq!(out.exit_code, Some(0), "{}", out.output);
    assert!(out.output.contains("lo:"), "{}", out.output);
    assert!(
        non_loopback_interfaces(&out.output).is_empty(),
        "network should be unshared: {}",
        out.output
    );

    let mut s = spec("cat /proc/net/dev", &ws.root);
    s.allow_network = true;
    let out = run(&backend, &s).await;
    assert_eq!(out.exit_code, Some(0), "{}", out.output);
    let host = std::fs::read_to_string("/proc/net/dev").unwrap_or_default();
    if !non_loopback_interfaces(&host).is_empty() {
        assert!(
            !non_loopback_interfaces(&out.output).is_empty(),
            "allowed network should see the host's interfaces: {}",
            out.output
        );
    }
}

#[tokio::test]
#[cfg_attr(not(target_os = "linux"), ignore = "bubblewrap is Linux-only")]
async fn environment_is_only_the_spec() {
    let Some(backend) = bwrap() else { return };
    let ws = workspace();
    let _secret = ScopedVar::set("AIDAEMON_BWRAP_TEST_SECRET", "synthetic-secret");
    let mut s = spec(
        "echo \"[$AIDAEMON_BWRAP_TEST_SECRET][$GREETING]\"",
        &ws.root,
    );
    s.env.push(("GREETING".into(), "hi".into()));
    let out = run(&backend, &s).await;
    assert!(out.output.contains("[][hi]"), "{}", out.output);
}

#[tokio::test]
#[cfg_attr(not(target_os = "linux"), ignore = "bubblewrap is Linux-only")]
async fn timeout_and_cancel_kill_the_sandbox() {
    let Some(backend) = bwrap() else { return };
    let ws = workspace();
    let mut s = spec("echo start; sleep 30; echo never", &ws.root);
    s.timeout = Duration::from_secs(1);
    let started = std::time::Instant::now();
    let out = run(&backend, &s).await;
    assert!(started.elapsed() < Duration::from_secs(10));
    assert!(out.timed_out && out.exit_code.is_none(), "{out:?}");
    assert!(out.output.contains("start") && !out.output.contains("never"));

    let mut s = spec("sleep 30", &ws.root);
    let cancel = CancellationToken::new();
    s.cancel = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        cancel.cancel();
    });
    let started = std::time::Instant::now();
    let out = run(&backend, &s).await;
    assert!(started.elapsed() < Duration::from_secs(10));
    assert!(out.cancelled, "{out:?}");
}

/// The default shell tool picks bubblewrap on Linux and stays confined.
#[tokio::test]
#[cfg_attr(not(target_os = "linux"), ignore = "bubblewrap is Linux-only")]
async fn default_shell_tool_runs_in_bubblewrap() {
    let Some(_) = bwrap() else { return };
    if std::env::var_os(aidaemon_tools::SANDBOX_ENV).is_some() {
        eprintln!("skipping: {} is set", aidaemon_tools::SANDBOX_ENV);
        return;
    }
    let ws = workspace();
    let outside = outside_dir();
    let tool: Arc<dyn Tool> = Arc::new(Shell::sandboxed(Workspace::new(&ws.root).unwrap()));
    let ctx = ToolContext::new("run", "call");
    let out = tool
        .call(
            json!({"command": format!(
                "echo hi > inside.txt; echo bad > '{}/x.txt'; cat inside.txt",
                outside.path().display()
            )}),
            &ctx,
        )
        .await
        .unwrap();
    assert_eq!(out.details["backend"], "bubblewrap", "{}", out.content);
    assert!(out.content.contains("hi"), "{}", out.content);
    assert!(ws.root.join("inside.txt").exists());
    assert!(!outside.path().join("x.txt").exists());
    assert!(tool.spec().description.contains("no network access"));
}
