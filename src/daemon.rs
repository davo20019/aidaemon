use std::sync::Arc;

use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};
use serde_json::json;
use sqlx::SqlitePool;
use tracing::info;

#[derive(Clone)]
struct HealthServerState {
    pool: SqlitePool,
    event_store: Arc<crate::events::EventStore>,
}

/// Start the health check HTTP server.
///
/// Binds to `bind_addr` (default "127.0.0.1") to avoid exposing the
/// endpoint on all interfaces. Set to "0.0.0.0" in config if external
/// access is needed.
pub async fn start_health_server(
    port: u16,
    bind_addr: &str,
    pool: SqlitePool,
    event_store: Arc<crate::events::EventStore>,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/health", get(health_handler))
        .with_state(HealthServerState { pool, event_store });

    let ip: std::net::IpAddr = bind_addr
        .parse()
        .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
    let addr = std::net::SocketAddr::new(ip, port);
    info!("Health server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_handler(State(state): State<HealthServerState>) -> impl IntoResponse {
    health_response(&state.pool, Some(state.event_store.as_ref())).await
}

/// Build the shared liveness/readiness response used by both the health-only
/// server and dashboard. No active provider request is made: readiness comes
/// from typed durable outcomes and database reachability.
pub async fn health_response(
    pool: &SqlitePool,
    event_store: Option<&crate::events::EventStore>,
) -> (StatusCode, Json<serde_json::Value>) {
    let database_ready = sqlx::query_scalar::<_, i64>("SELECT 1")
        .fetch_one(pool)
        .await
        .is_ok();
    let readiness = if let Some(event_store) = event_store {
        event_store
            .get_provider_readiness(chrono::Utc::now() - chrono::Duration::hours(24))
            .await
            .ok()
    } else {
        None
    };
    let recent = readiness.as_ref().is_some_and(|readiness| {
        readiness
            .last_call_at
            .is_some_and(|at| at >= chrono::Utc::now() - chrono::Duration::minutes(15))
    });
    let provider_unavailable = recent
        && readiness
            .as_ref()
            .is_some_and(crate::events::ProviderReadiness::is_unavailable);
    let healthy = database_ready && !provider_unavailable;
    let provider_status = match readiness.as_ref() {
        Some(_) if provider_unavailable => "unavailable",
        Some(readiness) if readiness.observed_calls == 0 => "unknown",
        Some(_) if !recent => "stale",
        Some(readiness) if readiness.latest_failed => "recovering",
        Some(_) => "ready",
        None => "unknown",
    };
    let status = if healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(json!({
            "status": if healthy { "ok" } else { "degraded" },
            "components": {
                "database": { "status": if database_ready { "ready" } else { "unavailable" } },
                "provider": {
                    "status": provider_status,
                    "last_call_at": readiness.as_ref().and_then(|value| value.last_call_at),
                    "consecutive_failures": readiness.as_ref().map(|value| value.consecutive_failures).unwrap_or_default(),
                    "failure_kind": readiness.as_ref().and_then(|value| value.failure_kind),
                    "provider_status": readiness.as_ref().and_then(|value| value.provider_status),
                }
            }
        })),
    )
}

/// The account the legacy system unit runs as: the user who ran it under
/// `sudo`. A system unit with no `User=` runs the daemon (its terminal, MCP
/// servers and coding agents) as root, from a binary and config the user
/// can rewrite, so a unit for root itself is refused.
#[cfg(any(target_os = "linux", test))]
fn legacy_unit_user(sudo_user: Option<&str>) -> anyhow::Result<String> {
    match sudo_user.map(str::trim) {
        Some(user) if !user.is_empty() && user != "root" && !user.contains(char::is_whitespace) => {
            Ok(user.to_string())
        }
        _ => anyhow::bail!(
            "Run this with sudo from the account that should own the daemon, or use \
             `aidaemon service install`, which installs a per-user unit and needs no root. \
             AIDaemon does not install a system unit that runs as root."
        ),
    }
}

#[cfg(any(target_os = "linux", test))]
fn legacy_system_unit(exe: &std::path::Path, working_dir: &std::path::Path, user: &str) -> String {
    format!(
        r#"[Unit]
Description=aidaemon - AI personal daemon
After=network.target

[Service]
Type=simple
User={user}
ExecStart={}
WorkingDirectory={}
Restart=on-failure
RestartSec=10
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
"#,
        exe.display(),
        working_dir.display()
    )
}

/// Generate and write a systemd service file (Linux).
#[cfg(target_os = "linux")]
pub fn install_service() -> anyhow::Result<()> {
    let exe = std::env::current_exe()?;
    let working_dir = std::env::current_dir()?;
    let user = legacy_unit_user(std::env::var("SUDO_USER").ok().as_deref())?;

    let unit = legacy_system_unit(&exe, &working_dir, &user);

    let path = "/etc/systemd/system/aidaemon.service";
    std::fs::write(path, unit)?;
    println!("Service file written to {}", path);
    println!("Run: sudo systemctl daemon-reload && sudo systemctl enable --now aidaemon");
    Ok(())
}

/// Bundle identifier used for the macOS `.app` and TCC permission grants.
///
/// macOS ties Accessibility / Screen Recording grants to a *stable code
/// identity*. A bare Mach-O binary launched by launchd has no durable identity
/// (ad-hoc signatures change on every rebuild and background agents can't be
/// granted reliably), which forces the `computer_use` permissions to be
/// re-granted constantly. Packaging the daemon as a signed `.app` with this
/// fixed bundle id makes grants persist across rebuilds — see
/// `COMPUTER_USE_MACOS.md`.
#[cfg(target_os = "macos")]
pub const MACOS_BUNDLE_ID: &str = "ai.aidaemon";

/// Generate and write a launchd plist file (macOS).
///
/// On macOS the daemon is installed as a signed `~/Applications/aidaemon.app`
/// bundle and launchd is pointed at the binary *inside* the bundle, so macOS
/// associates the running process with the bundle identifier and the
/// `computer_use` permission grants survive rebuilds.
#[cfg(target_os = "macos")]
pub fn install_service() -> anyhow::Result<()> {
    install_macos_agent(&std::env::current_dir()?)?;
    println!("Run: launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.aidaemon.plist");
    println!("Then grant Accessibility + Screen Recording to aidaemon — see COMPUTER_USE_MACOS.md");
    Ok(())
}

/// Build the signed app bundle and write the launchd agent running it in
/// `working_dir`. Returns the plist's path.
#[cfg(target_os = "macos")]
fn install_macos_agent(working_dir: &std::path::Path) -> anyhow::Result<String> {
    let exe = std::env::current_exe()?;
    let home = std::env::var("HOME")?;

    // Use user-private log directory instead of world-readable /tmp
    let log_dir = format!("{home}/Library/Logs/aidaemon");
    std::fs::create_dir_all(&log_dir)?;

    // From the signed download (AIdaemon.app's helper), the daemon runs where
    // it is: it already has its bundle id and a lasting signature.
    let bundle_bin = match signed_helper(&exe) {
        Some(path) => path,
        None => build_dev_bundle(&exe, &home)?,
    };
    write_launch_agent(&bundle_bin, working_dir, &home, &log_dir)
}

/// Whether this daemon runs from the signed app download (its helper):
/// replacing its file would break the signature, so it is updated with the
/// app.
pub(crate) fn in_signed_app() -> bool {
    #[cfg(target_os = "macos")]
    return std::env::current_exe()
        .ok()
        .is_some_and(|exe| signed_helper(&exe).is_some());
    #[cfg(not(target_os = "macos"))]
    false
}

/// `exe` when it is the daemon inside a bundle signed with a Developer ID
/// (the downloaded app's helper), not ad hoc.
#[cfg(target_os = "macos")]
fn signed_helper(exe: &std::path::Path) -> Option<std::path::PathBuf> {
    let in_bundle = exe
        .parent()
        .is_some_and(|dir| dir.ends_with("Contents/MacOS"));
    if !in_bundle {
        return None;
    }
    let bundle = exe.parent()?.parent()?.parent()?;
    let info = std::process::Command::new("codesign")
        .args(["-dv"])
        .arg(bundle)
        .output()
        .ok()?;
    let details = String::from_utf8_lossy(&info.stderr);
    let team = details
        .lines()
        .find_map(|line| line.strip_prefix("TeamIdentifier="))?;
    (team != "not set").then(|| exe.to_path_buf())
}

/// A checkout's or a script install's daemon: copied into its own signed
/// `~/Applications/aidaemon.app`, so its permission grants survive rebuilds.
#[cfg(target_os = "macos")]
fn build_dev_bundle(exe: &std::path::Path, home: &str) -> anyhow::Result<std::path::PathBuf> {
    use std::path::PathBuf;
    use std::process::Command;

    // 1. Build the .app bundle skeleton.
    let app_dir = PathBuf::from(home).join("Applications/aidaemon.app");
    let macos_dir = app_dir.join("Contents/MacOS");
    std::fs::create_dir_all(&macos_dir)?;
    let bundle_bin = macos_dir.join("aidaemon");
    std::fs::copy(exe, &bundle_bin)?;

    let info_plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>{bundle_id}</string>
    <key>CFBundleName</key><string>aidaemon</string>
    <key>CFBundleExecutable</key><string>aidaemon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>{version}</string>
    <key>CFBundleShortVersionString</key><string>{version}</string>
    <key>LSUIElement</key><true/>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
</dict>
</plist>
"#,
        bundle_id = MACOS_BUNDLE_ID,
        version = env!("CARGO_PKG_VERSION"),
    );
    std::fs::write(app_dir.join("Contents/Info.plist"), info_plist)?;

    // 2. Code-sign the bundle. Prefer a stable self-signed identity named
    //    "aidaemon-dev" (created by scripts/create-signing-identity.sh) so the
    //    designated requirement is identity-based and survives rebuilds. Fall
    //    back to ad-hoc signing with the fixed bundle id — durable for install-
    //    once users; developers who rebuild should create the identity.
    let have_identity = Command::new("security")
        .args(["find-identity", "-v", "-p", "codesigning"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("aidaemon-dev"))
        .unwrap_or(false);
    let sign_arg = if have_identity { "aidaemon-dev" } else { "-" };
    let sign = Command::new("codesign")
        .args(["-f", "-s", sign_arg, "--identifier", MACOS_BUNDLE_ID])
        .arg("--timestamp=none")
        .arg(&app_dir)
        .output()?;
    if !sign.status.success() {
        anyhow::bail!(
            "codesign failed: {}",
            String::from_utf8_lossy(&sign.stderr).trim()
        );
    }

    println!("Installed signed app bundle: {}", app_dir.display());
    println!(
        "Signed with {}.",
        if have_identity {
            "stable identity 'aidaemon-dev' (grants survive rebuilds)"
        } else {
            "ad-hoc signature (run scripts/create-signing-identity.sh for rebuild-durable grants)"
        }
    );
    Ok(bundle_bin)
}

/// Write the launch agent running `bundle_bin` in `working_dir`. Returns
/// its path.
#[cfg(target_os = "macos")]
fn write_launch_agent(
    bundle_bin: &std::path::Path,
    working_dir: &std::path::Path,
    home: &str,
    log_dir: &str,
) -> anyhow::Result<String> {
    // 3. Point launchd at the binary inside the bundle (NOT wrapped in
    //    caffeinate — a wrapper process muddies the TCC identity). Idle-sleep
    //    prevention is handled by the daemon itself at startup (see
    //    `spawn_macos_keep_awake`).
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{bin}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{wd}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/dev/null</string>
    <key>StandardErrorPath</key>
    <string>{log}/stderr.log</string>
</dict>
</plist>
"#,
        label = MACOS_BUNDLE_ID,
        bin = bundle_bin.display(),
        wd = working_dir.display(),
        log = log_dir,
    );

    let path = format!("{home}/Library/LaunchAgents/ai.aidaemon.plist");
    std::fs::write(&path, plist)?;

    println!("Plist written to {path}");
    println!("Logs will be written to {log_dir}/");
    Ok(path)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn install_service() -> anyhow::Result<()> {
    anyhow::bail!("Use `aidaemon service install` on this platform");
}

/// Where a per-user install keeps its config, database and logs.
pub fn default_home() -> anyhow::Result<std::path::PathBuf> {
    Ok(dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("no home directory"))?
        .join(".aidaemon"))
}

fn run_quiet(program: &str, args: &[&str]) -> anyhow::Result<std::process::Output> {
    std::process::Command::new(program)
        .args(args)
        .output()
        .map_err(|e| anyhow::anyhow!("could not run {program}: {e}"))
}

/// Run the daemon at login for this user, in `dir`, and start it now. No
/// administrator rights: a launchd agent (macOS), a systemd user unit
/// (Linux) or a logon task (Windows). Running it again replaces it.
pub fn install_user_service(dir: &std::path::Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    #[cfg(target_os = "macos")]
    {
        let plist = install_macos_agent(dir)?;
        let uid = run_quiet("id", &["-u"])?;
        let target = format!("gui/{}", String::from_utf8_lossy(&uid.stdout).trim());
        // A loaded agent is replaced.
        let _ = run_quiet(
            "launchctl",
            &["bootout", &format!("{target}/{MACOS_BUNDLE_ID}")],
        );
        let started = run_quiet("launchctl", &["bootstrap", &target, &plist])?;
        anyhow::ensure!(
            started.status.success(),
            "launchctl bootstrap failed: {}",
            String::from_utf8_lossy(&started.stderr).trim()
        );
    }
    #[cfg(target_os = "linux")]
    {
        let exe = std::env::current_exe()?;
        let units = dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("no config directory"))?
            .join("systemd/user");
        std::fs::create_dir_all(&units)?;
        let unit = format!(
            "[Unit]\nDescription=AIDaemon\nAfter=network-online.target\n\n[Service]\nType=simple\n\
             ExecStart={}\nWorkingDirectory={}\nRestart=on-failure\nRestartSec=10\n\n\
             [Install]\nWantedBy=default.target\n",
            exe.display(),
            dir.display()
        );
        std::fs::write(units.join("aidaemon.service"), unit)?;
        run_quiet("systemctl", &["--user", "daemon-reload"])?;
        let started = run_quiet("systemctl", &["--user", "enable", "--now", "aidaemon"])?;
        anyhow::ensure!(
            started.status.success(),
            "systemctl --user enable --now aidaemon failed: {}",
            String::from_utf8_lossy(&started.stderr).trim()
        );
        // A user unit stops at logout unless the user lingers.
        let _ = run_quiet("loginctl", &["enable-linger"]);
    }
    #[cfg(target_os = "windows")]
    {
        let exe = std::env::current_exe()?;
        let command = format!(
            "cmd /c cd /d \"{}\" && \"{}\"",
            dir.display(),
            exe.display()
        );
        let created = run_quiet(
            "schtasks",
            &[
                "/Create", "/F", "/SC", "ONLOGON", "/RL", "LIMITED", "/TN", "AIdaemon", "/TR",
                &command,
            ],
        )?;
        anyhow::ensure!(
            created.status.success(),
            "schtasks /Create failed: {}",
            String::from_utf8_lossy(&created.stderr).trim()
        );
        let _ = run_quiet("schtasks", &["/Run", "/TN", "AIdaemon"])?;
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    anyhow::bail!("Running at login is not supported on this platform");
    #[allow(unreachable_code)]
    Ok(())
}

/// Stop the per-user service and remove it.
pub fn uninstall_user_service() -> anyhow::Result<()> {
    #[cfg(target_os = "macos")]
    {
        let uid = run_quiet("id", &["-u"])?;
        let target = format!("gui/{}", String::from_utf8_lossy(&uid.stdout).trim());
        let _ = run_quiet(
            "launchctl",
            &["bootout", &format!("{target}/{MACOS_BUNDLE_ID}")],
        );
        if let Some(home) = dirs::home_dir() {
            let _ = std::fs::remove_file(home.join("Library/LaunchAgents/ai.aidaemon.plist"));
        }
    }
    #[cfg(target_os = "linux")]
    {
        let _ = run_quiet("systemctl", &["--user", "disable", "--now", "aidaemon"]);
        if let Some(config) = dirs::config_dir() {
            let _ = std::fs::remove_file(config.join("systemd/user/aidaemon.service"));
        }
    }
    #[cfg(target_os = "windows")]
    {
        let _ = run_quiet("schtasks", &["/End", "/TN", "AIdaemon"]);
        let _ = run_quiet("schtasks", &["/Delete", "/F", "/TN", "AIdaemon"]);
    }
    Ok(())
}

/// Prevent macOS idle sleep for the lifetime of the daemon.
///
/// The launchd plist launches the daemon binary directly (not wrapped in
/// `caffeinate`, which would muddy the TCC code identity needed for
/// `computer_use` permissions). To preserve the old keep-awake behaviour, the
/// daemon spawns its own `caffeinate -i -w <pid>` child that exits when the
/// daemon does. Best-effort: a missing `caffeinate` or spawn failure is logged
/// and ignored.
#[cfg(target_os = "macos")]
pub fn spawn_macos_keep_awake() {
    let pid = std::process::id();
    match std::process::Command::new("/usr/bin/caffeinate")
        .args(["-i", "-w", &pid.to_string()])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(_) => info!("macOS keep-awake active (caffeinate -i -w {pid})"),
        Err(e) => tracing::warn!("Could not start caffeinate keep-awake: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventStoreForTests as _;

    #[test]
    fn the_legacy_system_unit_never_runs_as_root() {
        assert!(legacy_unit_user(None).is_err());
        assert!(legacy_unit_user(Some("root")).is_err());
        assert!(legacy_unit_user(Some("")).is_err());
        let user = legacy_unit_user(Some("alice")).unwrap();
        let unit = legacy_system_unit(
            std::path::Path::new("/opt/aidaemon/aidaemon"),
            std::path::Path::new("/home/alice/.aidaemon"),
            &user,
        );
        assert!(unit.contains("\nUser=alice\n"));
    }

    async fn health_fixture() -> (
        SqlitePool,
        Arc<crate::events::EventStore>,
        tempfile::NamedTempFile,
    ) {
        let file = tempfile::NamedTempFile::new().expect("temporary health database");
        let pool = SqlitePool::connect(&format!("sqlite:{}", file.path().display()))
            .await
            .expect("connect health database");
        let event_store = Arc::new(
            crate::events::EventStore::new(pool.clone())
                .await
                .expect("event store"),
        );
        (pool, event_store, file)
    }

    async fn append_provider_outcome(
        event_store: &crate::events::EventStore,
        failed: bool,
        kind: Option<crate::providers::ProviderErrorKind>,
    ) {
        event_store
            .append(crate::events::Event::new(
                "synthetic-health-session",
                crate::events::EventType::LlmCall,
                json!({
                    "task_id": "synthetic-health-task",
                    "model": "synthetic-model",
                    "latency_ms": 1,
                    "failed": failed,
                    "provider_error_kind": kind,
                    "provider_status": if failed { Some(401) } else { None::<u16> }
                }),
            ))
            .await
            .expect("append provider outcome");
    }

    #[tokio::test]
    async fn health_is_degraded_by_typed_auth_failure_and_closed_by_later_success() {
        let (pool, event_store, _file) = health_fixture().await;
        append_provider_outcome(
            event_store.as_ref(),
            true,
            Some(crate::providers::ProviderErrorKind::Auth),
        )
        .await;

        let (status, body) = health_response(&pool, Some(event_store.as_ref())).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body.0["components"]["provider"]["status"], "unavailable");

        append_provider_outcome(event_store.as_ref(), false, None).await;
        let (status, body) = health_response(&pool, Some(event_store.as_ref())).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body.0["components"]["provider"]["status"], "ready");
    }
}
