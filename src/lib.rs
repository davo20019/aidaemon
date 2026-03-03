mod agent;
mod agent_handoff;
mod channels;
mod config;
mod conversation;
mod core;
mod cron_utils;
mod daemon;
mod dashboard;
mod db;
#[allow(dead_code)]
mod events;
mod execution_policy;
mod goal_tokens;
#[allow(dead_code)]
mod health;
mod heartbeat;
mod llm_markers;
mod llm_runtime;
mod mcp;
mod memory;
mod oauth;
#[allow(dead_code)]
mod plans;
mod providers;
mod queue_policy;
mod queue_telemetry;
mod router;
mod session;
mod skills;
mod startup;
mod state;
mod tasks;
#[cfg(feature = "terminal-bridge")]
mod terminal_bridge;
mod token_alerts;
mod tools;
mod traits;
mod triggers;
mod types;
mod updater;
pub mod utils;
mod wizard;

#[cfg(test)]
mod integration_tests;
#[cfg(test)]
mod live_e2e_tests;
#[cfg(test)]
mod testing;

use std::path::{Path, PathBuf};
#[cfg(not(feature = "terminal-bridge"))]
use std::process::Stdio;

use tracing_subscriber::EnvFilter;

const SUPPORTED_TERMINAL_AGENTS: &[&str] = &["codex", "claude", "gemini", "opencode"];
const MAX_AGENT_LAUNCH_ARGS: usize = 24;
const MAX_AGENT_LAUNCH_ARG_CHARS: usize = 256;

pub fn run() -> anyhow::Result<()> {
    // Load environment file.
    // - Default: .env discovered from current working directory and parents.
    // - Override: AIDAEMON_ENV_FILE=/absolute/path/to/envfile
    if let Ok(path) = std::env::var("AIDAEMON_ENV_FILE") {
        if !path.trim().is_empty() {
            if let Err(e) = dotenvy::from_path(&path) {
                eprintln!(
                    "Warning: failed to load AIDAEMON_ENV_FILE '{}': {}",
                    path, e
                );
            }
        } else {
            let _ = dotenvy::dotenv();
        }
    } else {
        let _ = dotenvy::dotenv();
    }

    // Tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            if cfg!(feature = "browser") {
                EnvFilter::new("info,chromiumoxide=off")
            } else {
                EnvFilter::new("info")
            }
        }))
        .init();

    let config_path = PathBuf::from("config.toml");

    // Handle CLI arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        match args[1].as_str() {
            "--version" | "-V" => {
                println!("aidaemon {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--help" | "-h" => {
                println!("aidaemon {}", env!("CARGO_PKG_VERSION"));
                println!("{}\n", env!("CARGO_PKG_DESCRIPTION"));
                println!("Usage: aidaemon [COMMAND]\n");
                println!("Commands:");
                println!("  install-service       Install as a system service (launchd/systemd)");
                println!("  check-update          Check for available updates");
                println!("  migrate [--config]    Run DB migrations and exit");
                println!(
                    "  browser login          Launch Chrome to log into services for the agent"
                );
                println!("  attach <code>         Attach native terminal to /agent session");
                println!("  agent attach <code>   Legacy alias for attach");
                println!("  share [session_id]    Generate a Telegram /agent resume code");
                println!(
                    "  agent start <agent>   Launch codex/claude/gemini/opencode via aidaemon"
                );
                println!("  codex [cwd] [-- ...]  Shortcut for: aidaemon agent start codex ...");
                println!("  claude [cwd] [-- ...] Shortcut for: aidaemon agent start claude ...");
                println!("  gemini [cwd] [-- ...] Shortcut for: aidaemon agent start gemini ...");
                println!(
                    "  opencode [cwd] [-- ...] Shortcut for: aidaemon agent start opencode ..."
                );
                println!("  keychain set <key>    Store a secret in the OS keychain");
                println!("  keychain get <key>    Retrieve a secret from the OS keychain");
                println!("  keychain delete <key> Remove a secret from the OS keychain");
                println!("\nOptions:");
                println!("  -h, --help       Print help");
                println!("  -V, --version    Print version");
                return Ok(());
            }
            "install-service" => {
                return daemon::install_service();
            }
            "check-update" => {
                println!("Checking for updates...");
                match updater::Updater::check_for_update() {
                    Ok(Some((version, _notes))) => {
                        println!(
                            "Update available: v{} -> v{}",
                            env!("CARGO_PKG_VERSION"),
                            version
                        );
                    }
                    Ok(None) => {
                        println!("aidaemon is up to date (v{}).", env!("CARGO_PKG_VERSION"));
                    }
                    Err(e) => {
                        eprintln!("Update check failed: {}", e);
                        std::process::exit(1);
                    }
                }
                return Ok(());
            }
            "migrate" => {
                return handle_migrate_command(&args[2..], config_path.as_path());
            }
            "browser" => {
                return crate::handle_browser_command(&args[2..], config_path.as_path());
            }
            "agent" => {
                return handle_agent_command(&args[2..]);
            }
            "attach" => {
                return handle_agent_command(&args[1..]);
            }
            "share" => {
                let mut forwarded = Vec::with_capacity(args.len().saturating_sub(1));
                forwarded.push("share".to_string());
                forwarded.extend(args.iter().skip(2).cloned());
                return handle_agent_command(&forwarded);
            }
            "keychain" => {
                return handle_keychain_command(&args[2..]);
            }
            other => {
                if let Some(agent) = normalize_terminal_agent_name(other) {
                    return launch_terminal_agent(agent, &args[2..]);
                }
                anyhow::bail!(
                    "Unknown command: {}. Run `aidaemon --help` for available commands.",
                    other
                );
            }
        }
    }

    // Wizard: if no config.toml, run interactive setup
    if !config_path.exists() {
        let start_now = wizard::run_wizard(&config_path)?;
        if !start_now {
            return Ok(());
        }
        // Fall through to load config and start the daemon
    }

    // Load config — if corrupted, try restoring from .lastgood first (proven
    // working config), then fall back to .bak, .bak.1, .bak.2 in order.
    let config = match config::AppConfig::load(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Config load failed: {}", e);

            let candidates = [
                config_path.with_extension("toml.lastgood"),
                config_path.with_extension("toml.bak"),
                config_path.with_extension("toml.bak.1"),
                config_path.with_extension("toml.bak.2"),
            ];

            let mut restored = false;
            for candidate in &candidates {
                if candidate.exists() {
                    eprintln!("Trying restore from {}...", candidate.display());
                    if let Ok(()) = std::fs::copy(candidate, &config_path).map(|_| ()) {
                        if config::AppConfig::load(&config_path).is_ok() {
                            eprintln!("Restored config from {}", candidate.display());
                            restored = true;
                            break;
                        }
                    }
                }
            }

            if !restored {
                return Err(e);
            }

            // We know this succeeds because we just tested it in the loop above
            config::AppConfig::load(&config_path)?
        }
    };

    // Run async
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(crate::core::run(config, config_path))
}

#[allow(unreachable_code)]
fn handle_browser_command(
    args: &[String],
    #[allow(unused)] config_path: &Path,
) -> anyhow::Result<()> {
    let action = args.first().map(|s| s.as_str()).unwrap_or("");
    match action {
        "login" => {
            #[cfg(not(feature = "browser"))]
            {
                eprintln!("Browser support is not compiled in.");
                eprintln!("Rebuild with: cargo build --features browser");
                std::process::exit(1);
            }

            #[cfg(feature = "browser")]
            {
                let config = config::AppConfig::load(config_path)?;
                let profile_dir = config.browser.user_data_dir.unwrap_or_else(|| {
                    dirs::home_dir()
                        .unwrap_or_else(|| PathBuf::from(".").to_path_buf())
                        .join(".aidaemon")
                        .join("chrome-profile")
                        .to_string_lossy()
                        .into_owned()
                });
                let expanded = shellexpand::tilde(&profile_dir).into_owned();

                // Ensure profile directory exists
                std::fs::create_dir_all(&expanded)?;

                println!("Launching Chrome for login...");
                println!("Profile: {}", expanded);
                println!();
                println!("Log into the services you want the agent to access,");
                println!("then close Chrome when you're done.");
                println!();

                // Find Chrome binary
                let chrome_bin = find_chrome_binary().ok_or_else(|| {
                    anyhow::anyhow!("Chrome not found. Install Chrome or Chromium and try again.")
                })?;

                let profile = config.browser.profile.as_deref().unwrap_or("Default");

                let status = std::process::Command::new(&chrome_bin)
                    .arg(format!("--user-data-dir={}", expanded))
                    .arg(format!("--profile-directory={}", profile))
                    .arg("--no-first-run")
                    .arg("--no-default-browser-check")
                    .status()?;

                if status.success() {
                    println!();
                    println!("Chrome closed. Your login sessions are saved.");
                    println!(
                        "The agent will use them automatically when the browser tool is enabled."
                    );
                } else {
                    eprintln!("Chrome exited with status: {}", status);
                }
            }
        }
        _ => {
            eprintln!("Usage: aidaemon browser login");
            eprintln!();
            eprintln!("Launch Chrome to log into services for the agent.");
            eprintln!("Sessions are saved and reused by the browser tool.");
            std::process::exit(1);
        }
    }
    Ok(())
}

fn handle_migrate_command(args: &[String], default_config_path: &Path) -> anyhow::Result<()> {
    let mut config_path = default_config_path.to_path_buf();

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("Usage: aidaemon migrate [--config PATH]");
                println!();
                println!("Run startup database migrations and exit.");
                println!("Defaults to ./config.toml if --config is not provided.");
                return Ok(());
            }
            "-c" | "--config" => {
                let Some(path) = args.get(i + 1) else {
                    anyhow::bail!("Missing value for --config");
                };
                config_path = PathBuf::from(path);
                i += 2;
            }
            other => {
                anyhow::bail!("Unknown migrate option: {other}");
            }
        }
    }

    if !config_path.exists() {
        anyhow::bail!(
            "Config file not found at {}. Run `aidaemon` once to generate config.toml, or pass --config PATH.",
            config_path.display()
        );
    }

    let config = config::AppConfig::load(&config_path)?;

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(crate::core::run_migrations_only(
            config,
            config_path.clone(),
        ))?;

    println!(
        "Migrations completed successfully (config: {}).",
        config_path.display()
    );
    Ok(())
}

fn handle_agent_command(args: &[String]) -> anyhow::Result<()> {
    let action = args.first().map(|s| s.as_str()).unwrap_or("");
    match action {
        "attach" => {
            let Some(code) = args.get(1).map(|v| v.trim()).filter(|v| !v.is_empty()) else {
                anyhow::bail!("Usage: aidaemon attach <code>");
            };
            #[cfg(feature = "terminal-bridge")]
            {
                crate::terminal_bridge::run_local_attach_cli(code)
            }
            #[cfg(not(feature = "terminal-bridge"))]
            {
                anyhow::bail!(
                    "This binary was built without terminal bridge support. Rebuild with --features terminal-bridge."
                );
            }
        }
        "start" => {
            let Some(raw_agent) = args.get(1).map(|v| v.as_str()) else {
                anyhow::bail!(
                    "Usage: aidaemon agent start <codex|claude|gemini|opencode> [cwd] [-- flags...]"
                );
            };
            let Some(agent) = normalize_terminal_agent_name(raw_agent) else {
                anyhow::bail!(
                    "Unknown agent `{}`. Supported: codex, claude, gemini, opencode.",
                    raw_agent
                );
            };
            launch_terminal_agent(agent, &args[2..])
        }
        "share" => {
            #[cfg(feature = "terminal-bridge")]
            {
                let explicit_session_id = args.get(1).map(|v| v.as_str());
                crate::terminal_bridge::run_local_share_cli(explicit_session_id)
            }
            #[cfg(not(feature = "terminal-bridge"))]
            {
                anyhow::bail!(
                    "This binary was built without terminal bridge support. Rebuild with --features terminal-bridge."
                );
            }
        }
        "-h" | "--help" | "" => {
            println!("Usage:");
            println!("  aidaemon attach <code>");
            println!("  aidaemon agent attach <code>");
            println!("  aidaemon agent start <codex|claude|gemini|opencode> [cwd] [-- flags...]");
            println!("  aidaemon share [session_id]");
            println!();
            println!("Attach your native terminal to an active /agent session started from Telegram Mini App.");
            println!(
                "The one-time code is generated inside Telegram when you tap Continue on Computer."
            );
            println!();
            println!("Shortcuts:");
            println!("  aidaemon codex [cwd] [-- flags...]");
            println!("  aidaemon claude [cwd] [-- flags...]");
            println!("  aidaemon gemini [cwd] [-- flags...]");
            println!("  aidaemon opencode [cwd] [-- flags...]");
            Ok(())
        }
        other => anyhow::bail!("Unknown agent command: {other}"),
    }
}

fn normalize_terminal_agent_name(raw: &str) -> Option<&'static str> {
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "codex" => Some("codex"),
        "claude" => Some("claude"),
        "gemini" => Some("gemini"),
        "opencode" => Some("opencode"),
        _ => None,
    }
}

fn parse_terminal_agent_launch_args(
    raw_args: &[String],
) -> anyhow::Result<(Option<PathBuf>, Vec<String>)> {
    let mut cwd: Option<PathBuf> = None;
    let mut args_start = 0usize;

    if let Some(first) = raw_args.first().map(|v| v.trim()) {
        if first == "--" {
            args_start = 1;
        } else if !first.starts_with("--") {
            if first.is_empty() || first.contains('\0') {
                anyhow::bail!("Invalid working directory.");
            }
            cwd = Some(PathBuf::from(shellexpand::tilde(first).into_owned()));
            args_start = 1;
            if raw_args
                .get(args_start)
                .map(|v| v.as_str() == "--")
                .unwrap_or(false)
            {
                args_start += 1;
            }
        }
    }

    let mut launch_args: Vec<String> = Vec::new();
    for raw in raw_args.iter().skip(args_start) {
        let value = raw.trim();
        if value.is_empty() {
            continue;
        }
        if value.contains('\0') {
            anyhow::bail!("Agent flag contains an invalid NUL byte.");
        }
        if value.len() > MAX_AGENT_LAUNCH_ARG_CHARS {
            anyhow::bail!(
                "Agent flag exceeds {} characters.",
                MAX_AGENT_LAUNCH_ARG_CHARS
            );
        }
        launch_args.push(value.to_string());
        if launch_args.len() > MAX_AGENT_LAUNCH_ARGS {
            anyhow::bail!("Too many agent flags (max {}).", MAX_AGENT_LAUNCH_ARGS);
        }
    }

    Ok((cwd, launch_args))
}

fn launch_terminal_agent(agent: &str, raw_args: &[String]) -> anyhow::Result<()> {
    if !SUPPORTED_TERMINAL_AGENTS.contains(&agent) {
        anyhow::bail!("Unsupported terminal agent: {}", agent);
    }

    let (cwd, launch_args) = parse_terminal_agent_launch_args(raw_args)?;
    #[cfg(feature = "terminal-bridge")]
    {
        crate::terminal_bridge::run_local_start_cli(agent, cwd.as_deref(), &launch_args)
    }
    #[cfg(not(feature = "terminal-bridge"))]
    {
        let mut cmd = std::process::Command::new(agent);
        cmd.args(&launch_args)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        if let Some(dir) = cwd.as_ref() {
            cmd.current_dir(dir);
        }

        let status = cmd.status().map_err(|err| {
            anyhow::anyhow!(
                "Failed to launch `{}`: {}. Ensure the CLI is installed and on PATH.",
                agent,
                err
            )
        })?;

        if status.success() {
            return Ok(());
        }
        if let Some(code) = status.code() {
            std::process::exit(code);
        }
        anyhow::bail!("`{}` terminated unexpectedly.", agent);
    }
}

#[cfg(test)]
mod cli_alias_tests {
    use super::*;

    #[test]
    fn parse_terminal_agent_launch_args_accepts_cwd_and_flags_with_delimiter() {
        let args = vec![
            "~/projects/aidaemon".to_string(),
            "--".to_string(),
            "--dangerously-skip-permissions".to_string(),
            "--chrome".to_string(),
        ];
        let (cwd, flags) = parse_terminal_agent_launch_args(&args).expect("parse");
        assert_eq!(
            cwd,
            Some(PathBuf::from(
                shellexpand::tilde("~/projects/aidaemon").into_owned()
            ))
        );
        assert_eq!(
            flags,
            vec![
                "--dangerously-skip-permissions".to_string(),
                "--chrome".to_string()
            ]
        );
    }

    #[test]
    fn parse_terminal_agent_launch_args_accepts_flag_only_invocation() {
        let args = vec!["--model".to_string(), "gpt-5".to_string()];
        let (cwd, flags) = parse_terminal_agent_launch_args(&args).expect("parse");
        assert_eq!(cwd, None);
        assert_eq!(flags, vec!["--model".to_string(), "gpt-5".to_string()]);
    }

    #[test]
    fn normalize_terminal_agent_name_is_case_insensitive() {
        assert_eq!(normalize_terminal_agent_name("CoDeX"), Some("codex"));
        assert_eq!(normalize_terminal_agent_name("claude"), Some("claude"));
        assert_eq!(normalize_terminal_agent_name("nope"), None);
    }
}

#[cfg(feature = "browser")]
fn find_chrome_binary() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        let candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ];
        for path in &candidates {
            if std::path::Path::new(path).exists() {
                return Some(path.to_string());
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        let candidates = [
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
        ];
        for name in &candidates {
            if std::process::Command::new("which")
                .arg(name)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
            {
                return Some(name.to_string());
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        let candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ];
        for path in &candidates {
            if std::path::Path::new(path).exists() {
                return Some(path.to_string());
            }
        }
    }

    None
}

fn handle_keychain_command(args: &[String]) -> anyhow::Result<()> {
    if args.is_empty() {
        eprintln!("Usage: aidaemon keychain <set|get|delete> <key>");
        eprintln!("\nExamples:");
        eprintln!("  aidaemon keychain set api_key");
        eprintln!("  aidaemon keychain set http_auth_twitter_api_key");
        eprintln!("  aidaemon keychain get api_key");
        eprintln!("  aidaemon keychain delete api_key");
        std::process::exit(1);
    }

    let action = args[0].as_str();
    let key = args.get(1).map(|s| s.as_str());

    match action {
        "set" => {
            let key = key.unwrap_or_else(|| {
                eprintln!("Usage: aidaemon keychain set <key>");
                std::process::exit(1);
            });
            let value = dialoguer::Password::new()
                .with_prompt(format!("Enter value for '{}'", key))
                .with_confirmation("Confirm value", "Values don't match, try again")
                .interact()?;
            config::store_in_keychain(key, &value)?;
            println!("Stored '{}' in OS keychain (service: aidaemon)", key);
        }
        "get" => {
            let key = key.unwrap_or_else(|| {
                eprintln!("Usage: aidaemon keychain get <key>");
                std::process::exit(1);
            });
            match config::resolve_from_keychain(key) {
                Ok(value) => {
                    // Show first 4 and last 4 chars, mask the rest
                    if value.len() > 12 {
                        let masked = format!(
                            "{}{}{}",
                            &value[..4],
                            "*".repeat(value.len() - 8),
                            &value[value.len() - 4..]
                        );
                        println!("{}", masked);
                    } else {
                        println!("{}", "*".repeat(value.len()));
                    }
                }
                Err(e) => {
                    eprintln!("Not found: {}", e);
                    std::process::exit(1);
                }
            }
        }
        "delete" => {
            let key = key.unwrap_or_else(|| {
                eprintln!("Usage: aidaemon keychain delete <key>");
                std::process::exit(1);
            });
            config::delete_from_keychain(key)?;
            println!("Deleted '{}' from OS keychain", key);
        }
        _ => {
            eprintln!(
                "Unknown keychain action: '{}'. Use set, get, or delete.",
                action
            );
            std::process::exit(1);
        }
    }

    Ok(())
}
