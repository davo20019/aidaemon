#![recursion_limit = "256"]

mod agent;
mod agent_handoff;
mod channels;
mod cli_agent_flags;
mod config;
mod context_budget;
mod conversation;
mod core;
use aidaemon_work::schedule as cron_utils;
mod daemon;
mod dashboard;
mod db;
use aidaemon_tools::duration_format;
#[allow(dead_code)]
mod events;
mod otel_export;
pub use events::{
    CompletionProofReference, Event, EventType, TaskCompletionProofData, TaskEndData, ToolCallData,
    UserMessageData,
};
pub(crate) mod execution;
mod execution_policy;
use aidaemon_work::goal_tokens;
pub mod harness_eval;
mod health;
mod heartbeat;
mod llm_markers;
mod llm_runtime;
mod logging;
mod mandates {
    //! The mandate policy lives in `aidaemon_work::mandates`.
    #[allow(unused_imports)]
    pub(crate) use aidaemon_work::mandates::{history, policy::*};
    #[allow(unused_imports)]
    pub use aidaemon_work::MANDATE_PROTOCOL_TOOLS;
    #[allow(unused_imports)]
    pub(crate) use aidaemon_work::{admissibility, authority};
}
mod mcp;
mod memory;
pub mod nodes;
mod oauth;
#[allow(unused_imports)]
use aidaemon_work::plans;
mod project_instructions;
mod providers;
use aidaemon_channels::queue::policy as queue_policy;
use aidaemon_channels::queue::telemetry as queue_telemetry;
use aidaemon_work::reminders;
mod router;
mod runtime_ports;
pub(crate) use aidaemon_events::session;
mod single_instance;
mod skills;
mod startup;
mod state;
mod stt_setup;
use aidaemon_channels::tasks;
#[cfg(feature = "terminal-bridge")]
mod terminal_bridge;
mod terminal_lite;
mod tools;
#[cfg(feature = "computer_use")]
pub use config::ComputerUseConfig;
#[cfg(feature = "computer_use")]
pub use tools::computer_use;
mod traits;
use aidaemon_work::triggers;
mod types;
mod updater;
pub mod utils;
mod wizard;

#[cfg(test)]
mod contract_shape_tests;
#[cfg(test)]
mod integration_tests;
#[cfg(test)]
mod live_e2e_tests;
#[cfg(test)]
mod testing;

use std::path::{Path, PathBuf};
#[cfg(not(feature = "terminal-bridge"))]
use std::process::Stdio;

const SUPPORTED_TERMINAL_AGENTS: &[&str] = &["codex", "claude", "gemini", "opencode"];
const MAX_AGENT_LAUNCH_ARGS: usize = 24;
const MAX_AGENT_LAUNCH_ARG_CHARS: usize = 256;
pub(crate) const RUNTIME_CONFIG_PATH_ENV_KEY: &str = "AIDAEMON_RUNTIME_CONFIG_PATH";
pub(crate) const RUNTIME_ENV_FILE_ENV_KEY: &str = "AIDAEMON_RUNTIME_ENV_FILE";
pub(crate) const RUNTIME_WORKDIR_ENV_KEY: &str = "AIDAEMON_RUNTIME_WORKDIR";

fn runtime_working_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn absolutize_path(path: &Path, base_dir: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn resolve_runtime_env_file_path(working_dir: &Path) -> PathBuf {
    if let Ok(path) = std::env::var("AIDAEMON_ENV_FILE") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return absolutize_path(Path::new(trimmed), working_dir);
        }
    }
    working_dir.join(".env")
}

fn export_runtime_context(config_path: &Path, env_file_path: &Path, working_dir: &Path) {
    std::env::set_var(
        RUNTIME_CONFIG_PATH_ENV_KEY,
        config_path.to_string_lossy().to_string(),
    );
    std::env::set_var(
        RUNTIME_ENV_FILE_ENV_KEY,
        env_file_path.to_string_lossy().to_string(),
    );
    std::env::set_var(
        RUNTIME_WORKDIR_ENV_KEY,
        working_dir.to_string_lossy().to_string(),
    );
}

fn load_startup_env_file(env_file_path: &Path) {
    let explicit_override = std::env::var("AIDAEMON_ENV_FILE")
        .ok()
        .is_some_and(|v| !v.trim().is_empty());

    if env_file_path.exists() {
        if let Err(e) = dotenvy::from_path(env_file_path) {
            eprintln!(
                "Warning: failed to load env file '{}': {}",
                env_file_path.display(),
                e
            );
        }
    } else if explicit_override {
        eprintln!(
            "Warning: AIDAEMON_ENV_FILE points to missing file '{}'",
            env_file_path.display()
        );
    }
}

pub fn run() -> anyhow::Result<()> {
    let working_dir = runtime_working_dir();
    let config_path = working_dir.join("config.toml");
    let env_file_path = resolve_runtime_env_file_path(&working_dir);

    // Load environment file.
    // - Default: .env in the daemon working directory.
    // - Override: AIDAEMON_ENV_FILE=/absolute/path/to/envfile
    export_runtime_context(&config_path, &env_file_path, &working_dir);
    let daemon_paths = tools::DaemonPaths::new(vec![
        config_path.clone(),
        env_file_path.clone(),
        working_dir.join(".env"),
    ]);
    load_startup_env_file(&env_file_path);

    // Interactive runs log to stderr. Background service logs are written to
    // an owner-only, size-rotated file so a long-lived daemon cannot consume
    // the host disk through launch-service output redirection.
    // html5ever warns about markup it cannot place ("foster parenting not
    // implemented") on ordinary fetched pages; nothing here can act on it.
    logging::init(if cfg!(feature = "browser") {
        "info,html5ever=error,chromiumoxide=off"
    } else {
        "info,html5ever=error"
    })?;

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
                    "  memory-export DIR     Write a readable copy of memory to a new directory"
                );
                println!("  setup low-latency     Configure Telegram webhook low-latency mode");
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
                println!("  auth login openai     Connect a ChatGPT subscription (Codex OAuth)");
                println!("  auth status           Show connected model subscriptions");
                println!("  auth logout openai    Disconnect the ChatGPT subscription");
                println!("  dashboard             Open the authenticated local dashboard");
                println!("  dashboard login-url   Print a local dashboard login URL");
                println!("  node <command>        Pair, inspect, authorize, or revoke Nodes");
                println!(
                    "  client-token          Mint the local Client Gateway token (revokes the old)"
                );
                println!("  pair                  Pair a phone or browser with the Client Gateway");
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
            "memory-export" => {
                return handle_memory_export_command(&args[2..], config_path.as_path());
            }
            "setup" => {
                return handle_setup_command(&args[2..], config_path.as_path());
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
            "dashboard" => {
                return handle_dashboard_command(&args[2..], config_path.as_path());
            }
            "auth" => {
                return handle_auth_command(&args[2..]);
            }
            "node" => {
                return nodes::cli::handle_node_command(&args[2..], config_path.as_path());
            }
            "client-token" => {
                return handle_client_token_command(&args[2..], config_path.as_path());
            }
            "pair" => {
                return handle_pair_command(&args[2..], config_path.as_path(), false);
            }
            "web" => {
                return handle_pair_command(&args[2..], config_path.as_path(), true);
            }
            "tui" => {
                return handle_tui_command(&args[2..], config_path.as_path());
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

    // Keep the Mac awake while the daemon runs. The launchd plist launches the
    // binary directly (not via a caffeinate wrapper) to keep a clean TCC code
    // identity for computer_use, so the daemon prevents idle sleep itself.
    #[cfg(target_os = "macos")]
    daemon::spawn_macos_keep_awake();

    // Run async.
    //
    // Use an enlarged worker-thread stack (default is 2 MB). The agent loop
    // (`handle_message_impl`) compiles to a very large async state machine, and
    // a spawn chain (orchestrator → task_lead → executor) can stack several of
    // these per worker thread; the 2 MB default overflowed and crash-looped the
    // daemon. Sub-agents are also now run on their own tasks (see
    // `runtime/spawn.rs`), but the larger stack gives durable headroom.
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(8 * 1024 * 1024)
        .build()?
        .block_on(crate::core::run(config, config_path, daemon_paths))
}

fn dashboard_login_url(config_path: &Path) -> anyhow::Result<String> {
    let config = config::AppConfig::load(config_path)?;
    if !config.daemon.dashboard_enabled {
        anyhow::bail!("The dashboard is disabled in config.toml");
    }
    let token = config::resolve_from_keychain("dashboard_token").map_err(|error| {
        anyhow::anyhow!(
            "Dashboard token is unavailable: {error}. Start or restart the daemon first."
        )
    })?;
    let host = match config.daemon.health_bind.trim() {
        "" | "0.0.0.0" | "::" => "127.0.0.1".to_string(),
        value if value.contains(':') && !value.starts_with('[') => format!("[{value}]"),
        value => value.to_string(),
    };
    Ok(format!(
        "http://{}:{}/#token={}",
        host, config.daemon.health_port, token
    ))
}

fn handle_dashboard_command(args: &[String], config_path: &Path) -> anyhow::Result<()> {
    let action = args.first().map(String::as_str).unwrap_or("open");
    let login_url = dashboard_login_url(config_path)?;
    match action {
        "open" => {
            #[cfg(target_os = "macos")]
            let status = std::process::Command::new("open")
                .arg(&login_url)
                .status()?;
            #[cfg(target_os = "linux")]
            let status = std::process::Command::new("xdg-open")
                .arg(&login_url)
                .status()?;
            #[cfg(target_os = "windows")]
            let status = std::process::Command::new("rundll32")
                .arg("url.dll,FileProtocolHandler")
                .arg(&login_url)
                .status()?;
            #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
            anyhow::bail!("Opening the dashboard is not supported on this platform");

            if !status.success() {
                anyhow::bail!("Could not open the local dashboard in a browser");
            }
            println!("Opened the authenticated local dashboard.");
        }
        "login-url" => {
            println!("{login_url}");
        }
        _ => anyhow::bail!("Usage: aidaemon dashboard [open|login-url]"),
    }
    Ok(())
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

fn handle_memory_export_command(args: &[String], default_config_path: &Path) -> anyhow::Result<()> {
    let mut config_path = default_config_path.to_path_buf();
    let mut directory = None;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("Usage: aidaemon memory-export DIR [--config PATH]");
                println!();
                println!("Write what the daemon's own memory holds as Markdown files. Facts you");
                println!("asked it to forget are left out.");
                println!("DIR must not exist yet. The files contain personal data and are");
                println!("readable only by you.");
                return Ok(());
            }
            "-c" | "--config" => {
                let Some(path) = args.get(i + 1) else {
                    anyhow::bail!("Missing value for --config");
                };
                config_path = PathBuf::from(path);
                i += 2;
            }
            other if directory.is_none() && !other.starts_with('-') => {
                directory = Some(PathBuf::from(other));
                i += 1;
            }
            other => anyhow::bail!("Unknown memory-export option: {other}"),
        }
    }
    let Some(directory) = directory else {
        anyhow::bail!("Usage: aidaemon memory-export DIR [--config PATH]");
    };
    let _ = dotenvy::dotenv();
    let config = config::AppConfig::load(&config_path)?;
    let written = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(crate::core::run_memory_export(config, directory.clone()))?;
    for (name, entries) in written {
        println!("{name}: {entries} entries");
    }
    println!("Wrote {}", directory.display());
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

fn handle_setup_command(args: &[String], default_config_path: &Path) -> anyhow::Result<()> {
    let action = args.first().map(|s| s.as_str()).unwrap_or("");
    match action {
        "low-latency" | "telegram-webhook" => {
            if !default_config_path.exists() {
                anyhow::bail!(
                    "Config file not found at {}. Run `aidaemon` once to generate config.toml first.",
                    default_config_path.display()
                );
            }
            wizard::run_low_latency_setup(default_config_path)
        }
        "-h" | "--help" | "" => {
            println!("Usage:");
            println!("  aidaemon setup low-latency");
            println!("  aidaemon setup telegram-webhook");
            println!();
            println!("Configure Telegram webhook low-latency mode (opt-in).");
            println!("Defaults remain polling unless you run this setup.");
            Ok(())
        }
        other => anyhow::bail!("Unknown setup command: {other}"),
    }
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

/// `aidaemon client-token [--print]`: mint the local token the Client
/// Gateway accepts. Its hash goes to the gateway's token file and the
/// owner's copy to `~/.aidaemon/client-token` (owner-only), which
/// `aidaemon tui`, `aidaemon web` and `aidaemon pair` read; `--print` also
/// writes it to stdout for scripts. Minting again revokes the previous token.
fn handle_client_token_command(args: &[String], config_path: &Path) -> anyhow::Result<()> {
    let mut print = false;
    for arg in args {
        match arg.as_str() {
            "--print" => print = true,
            "-h" | "--help" => {
                println!("Usage: aidaemon client-token [--print]");
                println!();
                println!("Mint the token local apps use with the Client Gateway");
                println!("([client_gateway] in config.toml) and save it for `aidaemon tui`");
                println!("and `aidaemon web`. The old token stops working. --print also");
                println!("writes it to stdout, for scripts (Authorization: Bearer <token>).");
                return Ok(());
            }
            other => anyhow::bail!("Unknown client-token option: {other}"),
        }
    }
    // Read only the gateway's table: loading the whole config would
    // resolve secrets from the keychain.
    let settings = crate::startup::client_gateway::settings_from_file(config_path);
    let path = crate::startup::client_gateway::mint_local_token(&settings)?;
    if print {
        println!("{}", std::fs::read_to_string(&path)?.trim());
    }
    eprintln!(
        "Client Gateway token minted and saved to {} (owner-only). The old token stops working.",
        path.display()
    );
    Ok(())
}

/// The gateway's settings for a client command, or the steps to turn it on.
fn client_gateway_ready(
    config_path: &Path,
) -> anyhow::Result<aidaemon_client_gateway::ClientGatewayConfig> {
    let settings = crate::startup::client_gateway::settings_from_file(config_path);
    anyhow::ensure!(
        settings.enabled,
        "The Client Gateway is off. Ask the agent to \"enable the client gateway\", or add\n\n\
         [client_gateway]\nenabled = true\n\n\
         to {} and restart the daemon (/restart in a chat).",
        config_path.display()
    );
    Ok(settings)
}

/// `aidaemon tui [--device NAME]`: the terminal client on the running
/// daemon's Client Gateway, with the gateway's own address and the local
/// token (minted on first use).
fn handle_tui_command(args: &[String], config_path: &Path) -> anyhow::Result<()> {
    let mut device = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--device" => {
                let Some(value) = args.get(i + 1) else {
                    anyhow::bail!("Missing value for --device");
                };
                device = Some(value.clone());
                i += 2;
            }
            "-h" | "--help" => {
                println!("Usage: aidaemon tui [--device NAME]");
                println!();
                println!("Chat with the daemon, watch every chat's turns and answer approvals");
                println!("in this terminal. Needs [client_gateway] enabled.");
                return Ok(());
            }
            other => anyhow::bail!("Unknown tui option: {other}"),
        }
    }
    let settings = client_gateway_ready(config_path)?;
    let token = crate::startup::client_gateway::local_token_or_mint(&settings)?;
    let url = format!("ws://{}/client/v1/ws", settings.bind_addr()?);
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(aidaemon_tui::run(aidaemon_tui::link_config(
            url,
            aidaemon_tui::link::Auth::LocalToken(token),
            device,
        )))
}

/// Open `url` in the default browser.
fn open_in_browser(url: &str) -> anyhow::Result<()> {
    #[cfg(target_os = "macos")]
    let status = std::process::Command::new("open").arg(url).status()?;
    #[cfg(target_os = "linux")]
    let status = std::process::Command::new("xdg-open").arg(url).status()?;
    #[cfg(target_os = "windows")]
    let status = std::process::Command::new("rundll32")
        .arg("url.dll,FileProtocolHandler")
        .arg(url)
        .status()?;
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    anyhow::bail!("Opening a browser is not supported on this platform");
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    anyhow::ensure!(status.success(), "Could not open a browser");
    Ok(())
}

/// `aidaemon pair [--token-file FILE]` / `aidaemon web`: make a pairing
/// offer on the running daemon's Client Gateway, show its code and a QR
/// code of the web client's link (`web` also opens it in this machine's
/// browser), and confirm the device here when it redeems the offer. Uses
/// the owner's local token (`AIDAEMON_CLIENT_TOKEN`, `--token-file`, or the
/// saved copy, minted on first use).
fn handle_pair_command(args: &[String], config_path: &Path, open_web: bool) -> anyhow::Result<()> {
    use aidaemon_client_protocol::{Enrollment, EnrollmentState, PairingDecision, PairingOffer};
    let command = if open_web { "web" } else { "pair" };
    let mut token = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--token-file" => {
                let Some(path) = args.get(i + 1) else {
                    anyhow::bail!("Missing value for --token-file");
                };
                token = Some(std::fs::read_to_string(path)?.trim().to_string());
                i += 2;
            }
            "-h" | "--help" => {
                if open_web {
                    println!("Usage: aidaemon web [--token-file FILE]");
                    println!();
                    println!("Open the web client in this machine's browser and pair it with the");
                    println!("Client Gateway. Needs [client_gateway] enabled.");
                } else {
                    println!("Usage: aidaemon pair [--token-file FILE]");
                    println!();
                    println!(
                        "Pair a phone or browser with the Client Gateway (scan the QR code or"
                    );
                    println!("enter the code). Needs [client_gateway] enabled.");
                }
                return Ok(());
            }
            other => anyhow::bail!("Unknown {command} option: {other}"),
        }
    }
    let settings = client_gateway_ready(config_path)?;
    let token = match token.filter(|t| !t.trim().is_empty()) {
        Some(token) => token,
        None => crate::startup::client_gateway::local_token_or_mint(&settings)?,
    };
    let address = settings.bind_addr()?;
    let base = format!("http://{address}");
    let http = reqwest::blocking::Client::new();
    let response = http
        .post(format!("{base}/client/v1/pairing/offers"))
        .bearer_auth(&token)
        .json(&serde_json::json!({}))
        .send()
        .map_err(|error| anyhow::anyhow!("Is the daemon running with the gateway on? {error}"))?;
    anyhow::ensure!(
        response.status().is_success(),
        "The gateway refused the offer ({}): check the token",
        response.status()
    );
    let offer: PairingOffer = response.json()?;
    // The daemon knows the live address (a tunnel's origin); an older
    // daemon does not say, so fall back to the config file's.
    let public_url = offer
        .public_url
        .clone()
        .filter(|url| !url.trim().is_empty())
        .unwrap_or_else(|| settings.public_url());
    let link = format!(
        "{}/client/v1/app#offer={}&code={}&fp={}",
        public_url.trim_end_matches('/'),
        offer.offer_id,
        offer.code,
        offer.fingerprint
    );
    if let Some(code) = (!open_web)
        .then(|| qrcode::QrCode::new(link.as_bytes()).ok())
        .flatten()
    {
        println!(
            "{}",
            code.render::<qrcode::render::unicode::Dense1x2>()
                .quiet_zone(true)
                .build()
        );
    }
    if open_web {
        let local = format!(
            "http://{address}/client/v1/app#offer={}&code={}&fp={}",
            offer.offer_id, offer.code, offer.fingerprint
        );
        match open_in_browser(&local) {
            Ok(()) => println!("Opened the web client in your browser."),
            Err(error) => println!("Could not open a browser ({error}); open {local}"),
        }
        println!("Elsewhere: open {link}");
    } else {
        println!("Open {link}");
    }
    println!(
        "or enter the code {} on the device (valid 10 minutes).",
        offer.code
    );
    println!("Daemon fingerprint: {}", offer.fingerprint);
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
        let enrollment: Enrollment = http
            .get(format!(
                "{base}/client/v1/pairing/offers/{}",
                offer.offer_id
            ))
            .bearer_auth(&token)
            .send()?
            .json()?;
        match enrollment.state {
            EnrollmentState::Open => continue,
            EnrollmentState::Pending => {
                let scopes: Vec<String> = enrollment
                    .scopes
                    .iter()
                    .map(|s| format!("{s:?}").to_lowercase())
                    .collect();
                let approve = dialoguer::Confirm::new()
                    .with_prompt(format!(
                        "Pair “{}” ({:?}) with access to: {}?",
                        enrollment.device_name.unwrap_or_default(),
                        enrollment
                            .device_kind
                            .unwrap_or(aidaemon_client_protocol::ClientKind::Unknown),
                        scopes.join(", ")
                    ))
                    .default(false)
                    .interact()?;
                let settled: Enrollment = http
                    .post(format!(
                        "{base}/client/v1/pairing/offers/{}/decision",
                        offer.offer_id
                    ))
                    .bearer_auth(&token)
                    .json(&PairingDecision { approve })
                    .send()?
                    .json()?;
                match settled.device_id {
                    Some(device_id) => println!("Paired: {device_id}"),
                    None => println!("Not paired."),
                }
                return Ok(());
            }
            EnrollmentState::Expired => {
                anyhow::bail!("The offer expired; run `aidaemon {command}` again")
            }
            other => {
                println!("Pairing ended: {other:?}");
                return Ok(());
            }
        }
    }
}

/// `aidaemon auth …` — manage subscription-based model logins.
fn handle_auth_command(args: &[String]) -> anyhow::Result<()> {
    let action = args.first().map(|s| s.as_str()).unwrap_or("");
    match action {
        "login" => match auth_provider_alias(args.get(1)) {
            Some(AuthProvider::OpenAiChatGpt) => {
                let paste = args.iter().any(|a| a == "--paste");
                run_chatgpt_login(paste)
            }
            None => anyhow::bail!("Usage: aidaemon auth login openai [--paste]"),
        },
        "logout" => match auth_provider_alias(args.get(1)) {
            Some(AuthProvider::OpenAiChatGpt) => run_chatgpt_logout(),
            None => anyhow::bail!("Usage: aidaemon auth logout openai"),
        },
        "status" => {
            match crate::oauth::chatgpt_codex::load_credentials() {
                Some(creds) => {
                    let state = if creds.needs_refresh(chrono::Utc::now()) {
                        "refresh due"
                    } else {
                        "valid"
                    };
                    println!("openai (ChatGPT subscription): connected");
                    println!("  access token: {state} (expires {})", creds.expires_at);
                }
                None => {
                    println!("openai (ChatGPT subscription): not connected");
                    println!("  connect with: aidaemon auth login openai");
                }
            }
            Ok(())
        }
        "-h" | "--help" | "" => {
            println!("Usage:");
            println!("  aidaemon auth login openai [--paste]");
            println!("  aidaemon auth logout openai");
            println!("  aidaemon auth status");
            println!();
            println!(
                "Connects a ChatGPT Plus/Pro/Business subscription so aidaemon can use it as a"
            );
            println!("model provider. Use --paste when no browser can reach this machine's");
            println!("localhost (containers, remote servers, headless installs).");
            Ok(())
        }
        other => anyhow::bail!("Unknown auth command: {other}"),
    }
}

enum AuthProvider {
    OpenAiChatGpt,
}

fn auth_provider_alias(raw: Option<&String>) -> Option<AuthProvider> {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("openai") | Some("chatgpt") | Some("openai_chatgpt") | Some("codex") => {
            Some(AuthProvider::OpenAiChatGpt)
        }
        _ => None,
    }
}

/// Build a small runtime for one-shot CLI auth work.
fn auth_runtime() -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?)
}

fn run_chatgpt_login(paste: bool) -> anyhow::Result<()> {
    use crate::oauth::chatgpt_codex::{self, CallbackMode};

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;

    let mode = if paste {
        CallbackMode::Paste(Box::new(|_url| {
            println!();
            println!(
                "After approving, your browser will land on a localhost URL that will not load."
            );
            println!("Copy that URL from the address bar and paste it here, then press Enter:");
            let mut line = String::new();
            std::io::stdin().read_line(&mut line)?;
            Ok(line.trim().to_string())
        }))
    } else {
        CallbackMode::Loopback
    };

    let creds = auth_runtime()?.block_on(chatgpt_codex::login(&client, mode, |url| {
        println!("Open this URL to authorize aidaemon with your ChatGPT account:");
        println!();
        println!("  {url}");
        println!();
        if !paste {
            println!("Waiting for the redirect… (Ctrl-C to cancel)");
        }
    }))?;

    println!();
    println!("Connected. Access token valid until {}.", creds.expires_at);
    println!("Point your config at it with:");
    println!();
    println!("  [provider]");
    println!("  kind = \"openai_chatgpt\"");
    println!("  models = {{ default_model = \"gpt-5.6-terra\" }}");
    Ok(())
}

fn run_chatgpt_logout() -> anyhow::Result<()> {
    if !crate::oauth::chatgpt_codex::is_connected() {
        println!("No ChatGPT subscription is connected.");
        return Ok(());
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    auth_runtime()?.block_on(crate::oauth::chatgpt_codex::logout(&client))?;
    println!("Disconnected the ChatGPT subscription.");
    Ok(())
}

pub(crate) fn normalize_terminal_agent_name(raw: &str) -> Option<&'static str> {
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "codex" => Some("codex"),
        "claude" => Some("claude"),
        "gemini" => Some("gemini"),
        "opencode" => Some("opencode"),
        _ => None,
    }
}

pub(crate) use aidaemon_tools::permission_bypass::normalize_terminal_agent_permission_aliases;

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
    let (launch_args, _) = normalize_terminal_agent_permission_aliases(Some(agent), launch_args);
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
