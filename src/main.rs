mod agent;
mod channels;
mod config;
mod core;
mod cron_utils;
mod daemon;
mod dashboard;
#[allow(dead_code)]
mod events;
mod execution_policy;
mod goal_tokens;
#[allow(dead_code)]
mod health;
mod heartbeat;
mod mcp;
mod memory;
mod oauth;
#[allow(dead_code)]
mod plans;
mod providers;
mod router;
mod skills;
mod state;
mod tasks;
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

use std::path::PathBuf;

use tracing_subscriber::EnvFilter;

fn main() -> anyhow::Result<()> {
    // Load .env if present
    let _ = dotenvy::dotenv();

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
            "keychain" => {
                return handle_keychain_command(&args[2..]);
            }
            _ => {}
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
