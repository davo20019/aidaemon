mod agent;
mod channels;
mod config;
mod core;
mod daemon;
mod dashboard;
#[allow(dead_code)]
mod events;
#[allow(dead_code)]
mod health;
mod mcp;
mod memory;
#[allow(dead_code)]
mod plans;
mod providers;
mod router;
mod scheduler;
mod skills;
mod state;
mod tools;
mod traits;
mod tasks;
mod types;
mod triggers;
mod updater;
pub mod utils;
mod wizard;

#[cfg(test)]
mod testing;
#[cfg(test)]
mod integration_tests;

use std::path::PathBuf;

use tracing_subscriber::EnvFilter;

fn main() -> anyhow::Result<()> {
    // Load .env if present
    let _ = dotenvy::dotenv();

    // Tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                if cfg!(feature = "browser") {
                    EnvFilter::new("info,chromiumoxide=off")
                } else {
                    EnvFilter::new("info")
                }
            }),
        )
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
                println!("  install-service  Install as a system service (launchd/systemd)");
                println!("  check-update     Check for available updates");
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
                        println!(
                            "aidaemon is up to date (v{}).",
                            env!("CARGO_PKG_VERSION")
                        );
                    }
                    Err(e) => {
                        eprintln!("Update check failed: {}", e);
                        std::process::exit(1);
                    }
                }
                return Ok(());
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
