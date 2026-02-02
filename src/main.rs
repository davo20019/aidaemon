mod agent;
mod channels;
mod config;
mod core;
mod daemon;
mod mcp;
mod memory;
mod providers;
mod router;
mod skills;
mod state;
mod tools;
mod traits;
mod tasks;
mod types;
mod triggers;
mod wizard;

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

    // Handle `install-service` subcommand
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "install-service" {
        return daemon::install_service();
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
