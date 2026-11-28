//! `aidaemon-bridge`: run the relay. Flags or `AIDAEMON_BRIDGE_*`
//! variables set the limits (`--help` lists them). It speaks plain HTTP;
//! put a TLS-terminating proxy that serves only https/wss in front.

use aidaemon_bridge::Bridge;
use aidaemon_bridge::config::{Config, OPTIONS};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("aidaemon-bridge [--option value]...  (or AIDAEMON_BRIDGE_<OPTION>=value)");
        for (name, what) in OPTIONS {
            println!("  --{name:<28} {what}");
        }
        return;
    }
    let config = match Config::parse(args, |var| std::env::var(var).ok()) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("aidaemon-bridge: {error}");
            std::process::exit(2);
        }
    };

    // LOGGING RULE: this relay is public and sees other people's traffic.
    // Log events and refusals only ("daemon logged in", "connection
    // refused: limit"). Never log frame contents, frame sizes, nameplates,
    // route ids, connection ids or client addresses, at any level.
    tracing_subscriber::fmt()
        .with_target(false)
        .with_max_level(tracing::Level::INFO)
        .init();

    let listener = match tokio::net::TcpListener::bind(config.listen).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("aidaemon-bridge: cannot listen: {error}");
            std::process::exit(1);
        }
    };
    tracing::info!("bridge listening");
    let bridge = Bridge::new(config);
    tokio::select! {
        result = bridge.serve(listener) => {
            if let Err(error) = result {
                eprintln!("aidaemon-bridge: {error}");
                std::process::exit(1);
            }
        }
        _ = tokio::signal::ctrl_c() => tracing::info!("bridge stopping"),
    }
}
