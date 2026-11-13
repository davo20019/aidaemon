//! `aidaemon-tui [--url ws://127.0.0.1:8788/client/v1/ws] [--device NAME]`
//!
//! The token comes from `AIDAEMON_CLIENT_TOKEN`, or `--token-file FILE`
//! (the output of `aidaemon client-token` saved to a file). `aidaemon tui`
//! runs the same client with the daemon's own settings and no flags.

use aidaemon_tui::link::Auth;

fn usage() -> ! {
    eprintln!(
        "Usage: aidaemon-tui [--url WS_URL] [--token-file FILE] [--device NAME]\n\
         The token is read from AIDAEMON_CLIENT_TOKEN or --token-file \
         (mint one with `aidaemon client-token`). `aidaemon tui` needs no flags."
    );
    std::process::exit(2)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut url = "ws://127.0.0.1:8788/client/v1/ws".to_string();
    let mut token = std::env::var("AIDAEMON_CLIENT_TOKEN").ok();
    let mut device = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--url" => url = args.next().unwrap_or_else(|| usage()),
            "--token-file" => {
                let path = args.next().unwrap_or_else(|| usage());
                token = Some(std::fs::read_to_string(path)?.trim().to_string());
            }
            "--device" => device = args.next(),
            _ => usage(),
        }
    }
    let Some(token) = token else { usage() };
    aidaemon_tui::run(aidaemon_tui::link_config(
        url,
        Auth::LocalToken(token),
        device,
    ))
    .await
}
