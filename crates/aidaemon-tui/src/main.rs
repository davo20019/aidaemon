//! `aidaemon-tui [--url ws://127.0.0.1:8788/client/v1/ws] [--device NAME]`
//!
//! The token comes from `AIDAEMON_CLIENT_TOKEN`, or `--token-file FILE`
//! (the output of `aidaemon client-token` saved to a file).

use std::time::Duration;

use aidaemon_tui::app::{Action, App};
use aidaemon_tui::link::{self, LinkConfig, Update};
use crossterm::event::{Event as TermEvent, EventStream, KeyEventKind};
use futures::StreamExt;
use tokio::sync::mpsc;

fn usage() -> ! {
    eprintln!(
        "Usage: aidaemon-tui [--url WS_URL] [--token-file FILE] [--device NAME]\n\
         The token is read from AIDAEMON_CLIENT_TOKEN or --token-file \
         (mint one with `aidaemon client-token`)."
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

    let (updates_tx, mut updates) = mpsc::unbounded_channel();
    let commands = link::spawn(
        LinkConfig {
            url,
            token,
            device,
            capabilities: Vec::new(),
        },
        updates_tx,
    );
    let mut terminal = ratatui::init();
    let mut app = App::default();
    let mut keys = EventStream::new();
    let mut redraw = tokio::time::interval(Duration::from_millis(500));
    let result = loop {
        if let Err(error) = terminal.draw(|frame| app.draw(frame)) {
            break Err(error.into());
        }
        tokio::select! {
            update = updates.recv() => match update {
                Some(Update::Connected) => {
                    app.connected = true;
                    let _ = commands.send((
                        link::request_id(),
                        aidaemon_client_protocol::Command::ListSessions { cursor: None, limit: 200 },
                    ));
                }
                Some(Update::Disconnected(reason)) => {
                    app.connected = false;
                    app.state.notice = Some(reason);
                }
                Some(Update::Frame(frame)) => {
                    app.state.apply(frame.clone());
                    app.after_frame(&frame);
                }
                None => break Ok(()),
            },
            key = keys.next() => match key {
                Some(Ok(TermEvent::Key(key))) if key.kind == KeyEventKind::Press => {
                    match app.key(key) {
                        Some(Action::Quit) => break Ok(()),
                        Some(Action::Send(command)) => {
                            let _ = commands.send((link::request_id(), command));
                        }
                        None => {}
                    }
                }
                Some(Err(error)) => break Err(error.into()),
                None => break Ok(()),
                _ => {}
            },
            _ = redraw.tick() => {}
        }
    };
    ratatui::restore();
    result
}
