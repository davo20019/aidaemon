//! # aidaemon-tui
//!
//! A terminal client for the AIdaemon Client Gateway on the shared client
//! core (`aidaemon-client-core`): the sessions list, chat with streaming and
//! live status, approval cards from the typed card facts, the running-work
//! panel with cancel, and a connection that reconnects and resumes from its
//! last event.

pub mod app;

pub use aidaemon_client_core::{link, state};

use std::time::Duration;

use aidaemon_client_core::link::{Auth, ClientIdentity, LinkConfig};
use aidaemon_client_protocol::ClientKind;
use crossterm::event::{Event as TermEvent, EventStream, KeyEventKind};
use futures::StreamExt;
use tokio::sync::mpsc;

use crate::app::{Action, App};

/// The link settings of this terminal client signing in with `auth`.
pub fn link_config(url: String, auth: Auth, device: Option<String>) -> LinkConfig {
    LinkConfig {
        url,
        auth,
        client: ClientIdentity {
            kind: ClientKind::Tui,
            name: "aidaemon-tui".into(),
            version: Some(env!("CARGO_PKG_VERSION").into()),
        },
        device,
        capabilities: app::CAPABILITIES.to_vec(),
    }
}

/// Run the TUI on this terminal until the user quits: connect with
/// `config`, draw, and send what the keys ask for.
pub async fn run(config: LinkConfig) -> anyhow::Result<()> {
    let (updates_tx, mut updates) = mpsc::unbounded_channel();
    let commands = link::spawn(config, updates_tx);
    let mut terminal = ratatui::init();
    let mut app = App::default();
    let mut keys = EventStream::new();
    let mut redraw = tokio::time::interval(Duration::from_millis(250));
    let result = loop {
        if let Err(error) = terminal.draw(|frame| app.draw(frame)) {
            break Err(error.into());
        }
        tokio::select! {
            update = updates.recv() => match update {
                Some(update) => {
                    for command in app.update(update) {
                        let _ = commands.send((link::request_id(), command));
                    }
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
            _ = redraw.tick() => app.tick(),
        }
    };
    ratatui::restore();
    result
}
