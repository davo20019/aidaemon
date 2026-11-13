//! # aidaemon-tui
//!
//! A terminal client for the AIdaemon Client Gateway, built on the client
//! protocol alone (`aidaemon-client-protocol`): the sessions list, chat
//! with streaming and live status, approval cards from the typed card
//! facts, the running-work panel with cancel, and a connection that
//! reconnects and resumes from its last event.

pub mod app;
pub mod link;
pub mod state;

use std::time::Duration;

use crossterm::event::{Event as TermEvent, EventStream, KeyEventKind};
use futures::StreamExt;
use tokio::sync::mpsc;

use crate::app::{Action, App};
use crate::link::{LinkConfig, Update};

/// Run the TUI on this terminal until the user quits: connect with
/// `config`, draw, and send what the keys ask for.
pub async fn run(config: LinkConfig) -> anyhow::Result<()> {
    let (updates_tx, mut updates) = mpsc::unbounded_channel();
    let commands = link::spawn(config, updates_tx);
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
