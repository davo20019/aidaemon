//! Conversation shells on the client side (protocol 1.42): the messages of
//! a shell's socket.
//!
//! Opening and closing a shell go through [`crate::Client`] like any
//! command (`open_shell`, `close_shell`). A shell's output comes on a
//! socket of its own, one per attached terminal; [`read`] turns one of its
//! messages into a [`ShellEvent`] and [`auth`], [`input`] and [`resize`]
//! are what the client sends there. Output never enters [`crate::State`]:
//! a UI writes it straight into its terminal.
//!
//! On the desktops [`run`] (feature `native`) drives the socket; the
//! browser drives the page's own WebSocket with the same messages.

use aidaemon_client_protocol::shell::{ShellClientMessage, ShellServerMessage};

/// Something that came on a shell's socket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellEvent {
    /// Authenticated: output follows.
    Ready,
    /// What the shell wrote, to write into the terminal as it is.
    Output(String),
    /// Clear the terminal: the output so far comes next.
    Reset,
    /// The shell's program ended.
    Ended(Option<u32>),
    /// The socket closed, and why.
    Closed(String),
}

fn text(message: &ShellClientMessage) -> String {
    serde_json::to_string(message).unwrap_or_default()
}

/// The first message: the ticket from `shell_opened`.
pub fn auth(ticket: &str) -> String {
    text(&ShellClientMessage::Auth {
        ticket: ticket.to_owned(),
    })
}

/// What the owner typed or pasted (and the terminal's answers).
pub fn input(data: &str) -> String {
    text(&ShellClientMessage::Input {
        data: data.to_owned(),
    })
}

/// The terminal's size in cells.
pub fn resize(cols: u16, rows: u16) -> String {
    text(&ShellClientMessage::Resize { cols, rows })
}

/// One message from the server; `None` for one this client does not know.
pub fn read(message: &str) -> Option<ShellEvent> {
    Some(
        match serde_json::from_str::<ShellServerMessage>(message).ok()? {
            ShellServerMessage::Ready { .. } => ShellEvent::Ready,
            ShellServerMessage::Output { data } => ShellEvent::Output(data),
            ShellServerMessage::Reset => ShellEvent::Reset,
            ShellServerMessage::Ended { exit_code } => ShellEvent::Ended(exit_code),
            ShellServerMessage::Closed { reason } => ShellEvent::Closed(reason),
            ShellServerMessage::Unknown => return None,
        },
    )
}

/// A terminal size the daemon takes: cells, at least 2 by 1. A terminal
/// laid out before it has room reads as 0 by 0.
pub fn cells(cols: u32, rows: u32) -> (u16, u16) {
    (
        u16::try_from(cols).unwrap_or(u16::MAX).max(2),
        u16::try_from(rows).unwrap_or(u16::MAX).max(1),
    )
}

/// Drive a shell's socket until it ends: sends the ticket, hands each
/// event to `events` and sends what `outgoing` gives it (from [`input`]
/// and [`resize`]). Returns why it ended.
#[cfg(feature = "native")]
pub async fn run(
    url: String,
    ticket: String,
    events: tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    outgoing: tokio::sync::mpsc::UnboundedReceiver<String>,
) -> String {
    use crate::ticketed::{Endings, Incoming};
    use tokio_tungstenite::tungstenite::Message;

    let endings = Endings {
        opening: "the shell",
        dropped: "the shell's connection dropped",
        closed_here: "the terminal was closed",
    };
    crate::ticketed::run(url, auth(&ticket), endings, events, outgoing, |message| {
        let Message::Text(text) = message else {
            return Incoming::Skip;
        };
        match read(&text) {
            None => Incoming::Skip,
            Some(ShellEvent::Closed(why)) => Incoming::Closed(ShellEvent::Closed(why.clone()), why),
            Some(event) => Incoming::Event(event),
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_round_trip_the_wire() {
        assert_eq!(
            read(r#"{"type":"output","data":"$ "}"#),
            Some(ShellEvent::Output("$ ".into()))
        );
        assert_eq!(read(r#"{"type":"ended"}"#), Some(ShellEvent::Ended(None)));
        assert_eq!(read(r#"{"type":"later_kind"}"#), None);
        assert!(input("ls\r").contains(r#""data":"ls\r""#));
        assert_eq!(resize(80, 24), r#"{"type":"resize","cols":80,"rows":24}"#);
        assert_eq!(cells(0, 0), (2, 1));
    }
}
