//! A ticketed socket on the desktops (feature `native`): a live view's
//! frames and a shell's output each come on a WebSocket of their own,
//! opened with a ticket. [`run`] drives one: TLS, the ticket, then events
//! in and messages out until either side ends.

use futures::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

/// What one message from the server means to its driver.
pub(crate) enum Incoming<E> {
    /// Nothing for the UI (a message type this client does not read).
    Skip,
    /// An event for the UI.
    Event(E),
    /// An event for the UI, then this message back to the server once it
    /// was handed over (a frame's ack).
    EventThenReply(E, String),
    /// The last event: the server ended the socket, and why.
    Closed(E, String),
}

/// How a socket's endings read to the owner.
pub(crate) struct Endings {
    /// What could not be opened ("the live view").
    pub opening: &'static str,
    /// The connection went away.
    pub dropped: &'static str,
    /// The UI side went away.
    pub closed_here: &'static str,
}

/// Drive the socket at `url` until it ends: send `ticket_message`, hand
/// each event to `events` (closing the socket once nothing receives them),
/// and send what `outgoing` gives it. Returns why it ended.
pub(crate) async fn run<E>(
    url: String,
    ticket_message: String,
    endings: Endings,
    events: tokio::sync::mpsc::UnboundedSender<E>,
    mut outgoing: tokio::sync::mpsc::UnboundedReceiver<String>,
    mut read: impl FnMut(Message) -> Incoming<E>,
) -> String {
    let tls = match crate::tls::client_config() {
        Ok(tls) => tls,
        Err(error) => return format!("could not set up TLS: {error}"),
    };
    let socket = tokio_tungstenite::connect_async_tls_with_config(
        url.as_str(),
        None,
        false,
        Some(tokio_tungstenite::Connector::Rustls(tls)),
    )
    .await;
    let mut socket = match socket {
        Ok((socket, _)) => socket,
        Err(error) => return format!("could not open {}: {error}", endings.opening),
    };
    if socket.send(Message::text(ticket_message)).await.is_err() {
        return endings.dropped.to_string();
    }
    loop {
        tokio::select! {
            incoming = socket.next() => {
                let message = match incoming {
                    Some(Ok(message)) => message,
                    _ => return endings.dropped.to_string(),
                };
                let (event, reply) = match read(message) {
                    Incoming::Skip => continue,
                    Incoming::Closed(event, why) => {
                        let _ = events.send(event);
                        return why;
                    }
                    Incoming::Event(event) => (event, None),
                    Incoming::EventThenReply(event, reply) => (event, Some(reply)),
                };
                if events.send(event).is_err() {
                    let _ = socket.close(None).await;
                    return endings.closed_here.to_string();
                }
                if let Some(reply) = reply
                    && socket.send(Message::text(reply)).await.is_err()
                {
                    return endings.dropped.to_string();
                }
            }
            sent = outgoing.recv() => {
                let Some(text) = sent else {
                    let _ = socket.close(None).await;
                    return endings.closed_here.to_string();
                };
                if socket.send(Message::text(text)).await.is_err() {
                    return endings.dropped.to_string();
                }
            }
        }
    }
}
