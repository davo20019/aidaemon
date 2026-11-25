//! Live surfaces on the client side (protocol 1.30): the live view the app
//! has open, and the frame socket's messages.
//!
//! The control plane (open, close, take and release control) goes through
//! [`crate::Client`] like any command. Frames come on a socket of their own,
//! one per live view; [`read`] turns one binary message into a
//! [`MediaEvent`] and [`auth`], [`ack`] and [`input`] are what the client
//! sends there. Frames never enter [`crate::State`]: a UI keeps the newest
//! one beside its screen, so ten pictures a second do not rebuild pages.
//!
//! On the desktops [`run`] (feature `native`) drives the socket; the browser
//! drives the page's own WebSocket with the same messages.

use aidaemon_client_protocol::media::{
    self, FocusTyping, MediaClientMessage, MediaHeader, SurfaceInput,
};
use aidaemon_client_protocol::{SurfaceController, SurfaceTransport};
use base64::Engine as _;

/// The live view the app has open: which surface, whether it asked for
/// control, and where its frames are once the daemon said.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveView {
    pub surface_id: String,
    /// Opened to control, not only watch.
    pub control: bool,
    /// Set once the daemon opened it.
    pub live_id: Option<String>,
    pub transport: Option<SurfaceTransport>,
    /// Why it could not open, or why it ended.
    pub error: Option<String>,
}

/// Something that came on the frame socket.
#[derive(Debug, Clone, PartialEq)]
pub enum MediaEvent {
    /// Authenticated: frames follow.
    Ready,
    /// A picture: `uri` is a `data:` URI of its bytes. Ack `seq` once shown.
    Frame {
        seq: u64,
        w: u32,
        h: u32,
        uri: String,
    },
    /// Who controls the surface now.
    Controller(SurfaceController),
    /// What has the focus now: in words ("Username, a text field"; `None`:
    /// nothing), its place on the frame (`[x, y, w, h]` fractions), and
    /// what typing does there (`None`: it takes none, as a button).
    Focus {
        words: Option<String>,
        rect: Option<[f64; 4]>,
        typing: Option<FocusTyping>,
    },
    /// The view ended, and why.
    Closed(String),
}

/// The socket URL of a live view: `transport.url` (a path) on the gateway
/// at `http_base` (`https://host` becomes `wss://host`).
pub fn socket_url(http_base: &str, transport: &SurfaceTransport) -> String {
    path_socket_url(http_base, &transport.url)
}

/// The WebSocket URL of `path` on the gateway at `http_base` (`https://host`
/// becomes `wss://host`, `http://` becomes `ws://`): a live view's or a
/// shell's socket.
pub fn path_socket_url(http_base: &str, path: &str) -> String {
    let base = http_base.trim_end_matches('/');
    let ws = match base.split_once("://") {
        Some(("https", rest)) => format!("wss://{rest}"),
        Some(("http", rest)) => format!("ws://{rest}"),
        _ => format!("ws://{base}"),
    };
    format!("{ws}{path}")
}

fn text(message: &MediaClientMessage) -> String {
    serde_json::to_string(message).unwrap_or_default()
}

/// The first message: the ticket.
pub fn auth(ticket: &str) -> String {
    text(&MediaClientMessage::Auth {
        ticket: ticket.to_string(),
    })
}

/// Frame `seq` is shown: the next may come.
pub fn ack(seq: u64) -> String {
    text(&MediaClientMessage::Ack { seq })
}

/// Input for the surface.
pub fn input(event: SurfaceInput) -> String {
    text(&MediaClientMessage::Input { event })
}

/// The room this device has for the picture changed (protocol 1.33).
pub fn fit(w: u32, h: u32) -> String {
    text(&MediaClientMessage::Fit { w, h })
}

/// Whether the room for the picture changed enough to lay the page out
/// again: its width moved by 20 points or more, as when the device turns
/// between portrait and landscape. A height alone does not count: a
/// phone's keyboard opening takes height (it can even leave the room wider
/// than tall), and the page must not change under the field being typed in.
pub fn room_turned(sent: (u32, u32), now: (u32, u32)) -> bool {
    sent.0.abs_diff(now.0) >= 20
}

/// A press, a move or a release at a fraction of the frame.
pub fn pointer(action: &str, x: f32, y: f32) -> SurfaceInput {
    SurfaceInput::Pointer {
        action: action.to_string(),
        x: f64::from(x.clamp(0.0, 1.0)),
        y: f64::from(y.clamp(0.0, 1.0)),
        button: None,
    }
}

/// A press, move or release of `button` (`left`, `right`, `middle`).
pub fn button_pointer(action: &str, x: f32, y: f32, button: &str) -> SurfaceInput {
    SurfaceInput::Pointer {
        action: action.to_string(),
        x: f64::from(x.clamp(0.0, 1.0)),
        y: f64::from(y.clamp(0.0, 1.0)),
        button: Some(button.to_string()),
    }
}

/// One key going down or up, by `KeyboardEvent.code` and `.key`, with
/// modifiers (`shift`, `ctrl`, `alt`, `meta`).
pub fn key_event(down: bool, code: &str, key: &str, mods: &[&str]) -> SurfaceInput {
    SurfaceInput::Key {
        action: if down { "down" } else { "up" }.to_string(),
        code: code.to_string(),
        key: key.to_string(),
        mods: mods.iter().map(|m| (*m).to_string()).collect(),
    }
}

/// A key pressed (down then up) by its `KeyboardEvent.code` and `.key`.
pub fn key_press(code: &str, key: &str) -> [SurfaceInput; 2] {
    let event = |action: &str| SurfaceInput::Key {
        action: action.to_string(),
        code: code.to_string(),
        key: key.to_string(),
        mods: Vec::new(),
    };
    [event("down"), event("up")]
}

/// A binary message from the frame socket.
pub fn read(bytes: &[u8]) -> Option<MediaEvent> {
    let (header, payload) = media::decode(bytes).ok()?;
    Some(match header {
        MediaHeader::Ready { .. } => MediaEvent::Ready,
        MediaHeader::Frame {
            seq, codec, w, h, ..
        } => MediaEvent::Frame {
            seq,
            w,
            h,
            uri: data_uri(&codec, payload)?,
        },
        MediaHeader::Controller { controller } => MediaEvent::Controller(controller),
        MediaHeader::Closed { reason } => MediaEvent::Closed(reason),
        MediaHeader::Focus {
            what,
            label,
            rect,
            typing,
        } => focus_event(what, label, rect, typing),
        MediaHeader::Unknown => return None,
    })
}

/// A focus report in words: "Username, a text field", "a button".
/// A focus header as an event.
pub fn focus_event(
    what: Option<String>,
    label: Option<String>,
    rect: Option<[f64; 4]>,
    typing: Option<FocusTyping>,
) -> MediaEvent {
    let words = focus_words(what, label);
    MediaEvent::Focus {
        rect: rect.filter(|_| words.is_some()),
        typing: typing.filter(|_| words.is_some()),
        words,
    }
}

pub fn focus_words(what: Option<String>, label: Option<String>) -> Option<String> {
    match (label, what) {
        (Some(label), Some(what)) => Some(format!("{label}, {what}")),
        (None, Some(what)) => Some(what),
        (Some(label), None) => Some(label),
        (None, None) => None,
    }
}

/// A frame's bytes as a `data:` URI an image shows.
pub fn data_uri(codec: &str, bytes: &[u8]) -> Option<String> {
    let mime = match codec {
        "jpeg" => "image/jpeg",
        "png" => "image/png",
        _ => return None,
    };
    Some(format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    ))
}

/// Drive a live view's frame socket until it ends: sends the ticket, hands
/// each event to `events`, acks every frame once handed over, and sends
/// what `outgoing` gives it (from [`input`]). Returns why it ended.
#[cfg(feature = "native")]
pub async fn run(
    url: String,
    ticket: String,
    events: tokio::sync::mpsc::UnboundedSender<MediaEvent>,
    outgoing: tokio::sync::mpsc::UnboundedReceiver<String>,
) -> String {
    use crate::ticketed::{Endings, Incoming};
    use tokio_tungstenite::tungstenite::Message;

    let endings = Endings {
        opening: "the live view",
        dropped: "the live view closed",
        closed_here: "the view was closed",
    };
    crate::ticketed::run(url, auth(&ticket), endings, events, outgoing, |message| {
        let Message::Binary(bytes) = message else {
            return Incoming::Skip;
        };
        match read(&bytes) {
            None => Incoming::Skip,
            Some(MediaEvent::Closed(why)) => Incoming::Closed(MediaEvent::Closed(why.clone()), why),
            // A frame is acked once handed over.
            Some(event @ MediaEvent::Frame { seq, .. }) => {
                Incoming::EventThenReply(event, ack(seq))
            }
            Some(event) => Incoming::Event(event),
        }
    })
    .await
}

#[cfg(test)]
mod tests {

    #[test]
    fn a_turn_or_a_new_width_lays_the_page_out_again_and_a_keyboard_does_not() {
        let portrait = (391, 549);
        assert!(room_turned(portrait, (829, 230)), "turned to landscape");
        assert!(room_turned((829, 230), portrait), "and back");
        assert!(room_turned(portrait, (420, 600)), "a wider window");
        assert!(!room_turned(portrait, (391, 260)), "the keyboard opened");
        assert!(!room_turned(portrait, (395, 549)), "a few points");
        assert!(fit(800, 300).contains("\"type\":\"fit\""));
    }

    use super::*;
    use aidaemon_client_protocol::TransportKind;

    #[test]
    fn the_socket_url_follows_the_gateway_scheme() {
        let transport = SurfaceTransport {
            kind: TransportKind::WsFrames,
            url: "/client/v1/surfaces/live_1/ws".into(),
            ticket: "tkt".into(),
            codecs: vec!["jpeg".into()],
        };
        assert_eq!(
            socket_url("https://daemon.example.invalid", &transport),
            "wss://daemon.example.invalid/client/v1/surfaces/live_1/ws"
        );
        assert_eq!(
            socket_url("http://127.0.0.1:8788/", &transport),
            "ws://127.0.0.1:8788/client/v1/surfaces/live_1/ws"
        );
    }

    #[test]
    fn a_frame_reads_as_a_data_uri_and_the_rest_as_events() {
        let frame = media::encode(
            &MediaHeader::Frame {
                v: 1,
                seq: 4,
                at_ms: 0,
                codec: "jpeg".into(),
                w: 2,
                h: 1,
                key: true,
            },
            b"ab",
        );
        assert_eq!(
            read(&frame),
            Some(MediaEvent::Frame {
                seq: 4,
                w: 2,
                h: 1,
                uri: "data:image/jpeg;base64,YWI=".into()
            })
        );
        let closed = media::encode(
            &MediaHeader::Closed {
                reason: "gone".into(),
            },
            &[],
        );
        assert_eq!(read(&closed), Some(MediaEvent::Closed("gone".into())));
        assert_eq!(read(b"junk"), None);
    }

    #[test]
    fn input_is_the_protocols_shape_and_clamped() {
        let sent: serde_json::Value =
            serde_json::from_str(&input(pointer("down", 1.4, 0.5))).unwrap();
        assert_eq!(sent["type"], "input");
        assert_eq!(sent["event"]["type"], "pointer");
        assert_eq!(sent["event"]["x"], 1.0);
        let [down, up] = key_press("Enter", "Enter");
        assert!(matches!(down, SurfaceInput::Key { action, .. } if action == "down"));
        assert!(matches!(up, SurfaceInput::Key { action, .. } if action == "up"));
    }
}
