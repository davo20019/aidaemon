//! The media plane of a live surface (protocol 1.30): one WebSocket per live
//! session, apart from the main socket so frames never wait behind chat
//! events or enter the resume log.
//!
//! - The client's first message is [`MediaClientMessage::Auth`] with the
//!   ticket from `surface_opened`. Nothing else is read before it.
//! - The server sends binary messages: a big-endian `u32` header length,
//!   the header as JSON ([`MediaHeader`]), then the payload (a frame's
//!   bytes, or nothing).
//! - The client sends JSON text: [`MediaClientMessage`].
//! - The server sends a viewer its next frame only after the viewer acked
//!   the previous one, and only the newest frame: a slow viewer skips
//!   frames, it never queues them.
//!
//! Unknown header and message types read as `Unknown` and are ignored, so
//! either side can add one in a minor version.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::SurfaceController;

/// The media framing version.
pub const MEDIA_VERSION: u8 = 1;

/// The largest header a peer accepts.
pub const MAX_HEADER_BYTES: usize = 16 * 1024;

/// The header of one binary message from the server.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MediaHeader {
    /// A picture of the surface; the payload is its encoded bytes.
    Frame {
        v: u8,
        seq: u64,
        at_ms: u64,
        codec: String,
        w: u32,
        h: u32,
        /// Stands alone.
        key: bool,
    },
    /// Who controls the surface now.
    Controller { controller: SurfaceController },
    /// The authenticated session is ready; frames follow.
    Ready { surface_id: String },
    /// What the owner's typing goes to now, after their input (protocol
    /// 1.31): its kind ("a text field") and label ("Username"); both absent
    /// when nothing on the surface takes typing. Never a field's value.
    Focus {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        what: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    /// The session ended; the server closes the socket next.
    Closed { reason: String },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// A text message from the client.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MediaClientMessage {
    /// The first message: the ticket from `surface_opened`.
    Auth { ticket: String },
    /// The client drew frame `seq` and can take the next.
    Ack { seq: u64 },
    /// Input for the surface. Only the controlling device's input is
    /// applied; anything else is dropped.
    Input { event: SurfaceInput },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// What the daemon tells a device that shares its own screen (protocol
/// 1.31), as JSON text on its provider socket. The device sends its frames
/// back as binary messages ([`encode`] with [`MediaHeader::Frame`]), after
/// [`MediaClientMessage::Auth`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderCommand {
    /// Someone watches: send frames, at most this size and rate.
    Start {
        max_w: u32,
        max_h: u32,
        max_fps: u32,
        quality: u8,
    },
    /// Nobody watches: stop sending.
    Stop,
    /// Apply this input from the owner.
    Input { event: SurfaceInput },
    /// The owner controls the screen now (`true`) or gave it back: tell
    /// whoever sits at it.
    Controlled { on: bool },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// Input for a surface. Coordinates are fractions of the frame (0 to 1).
#[derive(Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SurfaceInput {
    Pointer {
        /// `down`, `move`, `up` or `cancel`.
        action: String,
        x: f64,
        y: f64,
        /// `left` (default), `right` or `middle`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        button: Option<String>,
    },
    Wheel {
        x: f64,
        y: f64,
        #[serde(default)]
        dx: f64,
        #[serde(default)]
        dy: f64,
    },
    Key {
        /// `down` or `up`.
        action: String,
        /// `KeyboardEvent.code`: `KeyA`, `Enter`, `Backspace`.
        code: String,
        /// `KeyboardEvent.key`.
        #[serde(default)]
        key: String,
        /// `shift`, `ctrl`, `alt`, `meta`.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        mods: Vec<String>,
    },
    /// Committed text (a phone keyboard, a paste, a password manager).
    Text { text: String },
}

impl std::fmt::Debug for SurfaceInput {
    // Typed text is never printed.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pointer { action, .. } => write!(f, "Pointer({action})"),
            Self::Wheel { .. } => f.write_str("Wheel"),
            Self::Key { action, .. } => write!(f, "Key({action})"),
            Self::Text { text } => write!(f, "Text({} chars)", text.chars().count()),
        }
    }
}

/// Encode a binary message: header length, header JSON, payload.
pub fn encode(header: &MediaHeader, payload: &[u8]) -> Vec<u8> {
    let json = serde_json::to_vec(header).unwrap_or_default();
    let mut out = Vec::with_capacity(4 + json.len() + payload.len());
    out.extend_from_slice(&u32::try_from(json.len()).unwrap_or(0).to_be_bytes());
    out.extend_from_slice(&json);
    out.extend_from_slice(payload);
    out
}

/// Why a binary message could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    TooShort,
    HeaderTooLarge,
    BadHeader,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::TooShort => "message too short",
            Self::HeaderTooLarge => "header too large",
            Self::BadHeader => "header is not valid",
        })
    }
}

impl std::error::Error for DecodeError {}

/// Read a binary message into its header and payload.
pub fn decode(bytes: &[u8]) -> Result<(MediaHeader, &[u8]), DecodeError> {
    let len_bytes: [u8; 4] = bytes
        .get(..4)
        .and_then(|b| b.try_into().ok())
        .ok_or(DecodeError::TooShort)?;
    let len = u32::from_be_bytes(len_bytes) as usize;
    if len > MAX_HEADER_BYTES {
        return Err(DecodeError::HeaderTooLarge);
    }
    let header = bytes.get(4..4 + len).ok_or(DecodeError::TooShort)?;
    let header: MediaHeader = serde_json::from_slice(header).map_err(|_| DecodeError::BadHeader)?;
    Ok((header, &bytes[4 + len..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_round_trips() {
        let header = MediaHeader::Frame {
            v: MEDIA_VERSION,
            seq: 7,
            at_ms: 1,
            codec: "jpeg".into(),
            w: 640,
            h: 480,
            key: true,
        };
        let bytes = encode(&header, b"\xff\xd8pixels");
        let (back, payload) = decode(&bytes).unwrap();
        assert_eq!(back, header);
        assert_eq!(payload, b"\xff\xd8pixels");
    }

    #[test]
    fn bad_messages_are_refused() {
        assert_eq!(decode(&[0, 0]), Err(DecodeError::TooShort));
        assert_eq!(decode(&[0, 0, 0, 9, b'{']), Err(DecodeError::TooShort));
        assert_eq!(
            decode(&[0xff, 0xff, 0xff, 0xff]),
            Err(DecodeError::HeaderTooLarge)
        );
        let mut junk = 4u32.to_be_bytes().to_vec();
        junk.extend_from_slice(b"nope");
        assert_eq!(decode(&junk), Err(DecodeError::BadHeader));
    }

    #[test]
    fn unknown_types_read_as_unknown() {
        let mut bytes = 16u32.to_be_bytes().to_vec();
        bytes.extend_from_slice(br#"{"type":"audio"}"#);
        assert_eq!(decode(&bytes).unwrap().0, MediaHeader::Unknown);
        let message: MediaClientMessage = serde_json::from_str(r#"{"type":"clip"}"#).unwrap();
        assert_eq!(message, MediaClientMessage::Unknown);
    }

    #[test]
    fn typed_text_is_never_printed() {
        let input = SurfaceInput::Text {
            text: "correct horse".into(),
        };
        assert!(!format!("{input:?}").contains("horse"));
    }
}
