//! A conversation shell's socket (protocol 1.42): one WebSocket per
//! attached terminal, apart from the main socket so output never waits
//! behind chat events or enters the resume log.
//!
//! - The client's first message is [`ShellClientMessage::Auth`] with the
//!   ticket from `shell_opened`. Nothing else is read before it.
//! - Both sides send JSON text. The server answers the ticket with
//!   [`ShellServerMessage::Ready`], then sends the shell's output so far as
//!   one `output`, then its output as it comes.
//! - A client that fell behind gets `reset` and the output so far again:
//!   it clears its terminal and writes that.
//!
//! Unknown message types read as `Unknown` and are ignored, so either side
//! can add one in a minor version.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ShellInfo;

/// The largest message a client may send.
pub const MAX_CLIENT_MESSAGE: usize = 64 * 1024;

/// What the client sends.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ShellClientMessage {
    /// First, and only first.
    Auth { ticket: String },
    /// What the owner typed or pasted, and the terminal's answers to the
    /// program's queries.
    Input { data: String },
    /// The terminal's size in cells changed.
    Resize { cols: u16, rows: u16 },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// What the server sends.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ShellServerMessage {
    /// The ticket was good: output follows.
    Ready { shell: ShellInfo },
    /// What the shell wrote, to write into the terminal as it is.
    Output { data: String },
    /// Clear the terminal: the output so far comes next.
    Reset,
    /// The shell's program ended.
    Ended {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        exit_code: Option<u32>,
    },
    /// The socket closes, and why.
    Closed { reason: String },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_read_from_the_wire_shape() {
        let input: ShellClientMessage =
            serde_json::from_str(r#"{"type":"input","data":"ls\r"}"#).unwrap();
        assert_eq!(
            input,
            ShellClientMessage::Input {
                data: "ls\r".into()
            }
        );
        let later: ShellClientMessage = serde_json::from_str(r#"{"type":"zoom"}"#).unwrap();
        assert_eq!(later, ShellClientMessage::Unknown);
        let out = serde_json::to_string(&ShellServerMessage::Ended { exit_code: Some(3) }).unwrap();
        assert_eq!(out, r#"{"type":"ended","exit_code":3}"#);
    }
}
