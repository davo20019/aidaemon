/// Response to an approval request from the user.
#[derive(Debug, Clone)]
pub enum ApprovalResponse {
    AllowOnce,
    AllowAlways,
    Deny,
}

/// The kind of media being sent.
#[allow(dead_code)]
pub enum MediaKind {
    /// An in-memory photo (e.g. screenshot).
    Photo { data: Vec<u8> },
    /// A file on disk to send as a document.
    Document {
        file_path: String,
        filename: String,
    },
}

/// A media message to be sent through a channel.
pub struct MediaMessage {
    pub session_id: String,
    pub caption: String,
    pub kind: MediaKind,
}
