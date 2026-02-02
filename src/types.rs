/// Response to an approval request from the user.
#[derive(Debug, Clone)]
pub enum ApprovalResponse {
    AllowOnce,
    AllowAlways,
    Deny,
}

/// A media message to be sent through a channel.
pub struct MediaMessage {
    pub session_id: String,
    pub photo_bytes: Vec<u8>,
    pub caption: String,
}
