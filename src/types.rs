/// A screenshot message to be sent as a photo via Telegram.
pub struct MediaMessage {
    pub chat_id: i64,
    pub photo_bytes: Vec<u8>,
    pub caption: String,
}
