//! Received files as a model sees them: images and audio as content blocks
//! when the model takes them, a speech-to-text fallback for voice notes the
//! model cannot hear, and the hints that stand in for what was skipped.

pub mod attachment_content;
pub mod audio;
pub mod stt;
pub mod vision;
