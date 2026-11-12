//! Files a tool hands to the conversation: resource handles and saved
//! tool artifacts and observations.

use std::io::{self, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::conversation::{AttachmentProvenance, MessageAttachment};

/// Prefix of every session resource handle minted here and consumed by
/// `send_file`/`image_generation`; the checklist binds `res_*` targets to it.
pub const RESOURCE_ID_PREFIX: &str = "res_";

pub fn new_resource_id() -> String {
    format!("{RESOURCE_ID_PREFIX}{}", uuid::Uuid::new_v4().simple())
}

pub fn sha256_file(path: &Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Some(hex::encode(hasher.finalize()))
}

pub fn tool_artifact_attachment(
    path: &Path,
    filename: String,
    mime_type: String,
    size_bytes: u64,
    source_tool: &str,
) -> MessageAttachment {
    MessageAttachment {
        resource_id: Some(new_resource_id()),
        local_path: path.to_string_lossy().into_owned(),
        filename,
        mime_type,
        size_bytes,
        provenance: AttachmentProvenance::ToolArtifact,
        source_tool: Some(source_tool.to_string()),
        sha256: sha256_file(path),
    }
}

/// Save tool-produced image bytes to the shared inbox for agent vision context.
pub fn save_tool_observation_image(
    inbox_dir: &Path,
    bytes: &[u8],
    filename_hint: &str,
    mime_type: &str,
    source_tool: &str,
) -> io::Result<MessageAttachment> {
    std::fs::create_dir_all(inbox_dir)?;
    let sanitized = sanitize_filename(filename_hint);
    let uuid_prefix = uuid::Uuid::new_v4().to_string()[..8].to_string();
    let dest_name = format!("{uuid_prefix}_{sanitized}");
    let dest_path = inbox_dir.join(&dest_name);
    std::fs::write(&dest_path, bytes)?;
    Ok(MessageAttachment {
        resource_id: Some(new_resource_id()),
        local_path: dest_path.to_string_lossy().into_owned(),
        filename: sanitized,
        mime_type: mime_type.to_string(),
        size_bytes: bytes.len() as u64,
        provenance: AttachmentProvenance::ToolObservation,
        source_tool: Some(source_tool.to_string()),
        sha256: Some(hex::encode(Sha256::digest(bytes))),
    })
}

/// Sanitize a filename: remove path separators, null bytes, and limit length.
pub fn sanitize_filename(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .filter(|c| *c != '/' && *c != '\\' && *c != '\0')
        .collect();
    // Strip path traversal sequences
    let sanitized = sanitized.replace("..", "");
    // Limit to 200 chars, preserving extension
    if sanitized.len() <= 200 {
        sanitized
    } else if let Some(dot_pos) = sanitized.rfind('.') {
        let ext = &sanitized[dot_pos..];
        if ext.len() < 20 {
            let stem_len = 200 - ext.len();
            let boundary = aidaemon_tools::text::floor_char_boundary(&sanitized, stem_len);
            format!("{}{}", &sanitized[..boundary], ext)
        } else {
            let boundary = aidaemon_tools::text::floor_char_boundary(&sanitized, 200);
            sanitized[..boundary].to_string()
        }
    } else {
        let boundary = aidaemon_tools::text::floor_char_boundary(&sanitized, 200);
        sanitized[..boundary].to_string()
    }
}
