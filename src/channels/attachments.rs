//! Shared inbound file attachment helpers for Telegram, Slack, and Discord.

use std::io::{self, Read};
use std::path::{Path, PathBuf};

use crate::traits::{AttachmentProvenance, MessageAttachment};
use sha2::{Digest, Sha256};

use super::formatting::sanitize_filename;

/// Format a single saved file as the legacy text stub (backward compatible).
pub fn format_file_stub(
    filename: &str,
    size_bytes: u64,
    mime_type: &str,
    dest_path: &Path,
) -> String {
    let size_display = if size_bytes > 1_048_576 {
        format!("{:.1} MB", size_bytes as f64 / 1_048_576.0)
    } else {
        format!("{:.0} KB", size_bytes as f64 / 1024.0)
    };
    format!(
        "[File received: {} ({}, {})\nSaved to: {}]",
        filename,
        size_display,
        mime_type,
        dest_path.display()
    )
}

/// Prefix for channel-injected file metadata blocks (`format_file_stub`).
pub const INBOUND_FILE_STUB_PREFIX: &str = "[File received:";

/// Prefix for the opaque resource handle line paired with an inbound file stub.
pub const INBOUND_RESOURCE_STUB_PREFIX: &str = "[Resource: res_";

/// Remove channel-injected `[File received: ...]` blocks from inbound message text.
///
/// Channels prepend these stubs so the LLM knows a file was saved, but paths
/// inside them must not drive intent heuristics (`needs_tools`, project scope,
/// completion contract) — the user did not type them.
pub fn strip_inbound_file_stubs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    while cursor < text.len() {
        let Some(rel) = text[cursor..].find(INBOUND_FILE_STUB_PREFIX) else {
            out.push_str(&text[cursor..]);
            break;
        };
        let start = cursor + rel;
        out.push_str(&text[cursor..start]);
        let Some(close_rel) = text[start..].find(']') else {
            out.push_str(&text[start..]);
            break;
        };
        cursor = start + close_rel + 1;
        if text[cursor..].starts_with("\r\n") {
            cursor += 2;
        } else if text[cursor..].starts_with('\n') {
            cursor += 1;
        }
        // Resource handles are injected by the channel immediately after the
        // legacy file block. They are runtime metadata too, not user-authored
        // language, so keep them out of intent and completion heuristics.
        if text[cursor..].starts_with(INBOUND_RESOURCE_STUB_PREFIX) {
            let Some(resource_close_rel) = text[cursor..].find(']') else {
                out.push_str(&text[cursor..]);
                break;
            };
            cursor += resource_close_rel + 1;
            if text[cursor..].starts_with("\r\n") {
                cursor += 2;
            } else if text[cursor..].starts_with('\n') {
                cursor += 1;
            }
        }
    }
    out.trim().to_string()
}

/// User-typed caption/body only — excludes channel file metadata stubs.
pub fn user_authored_text(inbound_text: &str) -> String {
    strip_inbound_file_stubs(inbound_text)
}

/// Build the user-visible text from optional caption/body plus file stubs.
pub fn build_inbound_text(user_text: &str, attachments: &[MessageAttachment]) -> String {
    let stubs: Vec<String> = attachments
        .iter()
        .map(|a| {
            let stub = format_file_stub(
                &a.filename,
                a.size_bytes,
                &a.mime_type,
                Path::new(&a.local_path),
            );
            match a.resource_id.as_deref() {
                Some(resource_id) => format!("{stub}\n[Resource: {resource_id}]"),
                None => stub,
            }
        })
        .collect();
    let file_block = stubs.join("\n");
    let user_text = user_text.trim();
    if file_block.is_empty() {
        user_text.to_string()
    } else if user_text.is_empty() {
        file_block
    } else {
        format!("{file_block}\n{user_text}")
    }
}

/// Build non-owner attachment context without exposing the daemon's local
/// inbox path. The structured attachment still travels separately for vision
/// and file handling; only the model-visible text stub is redacted.
pub fn build_inbound_text_without_local_paths(
    user_text: &str,
    attachments: &[MessageAttachment],
) -> String {
    let stubs = attachments
        .iter()
        .map(|attachment| {
            let size_display = if attachment.size_bytes > 1_048_576 {
                format!("{:.1} MB", attachment.size_bytes as f64 / 1_048_576.0)
            } else {
                format!("{:.0} KB", attachment.size_bytes as f64 / 1024.0)
            };
            format!(
                "[File received: {} ({}, {})]",
                attachment.filename, size_display, attachment.mime_type
            ) + &attachment
                .resource_id
                .as_deref()
                .map(|id| format!("\n[Resource: {id}]"))
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join("\n");
    let user_text = user_text.trim();
    if stubs.is_empty() {
        user_text.to_string()
    } else if user_text.is_empty() {
        stubs
    } else {
        format!("{stubs}\n{user_text}")
    }
}

pub fn message_attachment(
    dest_path: PathBuf,
    filename: String,
    mime_type: String,
    size_bytes: u64,
) -> MessageAttachment {
    MessageAttachment {
        resource_id: Some(new_resource_id()),
        local_path: dest_path.to_string_lossy().into_owned(),
        filename,
        mime_type,
        size_bytes,
        provenance: AttachmentProvenance::Inbound,
        source_tool: None,
        sha256: sha256_file(&dest_path),
    }
}

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
    Some(format!("{:x}", hasher.finalize()))
}

pub fn ensure_attachment_resource_identity(attachment: &mut MessageAttachment) {
    if attachment.resource_id.is_none() {
        attachment.resource_id = Some(new_resource_id());
    }
    if attachment.sha256.is_none() {
        attachment.sha256 = sha256_file(Path::new(&attachment.local_path));
    }
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
        sha256: Some(format!("{:x}", Sha256::digest(bytes))),
    })
}

/// Infer image MIME from magic bytes when the platform metadata is unreliable.
pub fn sniff_image_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
        return Some("image/jpeg");
    }
    if bytes.len() >= 8 && bytes[0..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] {
        return Some("image/png");
    }
    if bytes.len() >= 6 && (&bytes[0..6] == b"GIF87a" || &bytes[0..6] == b"GIF89a") {
        return Some("image/gif");
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    None
}

/// Pick filename + MIME for Telegram photos (platform often lacks reliable metadata).
pub fn telegram_photo_filename_and_mime(bytes: &[u8]) -> (String, String) {
    match sniff_image_mime(bytes) {
        Some("image/png") => ("photo.png".to_string(), "image/png".to_string()),
        Some("image/gif") => ("photo.gif".to_string(), "image/gif".to_string()),
        Some("image/webp") => ("photo.webp".to_string(), "image/webp".to_string()),
        _ => ("photo.jpg".to_string(), "image/jpeg".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_file_stub_matches_legacy_shape() {
        let stub = format_file_stub(
            "doc.pdf",
            512 * 1024,
            "application/pdf",
            Path::new("/tmp/inbox/abc_doc.pdf"),
        );
        assert!(stub.starts_with("[File received: doc.pdf"));
        assert!(stub.contains("512 KB"));
        assert!(stub.contains("application/pdf"));
    }

    #[test]
    fn non_owner_attachment_context_hides_local_path() {
        let attachments = vec![MessageAttachment {
            local_path: "/Users/owner/.aidaemon/inbox/private.txt".to_string(),
            filename: "private.txt".to_string(),
            mime_type: "text/plain".to_string(),
            size_bytes: 1024,
            ..MessageAttachment::default()
        }];
        let text = build_inbound_text_without_local_paths("please review", &attachments);
        assert!(text.contains("private.txt"));
        assert!(text.contains("please review"));
        assert!(!text.contains("/Users/owner"));
        assert!(!text.contains("Saved to:"));
    }

    #[test]
    fn build_inbound_text_combines_stub_and_caption() {
        let attachments = vec![message_attachment(
            PathBuf::from("/tmp/a.png"),
            "a.png".to_string(),
            "image/png".to_string(),
            100,
        )];
        let text = build_inbound_text("what is this?", &attachments);
        assert!(text.contains("[File received: a.png"));
        assert!(text.ends_with("what is this?"));
    }

    #[test]
    fn strip_inbound_file_stubs_removes_metadata_paths() {
        let stub = format_file_stub(
            "photo.jpg",
            221 * 1024,
            "image/jpeg",
            Path::new("/Users/alice/.aidaemon/files/inbox/photo.jpg"),
        );
        assert_eq!(strip_inbound_file_stubs(&stub), "");
        assert_eq!(user_authored_text(&stub), "");
    }

    #[test]
    fn strip_inbound_file_stubs_preserves_user_caption() {
        let attachments = vec![message_attachment(
            PathBuf::from("/tmp/a.png"),
            "a.png".to_string(),
            "image/png".to_string(),
            100,
        )];
        let inbound = build_inbound_text("what is this?", &attachments);
        assert_eq!(user_authored_text(&inbound), "what is this?");
    }

    #[test]
    fn strip_inbound_file_stubs_preserves_user_paths_in_caption() {
        let attachments = vec![message_attachment(
            PathBuf::from("/tmp/inbox/doc.pdf"),
            "doc.pdf".to_string(),
            "application/pdf".to_string(),
            100,
        )];
        let inbound = build_inbound_text("Please check ~/project/Cargo.toml", &attachments);
        assert_eq!(
            user_authored_text(&inbound),
            "Please check ~/project/Cargo.toml"
        );
    }

    #[test]
    fn strip_inbound_file_stubs_leaves_plain_text_untouched() {
        let text = "Please check ~/project/Cargo.toml";
        assert_eq!(strip_inbound_file_stubs(text), text);
    }

    #[test]
    fn save_tool_observation_image_writes_inbox_file() {
        let dir = tempfile::tempdir().unwrap();
        let attachment = save_tool_observation_image(
            dir.path(),
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
            "screenshot.png",
            "image/png",
            "browser",
        )
        .unwrap();
        assert_eq!(attachment.provenance, AttachmentProvenance::ToolObservation);
        assert_eq!(attachment.source_tool.as_deref(), Some("browser"));
        assert_eq!(attachment.mime_type, "image/png");
        assert!(Path::new(&attachment.local_path).exists());
    }
}
