//! Shared inbound file attachment helpers for Telegram, Slack, and Discord.

use std::path::{Path, PathBuf};

use aidaemon_events::conversation::{AttachmentProvenance, MessageAttachment};

pub use aidaemon_events::attachments::{
    RESOURCE_ID_PREFIX, new_resource_id, sanitize_filename, save_tool_observation_image,
    sha256_file,
};

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

/// Prefix for the line that follows a marked-up picture's stub
/// ([`markup_stub`]).
pub const INBOUND_MARKUP_STUB_PREFIX: &str = "[Marked up by the owner:";

/// What the model is told about a picture the owner drew marks over: the
/// marks are theirs and point at what they mean; `original` is where the
/// unmarked picture is, when the reader may see local paths.
pub fn markup_stub(original: Option<&str>) -> String {
    let mut stub = format!(
        "{INBOUND_MARKUP_STUB_PREFIX} the red strokes on this picture are theirs, drawn to point \
         at what the message is about; focus on the marked areas"
    );
    if let Some(original) = original {
        stub.push_str(&format!(
            ". The unmarked picture, for what the marks cover: {original}"
        ));
    }
    stub.push(']');
    stub
}

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
        // A marked-up picture's note follows its stub: metadata as well.
        if text[cursor..].starts_with(INBOUND_MARKUP_STUB_PREFIX) {
            let Some(close_rel) = text[cursor..].find(']') else {
                out.push_str(&text[cursor..]);
                break;
            };
            cursor += close_rel + 1;
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

/// What names a message in a list: the words the owner typed, or, for a
/// file sent with none, the file's name.
pub fn title_words(inbound_text: &str) -> String {
    let typed = user_authored_text(inbound_text);
    if !typed.is_empty() {
        return typed;
    }
    inbound_text
        .split_once(INBOUND_FILE_STUB_PREFIX)
        .and_then(|(_, rest)| rest.split_once(" ("))
        .map(|(name, _)| name.trim().to_string())
        .unwrap_or_default()
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
            let stub = match a.resource_id.as_deref() {
                Some(resource_id) => format!("{stub}\n[Resource: {resource_id}]"),
                None => stub,
            };
            match a.marked_up_from.as_deref() {
                Some(original) => format!("{stub}\n{}", markup_stub(Some(original))),
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
                + &attachment
                    .marked_up_from
                    .as_ref()
                    .map(|_| format!("\n{}", markup_stub(None)))
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

/// The refusal for an inbound file over the owner's size limit.
fn too_large(size_bytes: Option<u64>, max_file_size_mb: u64) -> anyhow::Error {
    match size_bytes {
        Some(bytes) => anyhow::anyhow!(
            "File too large ({:.1} MB). Maximum is {} MB.",
            bytes as f64 / 1_048_576.0,
            max_file_size_mb
        ),
        None => anyhow::anyhow!("File too large. Maximum is {max_file_size_mb} MB."),
    }
}

/// Download an inbound file through its bot's transport, refused over
/// `max_file_size_mb`: before downloading when the platform reported the
/// size, and while downloading ([`crate::Channel::download_limited`]) when it
/// did not or reported less.
pub async fn download_inbound<T: crate::Channel + ?Sized>(
    transport: &T,
    attachment: &crate::Attachment,
    max_file_size_mb: u64,
) -> anyhow::Result<Vec<u8>> {
    let max_bytes = max_file_size_mb * 1_048_576;
    if let Some(size) = attachment.size_bytes.filter(|size| *size > max_bytes) {
        return Err(too_large(Some(size), max_file_size_mb));
    }
    match transport.download_limited(attachment, max_bytes).await {
        Ok(bytes) => Ok(bytes),
        Err(crate::ChannelError::TooLarge { .. }) => Err(too_large(None, max_file_size_mb)),
        Err(e) => Err(anyhow::anyhow!(
            "Failed to download {}: {e}",
            attachment.name
        )),
    }
}

/// Save a downloaded inbound file to `inbox_dir`, private to this user,
/// under a sanitized name with a random prefix, and describe it for the
/// message.
pub fn save_inbound(
    inbox_dir: &Path,
    filename: &str,
    mime_type: String,
    bytes: &[u8],
) -> std::io::Result<MessageAttachment> {
    let sanitized = sanitize_filename(filename);
    let dest_name = format!(
        "{}_{}",
        &uuid::Uuid::new_v4().simple().to_string()[..8],
        sanitized
    );
    let dest_path = inbox_dir.join(dest_name);
    crate::write_private_file(inbox_dir, &dest_path, bytes)?;
    tracing::info!(
        file = %dest_path.display(),
        size = bytes.len(),
        mime = %mime_type,
        "Saved inbound file"
    );
    Ok(message_attachment(
        dest_path,
        sanitized,
        mime_type,
        bytes.len() as u64,
    ))
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
        marked_up_from: None,
    }
}

pub fn ensure_attachment_resource_identity(attachment: &mut MessageAttachment) {
    if attachment.resource_id.is_none() {
        attachment.resource_id = Some(new_resource_id());
    }
    if attachment.sha256.is_none() {
        attachment.sha256 = sha256_file(Path::new(&attachment.local_path));
    }
}

/// Infer image MIME from magic bytes when the platform metadata is unreliable.
pub use aidaemon_tools::sniff_image_mime;

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
    fn a_marked_up_picture_says_whose_marks_and_where_the_unmarked_one_is() {
        let mut marked = message_attachment(
            PathBuf::from("/tmp/uploads/b_shot-marked.jpg"),
            "shot-marked.jpg".to_string(),
            "image/jpeg".to_string(),
            100,
        );
        marked.marked_up_from = Some("/tmp/uploads/a_shot.png".to_string());
        let text = build_inbound_text("what is this part?", std::slice::from_ref(&marked));
        assert!(text.contains(INBOUND_MARKUP_STUB_PREFIX), "{text}");
        assert!(text.contains("red strokes"), "{text}");
        assert!(text.contains("/tmp/uploads/a_shot.png"), "{text}");
        // The note is metadata: only the owner's words are theirs.
        assert_eq!(user_authored_text(&text), "what is this part?");
        // Without local paths: the note, no path.
        let text = build_inbound_text_without_local_paths("what is this part?", &[marked]);
        assert!(text.contains(INBOUND_MARKUP_STUB_PREFIX), "{text}");
        assert!(!text.contains("/tmp/uploads"), "{text}");
    }

    #[test]
    fn a_title_is_the_typed_words_or_the_file_name() {
        let file = message_attachment(
            PathBuf::from("/tmp/inbox/a_photo.jpg"),
            "photo.jpg".to_string(),
            "image/jpeg".to_string(),
            286 * 1024,
        );
        let only_file = build_inbound_text("", std::slice::from_ref(&file));
        assert_eq!(title_words(&only_file), "photo.jpg");
        let with_words = build_inbound_text("What is this?", &[file]);
        assert_eq!(title_words(&with_words), "What is this?");
        assert_eq!(title_words("Plan the week"), "Plan the week");
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
