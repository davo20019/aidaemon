//! Image uploads from the chat into a session's workspace.

use super::*;

pub(crate) struct PendingUpload {
    pub(crate) upload_id: String,
    pub(crate) filename: String,
    pub(crate) mime_type: String,
    pub(crate) expected_bytes: usize,
    pub(crate) bytes: Vec<u8>,
    pub(crate) chunks_received: usize,
    pub(crate) caption: Option<String>,
    pub(crate) touched_at: Instant,
}

pub(crate) struct UploadCommitResult {
    pub(crate) status_message: String,
    pub(crate) prompt_for_agent: Option<String>,
    pub(crate) agent_status_message: Option<String>,
}

pub(crate) struct StoredUpload {
    pub(crate) path: PathBuf,
    pub(crate) saved_inside_session_workspace: bool,
    pub(crate) fallback_reason: Option<String>,
}

pub(crate) fn normalize_image_mime(value: Option<&str>) -> Option<&'static str> {
    let raw = value?.trim().to_ascii_lowercase();
    let normalized = raw.split(';').next().unwrap_or("").trim();
    match normalized {
        "image/png" => Some("image/png"),
        "image/jpeg" | "image/jpg" => Some("image/jpeg"),
        "image/webp" => Some("image/webp"),
        _ => None,
    }
}

pub(crate) fn normalize_upload_id(value: Option<&str>) -> Option<String> {
    let raw = value?.trim();
    if raw.is_empty() {
        return None;
    }
    let cleaned = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        .collect::<String>();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned.chars().take(64).collect())
    }
}

pub(crate) fn sanitize_upload_filename(name: &str, mime_type: &str) -> String {
    let mut out = name
        .chars()
        .filter(|c| *c != '/' && *c != '\\' && *c != '\0')
        .collect::<String>();
    out = out
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | ' '))
        .collect::<String>()
        .trim()
        .to_string();

    if out.is_empty() {
        out = "image".to_string();
    }

    out = out.chars().take(100).collect();
    if Path::new(&out).extension().is_none() {
        let ext = match mime_type {
            "image/png" => "png",
            "image/webp" => "webp",
            _ => "jpg",
        };
        out.push('.');
        out.push_str(ext);
    }
    out
}

pub(crate) fn format_size(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    }
}

pub(crate) fn build_upload_prompt(
    filename: &str,
    path: &Path,
    mime_type: &str,
    size_bytes: usize,
    caption: Option<&str>,
) -> String {
    let mut prompt = format!(
        "I uploaded an image from my phone.\nFile: {}\nSaved path: {}\nMIME: {}\nSize: {}\nPlease analyze this screenshot and help me fix the issue shown.",
        filename,
        path.display(),
        mime_type,
        format_size(size_bytes)
    );
    if let Some(caption) = caption {
        let trimmed = caption.trim();
        if !trimmed.is_empty() {
            let clipped: String = trimmed.chars().take(2000).collect();
            prompt.push_str("\nContext from me: ");
            prompt.push_str(&clipped);
        }
    }
    prompt.push('\n');
    prompt
}

pub(crate) fn session_upload_dir(cwd: &Path) -> PathBuf {
    cwd.join(".aidaemon/files/inbox")
}

pub(crate) fn write_uploaded_image(
    dest_dir: &Path,
    filename: &str,
    bytes: &[u8],
) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dest_dir)?;
    let prefix: String = uuid::Uuid::new_v4().to_string().chars().take(8).collect();
    let dest_name = format!("{}_{}", prefix, filename);
    let dest_path = dest_dir.join(dest_name);
    std::fs::write(&dest_path, bytes)?;
    Ok(dest_path)
}

pub(crate) fn store_uploaded_image(
    session_cwd: &Path,
    inbox_dir: &Path,
    filename: &str,
    bytes: &[u8],
) -> anyhow::Result<StoredUpload> {
    let preferred_dir = session_upload_dir(session_cwd);
    match write_uploaded_image(&preferred_dir, filename, bytes) {
        Ok(path) => Ok(StoredUpload {
            path,
            saved_inside_session_workspace: true,
            fallback_reason: None,
        }),
        Err(err) => {
            if preferred_dir == inbox_dir {
                return Err(err);
            }
            warn!(
                error = %err,
                session_cwd = %session_cwd.display(),
                fallback_inbox_dir = %inbox_dir.display(),
                "Failed to save uploaded image inside session workspace; falling back to global inbox"
            );
            let fallback_path = write_uploaded_image(inbox_dir, filename, bytes)?;
            Ok(StoredUpload {
                path: fallback_path,
                saved_inside_session_workspace: false,
                fallback_reason: Some(err.to_string()),
            })
        }
    }
}

pub(crate) fn payload_upload_id(payload: &Value) -> Option<String> {
    normalize_upload_id(
        payload
            .get("upload_id")
            .and_then(|v| v.as_str())
            .or_else(|| payload.get("id").and_then(|v| v.as_str())),
    )
}

impl TerminalBridge {
    pub(crate) fn prune_stale_uploads(active: &mut ActiveSession) {
        let cutoff = Duration::from_secs(UPLOAD_IDLE_TTL_SECS);
        active
            .pending_uploads
            .retain(|_, upload| upload.touched_at.elapsed() <= cutoff);
    }

    pub(crate) fn begin_image_upload(
        active: &mut ActiveSession,
        payload: &Value,
        max_upload_bytes: usize,
    ) -> anyhow::Result<String> {
        Self::prune_stale_uploads(active);
        if max_upload_bytes == 0 {
            anyhow::bail!("File uploads are disabled in daemon config ([files].enabled = false).");
        }

        if active.pending_uploads.len() >= MAX_PENDING_UPLOADS_PER_SESSION {
            anyhow::bail!(
                "Too many active uploads in this session. Finish or cancel an existing upload first."
            );
        }

        let upload_id = payload_upload_id(payload)
            .ok_or_else(|| anyhow::anyhow!("upload_start requires upload_id"))?;
        if active.pending_uploads.contains_key(&upload_id) {
            anyhow::bail!("Upload id `{}` is already active.", upload_id);
        }

        let mime_type = normalize_image_mime(
            payload
                .get("mime_type")
                .and_then(|v| v.as_str())
                .or_else(|| payload.get("mime").and_then(|v| v.as_str())),
        )
        .ok_or_else(|| anyhow::anyhow!("Unsupported image MIME type. Use png/jpeg/webp."))?
        .to_string();

        let expected_bytes_u64 = payload
            .get("total_bytes")
            .or_else(|| payload.get("size"))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow::anyhow!("upload_start requires total_bytes"))?;
        let expected_bytes = usize::try_from(expected_bytes_u64)
            .map_err(|_| anyhow::anyhow!("upload size is too large for this platform"))?;
        if expected_bytes == 0 {
            anyhow::bail!("Image upload cannot be empty.");
        }
        if expected_bytes > max_upload_bytes {
            anyhow::bail!(
                "Image too large ({}). Maximum is {}.",
                format_size(expected_bytes),
                format_size(max_upload_bytes)
            );
        }

        let raw_filename = payload
            .get("filename")
            .and_then(|v| v.as_str())
            .or_else(|| payload.get("name").and_then(|v| v.as_str()))
            .unwrap_or("image");
        let filename = sanitize_upload_filename(raw_filename, &mime_type);
        let caption = payload
            .get("caption")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        active.pending_uploads.insert(
            upload_id.clone(),
            PendingUpload {
                upload_id: upload_id.clone(),
                filename,
                mime_type,
                expected_bytes,
                bytes: Vec::with_capacity(expected_bytes.min(max_upload_bytes)),
                chunks_received: 0,
                caption,
                touched_at: Instant::now(),
            },
        );

        Ok(format!(
            "Image upload started (id={}, size={}).",
            upload_id,
            format_size(expected_bytes)
        ))
    }

    pub(crate) fn append_image_upload_chunk(
        active: &mut ActiveSession,
        payload: &Value,
        max_upload_bytes: usize,
    ) -> anyhow::Result<Option<String>> {
        let upload_id = payload_upload_id(payload)
            .ok_or_else(|| anyhow::anyhow!("upload_chunk requires upload_id"))?;
        let data_b64 = payload
            .get("data_b64")
            .or_else(|| payload.get("chunk_b64"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if data_b64.is_empty() {
            anyhow::bail!("upload_chunk requires data_b64");
        }
        let chunk = b64_decode_flexible(data_b64)?;
        if chunk.is_empty() {
            anyhow::bail!("upload chunk was empty");
        }
        if chunk.len() > MAX_UPLOAD_CHUNK_BYTES {
            anyhow::bail!(
                "Upload chunk too large ({} bytes > {} bytes).",
                chunk.len(),
                MAX_UPLOAD_CHUNK_BYTES
            );
        }

        let Some(upload) = active.pending_uploads.get_mut(&upload_id) else {
            anyhow::bail!("No active upload for id `{}`.", upload_id);
        };
        upload.touched_at = Instant::now();

        let next_len = upload.bytes.len().saturating_add(chunk.len());
        if next_len > upload.expected_bytes || next_len > max_upload_bytes {
            anyhow::bail!(
                "Upload exceeds declared size ({} > {}).",
                format_size(next_len),
                format_size(upload.expected_bytes)
            );
        }

        upload.bytes.extend_from_slice(&chunk);
        upload.chunks_received = upload.chunks_received.saturating_add(1);

        if upload.bytes.len() == upload.expected_bytes {
            return Ok(Some(format!(
                "Image upload received (id={}, size={}). Send upload_commit to finish.",
                upload.upload_id,
                format_size(upload.bytes.len())
            )));
        }

        if upload.chunks_received % 8 == 0 {
            let pct = (upload.bytes.len() * 100 / upload.expected_bytes).min(99);
            return Ok(Some(format!(
                "Image upload progress: {}% (id={}).",
                pct, upload.upload_id
            )));
        }

        Ok(None)
    }

    pub(crate) fn cancel_image_upload(
        active: &mut ActiveSession,
        payload: &Value,
    ) -> anyhow::Result<String> {
        let upload_id = payload_upload_id(payload)
            .ok_or_else(|| anyhow::anyhow!("upload_cancel requires upload_id"))?;
        if active.pending_uploads.remove(&upload_id).is_some() {
            Ok(format!("Upload `{}` cancelled.", upload_id))
        } else {
            anyhow::bail!("No active upload for id `{}`.", upload_id);
        }
    }

    pub(crate) fn commit_image_upload(
        active: &mut ActiveSession,
        payload: &Value,
        inbox_dir: &Path,
    ) -> anyhow::Result<UploadCommitResult> {
        let upload_id = payload_upload_id(payload)
            .ok_or_else(|| anyhow::anyhow!("upload_commit requires upload_id"))?;
        let Some(mut upload) = active.pending_uploads.remove(&upload_id) else {
            anyhow::bail!("No active upload for id `{}`.", upload_id);
        };
        upload.touched_at = Instant::now();

        if upload.bytes.is_empty() {
            anyhow::bail!("Upload `{}` has no data.", upload_id);
        }
        if upload.bytes.len() != upload.expected_bytes {
            anyhow::bail!(
                "Upload `{}` incomplete (received {}, expected {}).",
                upload_id,
                format_size(upload.bytes.len()),
                format_size(upload.expected_bytes)
            );
        }

        let final_caption = payload
            .get("caption")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .or(upload.caption);

        let stored = store_uploaded_image(&active.cwd, inbox_dir, &upload.filename, &upload.bytes)?;
        let has_active_agent = active.crypto.bootstrapped_agent || active.crypto.agent.is_some();
        let prompt_for_agent = if has_active_agent && stored.saved_inside_session_workspace {
            Some(build_upload_prompt(
                &upload.filename,
                &stored.path,
                &upload.mime_type,
                upload.bytes.len(),
                final_caption.as_deref(),
            ))
        } else {
            None
        };
        let agent_status_message = if has_active_agent && !stored.saved_inside_session_workspace {
            Some(
                "Image was saved outside the session workspace, so it was not auto-sent to the active agent to avoid another sandbox permission error.".to_string(),
            )
        } else {
            None
        };
        let status_message = if let Some(reason) = stored.fallback_reason.as_deref() {
            format!(
                "Image saved to {} ({}). Session workspace upload dir was unavailable, so a fallback inbox path was used: {}",
                stored.path.display(),
                format_size(upload.bytes.len()),
                reason
            )
        } else {
            format!(
                "Image saved to {} ({}).",
                stored.path.display(),
                format_size(upload.bytes.len())
            )
        };

        Ok(UploadCommitResult {
            status_message,
            prompt_for_agent,
            agent_status_message,
        })
    }
}
