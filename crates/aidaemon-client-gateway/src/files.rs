//! Files with messages, as clients see them: pictures and documents the
//! owner uploads to send, and those the agent sends. Each is registered by
//! the path it lives at (an upload is saved to the daemon's inbox first) and
//! served at a signed path, `/client/v1/files/{id}/{sig}/{name}`, so an image
//! element loads it with no token. The signing key is kept beside the
//! records, so links in the event log keep working across restarts.

use std::path::{Path, PathBuf};

use aidaemon_client_protocol::AttachmentRef;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The largest file a client may upload.
pub const MAX_UPLOAD_BYTES: usize = 10 * 1024 * 1024;

/// Pictures served inline. Anything else (SVG and HTML included, which can
/// run script on the gateway's origin) is served as a download.
const INLINE_IMAGES: &[&str] = &["image/png", "image/jpeg", "image/gif", "image/webp"];

/// One registered file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileRecord {
    pub path: PathBuf,
    pub name: String,
    pub mime: String,
    pub size_bytes: u64,
}

/// The registered files and the key their links are signed with.
pub struct FileStore {
    /// The records (`<id>.json`) and the key.
    dir: PathBuf,
    /// Where uploads are saved (the daemon's inbox, so the agent may read
    /// them like any file a chat sent).
    uploads: PathBuf,
    key: [u8; 32],
}

fn sanitize_name(name: &str) -> String {
    let base = name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or_default()
        .trim()
        .trim_start_matches('.');
    let clean: String = base
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '.' | '-' | '_' | ' ') {
                c
            } else {
                '_'
            }
        })
        .take(100)
        .collect();
    if clean.trim().is_empty() {
        "file".to_string()
    } else {
        clean
    }
}

fn hmac(key: &[u8; 32], message: &[u8]) -> [u8; 32] {
    let mut inner_pad = [0x36u8; 64];
    let mut outer_pad = [0x5cu8; 64];
    for (i, byte) in key.iter().enumerate() {
        inner_pad[i] ^= byte;
        outer_pad[i] ^= byte;
    }
    let inner = Sha256::new()
        .chain_update(inner_pad)
        .chain_update(message)
        .finalize();
    Sha256::new()
        .chain_update(outer_pad)
        .chain_update(inner)
        .finalize()
        .into()
}

fn valid_id(id: &str) -> bool {
    id.len() == 24 && id.bytes().all(|b| b.is_ascii_hexdigit())
}

impl FileStore {
    /// Open (or start) the store in `dir`; uploads are saved to `uploads`.
    pub fn open(dir: impl Into<PathBuf>, uploads: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        owner_only_dir(&dir);
        let key_path = dir.join("signing.key");
        let key = match std::fs::read(&key_path) {
            Ok(bytes) if bytes.len() == 32 => {
                let mut key = [0u8; 32];
                key.copy_from_slice(&bytes);
                key
            }
            _ => {
                let mut key = [0u8; 32];
                key[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
                key[16..].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
                std::fs::write(&key_path, key)?;
                aidaemon_tools::config_file::set_owner_only_permissions(&key_path);
                key
            }
        };
        Ok(Self {
            dir,
            uploads: uploads.into(),
            key,
        })
    }

    fn signature(&self, id: &str) -> String {
        hex::encode(&hmac(&self.key, id.as_bytes())[..16])
    }

    /// Where a client fetches file `id`.
    pub fn url(&self, id: &str, name: &str) -> String {
        format!(
            "/client/v1/files/{id}/{}/{}",
            self.signature(id),
            percent_encode(name)
        )
    }

    fn record_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }

    /// Register the file at `path` (the same path is the same id) and say
    /// how clients reach it.
    pub fn register(
        &self,
        path: &Path,
        name: &str,
        mime: &str,
        size_bytes: u64,
    ) -> std::io::Result<AttachmentRef> {
        let id = hex::encode(&Sha256::digest(path.to_string_lossy().as_bytes())[..12]);
        let name = sanitize_name(name);
        let record = FileRecord {
            path: path.to_path_buf(),
            name: name.clone(),
            mime: mime.to_string(),
            size_bytes,
        };
        let record_path = self.record_path(&id);
        std::fs::write(&record_path, serde_json::to_vec(&record)?)?;
        aidaemon_tools::config_file::set_owner_only_permissions(&record_path);
        Ok(AttachmentRef {
            url: Some(self.url(&id, &name)),
            id,
            name,
            mime: Some(record.mime),
            size_bytes: Some(size_bytes),
        })
    }

    /// Save `bytes` (an upload, or a picture the agent sent) and register
    /// them. A picture's type is read from its bytes, not taken on trust.
    pub fn save(
        &self,
        name: &str,
        mime: Option<&str>,
        bytes: &[u8],
    ) -> std::io::Result<AttachmentRef> {
        let name = sanitize_name(name);
        let mime = aidaemon_tools::sniff_image_mime(bytes)
            .map(str::to_string)
            .or_else(|| mime.map(str::to_string))
            .filter(|m| !m.trim().is_empty())
            .unwrap_or_else(|| "application/octet-stream".to_string());
        std::fs::create_dir_all(&self.uploads)?;
        let dest = self.uploads.join(format!(
            "{}_{}",
            &uuid::Uuid::new_v4().simple().to_string()[..8],
            name
        ));
        std::fs::write(&dest, bytes)?;
        aidaemon_tools::config_file::set_owner_only_permissions(&dest);
        self.register(&dest, &name, &mime, bytes.len() as u64)
    }

    /// A message's files as clients reach them; one that cannot be
    /// registered is left out (and logged).
    pub fn show(
        &self,
        attachments: &[aidaemon_tools::semantics::MessageAttachment],
    ) -> Vec<AttachmentRef> {
        attachments
            .iter()
            .filter_map(|a| {
                self.register(
                    Path::new(&a.local_path),
                    &a.filename,
                    &a.mime_type,
                    a.size_bytes,
                )
                .map_err(|e| tracing::warn!(error = %e, "Could not register a message's file"))
                .ok()
            })
            .collect()
    }

    /// The record of file `id`.
    pub fn get(&self, id: &str) -> Option<FileRecord> {
        if !valid_id(id) {
            return None;
        }
        let text = std::fs::read(self.record_path(id)).ok()?;
        serde_json::from_slice(&text).ok()
    }

    /// The record of file `id` when `signature` is its link's.
    pub fn signed(&self, id: &str, signature: &str) -> Option<FileRecord> {
        let expected = self.signature(id);
        // Constant time over equal lengths.
        if expected.len() != signature.len()
            || expected
                .bytes()
                .zip(signature.bytes())
                .fold(0u8, |acc, (a, b)| acc | (a ^ b))
                != 0
        {
            return None;
        }
        self.get(id)
    }

    /// Whether a file of `mime` is shown in place rather than downloaded.
    pub fn inline(mime: &str) -> bool {
        INLINE_IMAGES.contains(&mime)
    }
}

fn percent_encode(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for byte in name.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

#[cfg(unix)]
fn owner_only_dir(dir: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
}

#[cfg(not(unix))]
fn owner_only_dir(_dir: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;

    const PNG: &[u8] = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR";

    #[test]
    fn a_saved_picture_is_served_only_at_its_signed_link() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::open(dir.path().join("files"), dir.path().join("inbox")).unwrap();
        let saved = store
            .save("../../etc/receipt.png", Some("text/html"), PNG)
            .unwrap();
        assert_eq!(saved.name, "receipt.png");
        // The bytes say what it is.
        assert_eq!(saved.mime.as_deref(), Some("image/png"));
        let url = saved.url.clone().unwrap();
        let parts: Vec<&str> = url
            .trim_start_matches("/client/v1/files/")
            .split('/')
            .collect();
        let record = store.signed(parts[0], parts[1]).expect("its own link");
        assert!(record.path.starts_with(dir.path().join("inbox")));
        assert_eq!(std::fs::read(&record.path).unwrap(), PNG);
        assert!(store.signed(parts[0], "0".repeat(32).as_str()).is_none());
        assert!(store.get("../secret").is_none());
        // Links survive a restart: the key is kept.
        let again = FileStore::open(dir.path().join("files"), dir.path().join("inbox")).unwrap();
        assert!(again.signed(parts[0], parts[1]).is_some());
        assert!(FileStore::inline("image/png"));
        assert!(!FileStore::inline("image/svg+xml"));
    }

    #[test]
    fn the_same_path_is_the_same_file() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::open(dir.path().join("files"), dir.path().join("inbox")).unwrap();
        let path = dir.path().join("report.pdf");
        std::fs::write(&path, b"%PDF").unwrap();
        let a = store
            .register(&path, "report.pdf", "application/pdf", 4)
            .unwrap();
        let b = store
            .register(&path, "report.pdf", "application/pdf", 4)
            .unwrap();
        assert_eq!(a, b);
    }
}
