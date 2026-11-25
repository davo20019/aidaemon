//! Files with messages, as clients see them: pictures and documents the
//! owner uploads to send, and those the agent sends. Each is registered by
//! the path it lives at (an upload is saved to the daemon's inbox first) and
//! served at a signed path, `/client/v1/files/{id}/{expires}-{sig}/{name}`,
//! so an image element loads it with no token. The signing key is kept
//! beside the records, so a link keeps working across restarts until it
//! expires ([`LINK_LIFETIME`]); a conversation read again gets new links.

use std::path::{Path, PathBuf};

use aidaemon_client_protocol::AttachmentRef;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The largest file a client may upload.
pub const MAX_UPLOAD_BYTES: usize = 10 * 1024 * 1024;

/// Pictures served inline. Anything else (SVG and HTML included, which can
/// run script on the gateway's origin) is served as a download.
const INLINE_IMAGES: &[&str] = &["image/png", "image/jpeg", "image/gif", "image/webp"];

/// How long a file's link works. A link is a bearer capability (no token,
/// no device), so it must not outlive the device or the leak that carries
/// it for long; reading a conversation's history signs its files again, so
/// only a view held open for more than a day meets an expired link, and
/// reading it again mends that.
pub const LINK_LIFETIME: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// One registered file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileRecord {
    pub path: PathBuf,
    pub name: String,
    pub mime: String,
    pub size_bytes: u64,
    /// A picture's size in pixels, read from its header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pixels: Option<(u32, u32)>,
    /// The file whose picture the owner drew marks over to make this one,
    /// by its id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub marked_from: Option<String>,
    /// What the file at `path` was when registered (its size and change
    /// time, [`stamp_of`]): its link signs this too, and it is served only
    /// while the file is still that, so a link never reads what is written
    /// at the path later. `None`: registered before links were bound so.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stamp: Option<String>,
}

/// The file at `path` as a link is bound to it: its size and modification
/// time (nanoseconds), or `None` when it cannot be read.
pub fn stamp_of(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    Some(format!("{}:{modified}", meta.len()))
}

/// A picture's width and height in pixels, from its header (PNG, GIF,
/// JPEG, WebP); `None` for anything else or a header it cannot read.
pub fn image_size(bytes: &[u8]) -> Option<(u32, u32)> {
    let be16 = |at: usize| Some(u16::from_be_bytes([*bytes.get(at)?, *bytes.get(at + 1)?]) as u32);
    let le16 = |at: usize| Some(u16::from_le_bytes([*bytes.get(at)?, *bytes.get(at + 1)?]) as u32);
    let le24 = |at: usize| {
        Some(u32::from_le_bytes([
            *bytes.get(at)?,
            *bytes.get(at + 1)?,
            *bytes.get(at + 2)?,
            0,
        ]))
    };
    let size = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        let w = u32::from_be_bytes(bytes.get(16..20)?.try_into().ok()?);
        let h = u32::from_be_bytes(bytes.get(20..24)?.try_into().ok()?);
        (w, h)
    } else if bytes.starts_with(b"GIF8") {
        (le16(6)?, le16(8)?)
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        match bytes.get(12..16)? {
            b"VP8 " => (le16(26)? & 0x3fff, le16(28)? & 0x3fff),
            b"VP8L" => {
                let b = bytes.get(21..25)?;
                let bits = u32::from_le_bytes(b.try_into().ok()?);
                ((bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1)
            }
            b"VP8X" => (le24(24)? + 1, le24(27)? + 1),
            _ => return None,
        }
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        // Walk the segments to the frame header (SOF0..SOF15, but not DHT,
        // JPG or DAC, which share the range).
        let mut at = 2;
        loop {
            while *bytes.get(at)? != 0xff {
                at += 1;
            }
            while *bytes.get(at)? == 0xff {
                at += 1;
            }
            let marker = *bytes.get(at)?;
            at += 1;
            if matches!(marker, 0xd8 | 0x01 | 0xd0..=0xd7) {
                continue;
            }
            let length = be16(at)? as usize;
            if (0xc0..=0xcf).contains(&marker) && !matches!(marker, 0xc4 | 0xc8 | 0xcc) {
                break (be16(at + 5)?, be16(at + 3)?);
            }
            at += length;
        }
    } else {
        return None;
    };
    (size.0 > 0 && size.1 > 0).then_some(size)
}

/// The size of the picture at `path`, read from its first bytes.
fn image_size_of(path: &Path) -> Option<(u32, u32)> {
    use std::io::Read;
    let mut head = Vec::new();
    std::fs::File::open(path)
        .ok()?
        .take(1024 * 1024)
        .read_to_end(&mut head)
        .ok()?;
    image_size(&head)
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

    /// The signature of file `id`'s link, as registered with `stamp`,
    /// working until `expires` (seconds since the epoch).
    fn signature(&self, id: &str, stamp: Option<&str>, expires: u64) -> String {
        let signed = format!("{id}\n{}\n{expires}", stamp.unwrap_or_default());
        hex::encode(&hmac(&self.key, signed.as_bytes())[..16])
    }

    /// Where a client fetches file `id`, as registered with `stamp`, for
    /// the next [`LINK_LIFETIME`].
    fn url(&self, id: &str, stamp: Option<&str>, name: &str) -> String {
        self.url_until(id, stamp, name, now_secs() + LINK_LIFETIME.as_secs())
    }

    fn url_until(&self, id: &str, stamp: Option<&str>, name: &str, expires: u64) -> String {
        format!(
            "/client/v1/files/{id}/{expires}-{}/{}",
            self.signature(id, stamp, expires),
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
        // Registered again (a message shown): what it was marked up from
        // stays.
        let marked_from = self.get(&id).and_then(|r| r.marked_from);
        let record = FileRecord {
            marked_from,
            stamp: stamp_of(path),
            path: path.to_path_buf(),
            name: name.clone(),
            mime: mime.to_string(),
            size_bytes,
            pixels: mime
                .starts_with("image/")
                .then(|| image_size_of(path))
                .flatten(),
        };
        let record_path = self.record_path(&id);
        std::fs::write(&record_path, serde_json::to_vec(&record)?)?;
        aidaemon_tools::config_file::set_owner_only_permissions(&record_path);
        Ok(AttachmentRef {
            url: Some(self.url(&id, record.stamp.as_deref(), &name)),
            id,
            name,
            mime: Some(record.mime),
            size_bytes: Some(size_bytes),
            width: record.pixels.map(|(w, _)| w),
            height: record.pixels.map(|(_, h)| h),
            marked_from: record.marked_from,
        })
    }

    /// Note that the owner drew marks over file `original`'s picture to
    /// make file `id`; an unknown `original` is not noted.
    pub fn mark_up(&self, id: &str, original: &str) -> std::io::Result<bool> {
        let (Some(mut record), Some(_)) = (self.get(id), self.get(original)) else {
            return Ok(false);
        };
        record.marked_from = Some(original.to_string());
        let record_path = self.record_path(id);
        std::fs::write(&record_path, serde_json::to_vec(&record)?)?;
        aidaemon_tools::config_file::set_owner_only_permissions(&record_path);
        Ok(true)
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

    /// The record of file `id` when `signature` (`{expires}-{sig}`, the
    /// link's segment) is its link's, the link has not expired, and the
    /// file is still what it was registered as.
    pub fn signed(&self, id: &str, signature: &str) -> Option<FileRecord> {
        self.signed_at(id, signature, now_secs())
    }

    fn signed_at(&self, id: &str, signature: &str, now: u64) -> Option<FileRecord> {
        let (expires, signature) = signature.split_once('-')?;
        let expires: u64 = expires.parse().ok()?;
        // A link signed for longer than links last was not signed here.
        if now >= expires || expires > now + LINK_LIFETIME.as_secs() + 300 {
            return None;
        }
        let record = self.get(id)?;
        if record.stamp.is_some() && stamp_of(&record.path) != record.stamp {
            return None;
        }
        let expected = self.signature(id, record.stamp.as_deref(), expires);
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
        Some(record)
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
        let expires = parts[1].split_once('-').unwrap().0;
        assert!(
            store
                .signed(parts[0], &format!("{expires}-{}", "0".repeat(32)))
                .is_none()
        );
        assert!(store.get("../secret").is_none());
        // Links survive a restart: the key is kept.
        let again = FileStore::open(dir.path().join("files"), dir.path().join("inbox")).unwrap();
        assert!(again.signed(parts[0], parts[1]).is_some());
        assert!(FileStore::inline("image/png"));
        assert!(!FileStore::inline("image/svg+xml"));
    }

    #[test]
    fn picture_sizes_come_from_their_headers() {
        let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
        png.extend_from_slice(&640u32.to_be_bytes());
        png.extend_from_slice(&480u32.to_be_bytes());
        assert_eq!(image_size(&png), Some((640, 480)));
        assert_eq!(image_size(b"GIF89a\x20\x00\x10\x00"), Some((32, 16)));
        // A JPEG: SOI, an APP0 segment, then SOF0 with 100 x 50.
        let jpeg = [
            0xff, 0xd8, 0xff, 0xe0, 0x00, 0x04, 0x00, 0x00, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x00,
            0x32, 0x00, 0x64, 0x03,
        ];
        assert_eq!(image_size(&jpeg), Some((100, 50)));
        let mut webp = b"RIFF\0\0\0\0WEBPVP8X".to_vec();
        webp.extend_from_slice(&[0; 8]);
        webp.extend_from_slice(&[199, 0, 0, 99, 0, 0]);
        assert_eq!(image_size(&webp), Some((200, 100)));
        assert_eq!(image_size(b"%PDF-1.7"), None);
        assert_eq!(image_size(&png[..18]), None);

        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::open(dir.path().join("files"), dir.path().join("inbox")).unwrap();
        let saved = store.save("shot.png", None, &png).unwrap();
        assert_eq!((saved.width, saved.height), (Some(640), Some(480)));
    }

    #[test]
    fn a_link_reads_only_what_was_registered_at_its_path() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::open(dir.path().join("files"), dir.path().join("inbox")).unwrap();
        let path = dir.path().join("plan.pdf");
        std::fs::write(&path, b"%PDF-1").unwrap();
        let first = store
            .register(&path, "plan.pdf", "application/pdf", 6)
            .unwrap();
        let parts = |url: &str| -> (String, String) {
            let mut it = url.trim_start_matches("/client/v1/files/").split('/');
            (it.next().unwrap().into(), it.next().unwrap().into())
        };
        let (id, sig) = parts(first.url.as_deref().unwrap());
        assert!(store.signed(&id, &sig).is_some());
        // Something else written at the path: the old link reads nothing.
        std::fs::write(&path, b"%PDF-1.7 other").unwrap();
        assert!(store.signed(&id, &sig).is_none());
        // Shown again, it gets a link of its own.
        let second = store
            .register(&path, "plan.pdf", "application/pdf", 14)
            .unwrap();
        let (_, sig2) = parts(second.url.as_deref().unwrap());
        assert_ne!(sig, sig2);
        assert!(store.signed(&id, &sig2).is_some());
        assert!(store.signed(&id, &sig).is_none());
    }

    #[test]
    fn a_link_expires() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::open(dir.path().join("files"), dir.path().join("inbox")).unwrap();
        let saved = store.save("shot.png", None, PNG).unwrap();
        let url = saved.url.unwrap();
        let mut it = url.trim_start_matches("/client/v1/files/").split('/');
        let (id, segment) = (it.next().unwrap(), it.next().unwrap());
        let expires: u64 = segment.split_once('-').unwrap().0.parse().unwrap();
        assert!(store.signed_at(id, segment, expires - 1).is_some());
        assert!(store.signed_at(id, segment, expires).is_none());
        // A later expiry under the same signature is refused.
        let sig = segment.split_once('-').unwrap().1;
        assert!(
            store
                .signed(id, &format!("{}-{sig}", expires + 60))
                .is_none()
        );
        // Nor can a link be signed far into the future.
        let record = store.get(id).unwrap();
        let far = now_secs() + 30 * LINK_LIFETIME.as_secs();
        let far_url = store.url_until(id, record.stamp.as_deref(), "shot.png", far);
        let far_segment = far_url.split('/').nth(5).unwrap();
        assert!(store.signed(id, far_segment).is_none());
        // An old, unexpiring link is refused.
        assert!(store.signed(id, sig).is_none());
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
        // The same id (links differ only once a second has passed).
        assert_eq!(a.id, b.id);
        assert_eq!(a.name, b.name);
    }
}
