//! What a device paired over the bridge (an e-ink reader) may do.
//!
//! Such a device's page is served by a host it cannot verify and runs in an
//! old browser, so the daemon gives it a small, fixed authority whatever the
//! page asks for: these commands only, project folders inside the home
//! folder and not hidden, and PNG uploads within size limits. Everything is
//! judged from the device's record ([`crate::pairing::Principal::eink`]),
//! never from what a request says about itself.

use std::path::{Component, Path, PathBuf};

use aidaemon_client_protocol::{Command, ErrorCode, ProtocolError};
use base64::Engine as _;

/// The device kind recorded for a bridge device.
pub(crate) const EINK_KIND: &str = "eink";

/// Largest PNG an e-ink device may upload (the vision default per image).
pub(crate) const MAX_PNG_BYTES: usize = 4 * 1024 * 1024;
/// Largest width or height of that PNG.
pub(crate) const MAX_PNG_SIDE: u32 = 4096;
/// Most pixels in that PNG (checked before anything decodes it).
pub(crate) const MAX_PNG_PIXELS: u64 = 8_000_000;

/// The ceiling a message from `principal` puts on the run it starts or
/// steers: an e-ink device's messages cap it at Medium, so a call above
/// that waits for the owner on the phone or the computer. From the
/// device's record, never from the message.
pub(crate) fn sender_ceiling(
    principal: &crate::pairing::Principal,
) -> Option<aidaemon_channels::context::SenderCeiling> {
    principal
        .eink
        .then_some(aidaemon_channels::context::SenderCeiling::Medium)
}

fn refused(message: &str) -> ProtocolError {
    ProtocolError::new(ErrorCode::Forbidden, message)
}

/// Whether an e-ink device may send `command`, and with these arguments.
pub(crate) fn check(command: &Command, home: Option<&Path>) -> Result<(), ProtocolError> {
    match command {
        Command::ListSessions { .. }
        | Command::GetHistory { .. }
        | Command::SendMessage { .. }
        | Command::StopTurn { .. }
        | Command::CancelTask { .. }
        | Command::GetRun { .. }
        | Command::GetGlance
        | Command::GetSnapshot
        | Command::GetChanges { .. }
        | Command::ListProjects
        | Command::ListModels
        | Command::ResolveApproval { .. } => Ok(()),
        Command::CreateProject { path, .. } => match path.as_deref() {
            None | Some("") => Ok(()),
            Some(path) => check_project_folder(path, home),
        },
        Command::UploadFile { data, .. } => check_png(data),
        _ => Err(refused("This e-ink device may not do that")),
    }
}

/// A project folder an e-ink device may name: inside the home folder, with
/// no hidden part (`.ssh`, `.config`) and no `..`.
fn check_project_folder(path: &str, home: Option<&Path>) -> Result<(), ProtocolError> {
    let Some(home) = home else {
        return Err(refused("No home folder to make projects in"));
    };
    let full: PathBuf = match path.strip_prefix("~/") {
        Some(rest) => home.join(rest),
        None if path == "~" => home.to_path_buf(),
        None => PathBuf::from(path),
    };
    let Ok(inside) = full.strip_prefix(home) else {
        return Err(refused(
            "A project folder from this device must be inside your home folder",
        ));
    };
    if inside.as_os_str().is_empty() {
        return Err(refused(
            "Pick a folder inside your home folder, not the home folder itself",
        ));
    }
    for part in inside.components() {
        match part {
            Component::Normal(name) if !name.to_string_lossy().starts_with('.') => {}
            _ => {
                return Err(refused(
                    "A project folder from this device may not be hidden or go up a level",
                ));
            }
        }
    }
    Ok(())
}

/// A PNG within limits, judged from its header before anything decodes it.
fn check_png(data: &str) -> Result<(), ProtocolError> {
    // The base64 text bounds the size before decoding it.
    if data.len() > MAX_PNG_BYTES.div_ceil(3) * 4 + 4 {
        return Err(refused("Pictures from this device must be 4 MB or smaller"));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data.trim())
        .map_err(|_| ProtocolError::new(ErrorCode::Invalid, "The file is not base64"))?;
    if bytes.len() > MAX_PNG_BYTES {
        return Err(refused("Pictures from this device must be 4 MB or smaller"));
    }
    let (width, height) =
        png_size(&bytes).ok_or_else(|| refused("This device may upload PNG pictures only"))?;
    if width == 0
        || height == 0
        || width > MAX_PNG_SIDE
        || height > MAX_PNG_SIDE
        || u64::from(width) * u64::from(height) > MAX_PNG_PIXELS
    {
        return Err(refused("That picture is too large"));
    }
    Ok(())
}

/// A PNG's width and height from its signature and first (IHDR) chunk.
fn png_size(bytes: &[u8]) -> Option<(u32, u32)> {
    // PNG machine format: 8-byte signature, then the IHDR chunk
    // (length 13, type "IHDR", width, height).
    const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    if bytes.len() < 24 || bytes[..8] != SIGNATURE || &bytes[12..16] != b"IHDR" {
        return None;
    }
    if u32::from_be_bytes(bytes[8..12].try_into().ok()?) != 13 {
        return None;
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let height = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    Some((width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32) -> String {
        let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        bytes.extend_from_slice(&13u32.to_be_bytes());
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[8, 6, 0, 0, 0]);
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    /// Only an e-ink device's messages carry the Medium ceiling, from its
    /// record; the owner's own token and an app device carry none.
    #[test]
    fn only_an_eink_device_caps_the_run_its_message_starts() {
        let owner = crate::pairing::Principal::owner();
        assert_eq!(sender_ceiling(&owner), None);
        let app = crate::pairing::Principal {
            device_id: Some("dv-synthetic-phone".into()),
            ..crate::pairing::Principal::owner()
        };
        assert_eq!(sender_ceiling(&app), None);
        let reader = crate::pairing::Principal {
            device_id: Some("dv-synthetic-reader".into()),
            eink: true,
            ..crate::pairing::Principal::owner()
        };
        assert_eq!(
            sender_ceiling(&reader),
            Some(aidaemon_channels::context::SenderCeiling::Medium)
        );
    }

    fn upload(data: String) -> Command {
        Command::UploadFile {
            name: "line.png".into(),
            mime: Some("image/png".into()),
            data,
            marked_from: None,
        }
    }

    fn create(path: &str) -> Command {
        Command::CreateProject {
            name: "Garden".into(),
            path: Some(path.into()),
            create_folder: true,
        }
    }

    #[test]
    fn only_the_listed_commands_pass() {
        assert!(check(&Command::ListDevices, None).is_err());
        assert!(
            check(
                &Command::DeleteSession {
                    session_id: "s".into()
                },
                None
            )
            .is_err()
        );
        assert!(check(&Command::GetSystem, None).is_err());
    }

    #[test]
    fn project_folders_stay_inside_home_and_unhidden() {
        let home = Path::new("/home/alice");
        assert!(check(&create("~/projects/garden"), Some(home)).is_ok());
        assert!(check(&create("/home/alice/projects/garden"), Some(home)).is_ok());
        for bad in [
            "~/.ssh",
            "~/projects/.git",
            "~/../bob",
            "/etc",
            "/home/alice",
            "~",
            "/home/alicex/projects",
        ] {
            assert!(check(&create(bad), Some(home)).is_err(), "{bad} passed");
        }
        assert!(check(&create("~/projects/garden"), None).is_err());
    }

    #[test]
    fn uploads_must_be_small_pngs() {
        assert!(check(&upload(png(930, 240)), None).is_ok());
        assert!(check(&upload(png(100_000, 100_000)), None).is_err());
        assert!(check(&upload(png(4000, 4000)), None).is_err());
        assert!(check(&upload(png(0, 10)), None).is_err());
        let gif = base64::engine::general_purpose::STANDARD.encode(b"GIF89a................");
        assert!(check(&upload(gif), None).is_err());
        assert!(check(&upload("not base64!".into()), None).is_err());
    }
}
