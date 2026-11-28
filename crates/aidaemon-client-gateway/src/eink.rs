//! What a device paired over the bridge (an e-ink reader) may do.
//!
//! Such a device's page is served by a host it cannot verify and runs in an
//! old browser, so the daemon gives it a small, fixed authority whatever the
//! page asks for: these commands only, project folders inside the home
//! folder and not hidden, and PNG uploads within size limits. Everything is
//! judged from the device's record ([`crate::pairing::Principal::eink`]),
//! never from what a request says about itself.

use std::path::{Component, Path, PathBuf};

use aidaemon_client_protocol::{Command, EinkAccess, ErrorCode, ProtocolError, Risk, Scope};
use base64::Engine as _;

/// The device kind recorded for a bridge device.
pub(crate) const EINK_KIND: &str = "eink";

/// Largest PNG an e-ink device may upload (the vision default per image).
pub(crate) const MAX_PNG_BYTES: usize = 4 * 1024 * 1024;
/// Largest width or height of that PNG.
pub(crate) const MAX_PNG_SIDE: u32 = 4096;
/// Most pixels in that PNG (checked before anything decodes it).
pub(crate) const MAX_PNG_PIXELS: u64 = 8_000_000;

/// A reader's access level, read from its record's scopes and ceiling (no
/// column of its own): no Chat is View only, a High ceiling is Full, and
/// anything else (a Medium ceiling, the level readers paired before levels
/// had) is Limited.
pub(crate) fn level_of(scopes: &[Scope], ceiling: Option<Risk>) -> EinkAccess {
    if !scopes.contains(&Scope::Chat) {
        EinkAccess::ViewOnly
    } else if ceiling == Some(Risk::High) {
        EinkAccess::Full
    } else {
        EinkAccess::Limited
    }
}

/// [`level_of`] for a reader's principal.
pub(crate) fn level(principal: &crate::pairing::Principal) -> EinkAccess {
    level_of(&principal.scopes, principal.ceiling)
}

/// The scopes a level records. `None`: not a level to set.
pub(crate) fn scopes_for(level: EinkAccess) -> Option<Vec<Scope>> {
    match level {
        EinkAccess::ViewOnly => Some(vec![Scope::Observe]),
        EinkAccess::Limited | EinkAccess::Full => {
            Some(vec![Scope::Observe, Scope::Chat, Scope::Approve])
        }
        EinkAccess::Unknown => None,
    }
}

/// The risk ceiling a level records: what it may approve once, at most.
/// Never above High, so a Critical card always waits for a trusted device.
pub(crate) fn ceiling_for(level: EinkAccess) -> Risk {
    match level {
        EinkAccess::Full => Risk::High,
        _ => Risk::Medium,
    }
}

/// The ceiling a message from `principal` puts on the run it starts or
/// steers: a Limited reader's messages cap it at Medium, so a call above
/// that waits for the owner on the phone or the computer; a Full reader's
/// runs follow the owner's settings like any app message. From the
/// device's record, never from the message.
pub(crate) fn sender_ceiling(
    principal: &crate::pairing::Principal,
) -> Option<aidaemon_channels::context::SenderCeiling> {
    (principal.eink && level(principal) == EinkAccess::Limited)
        .then_some(aidaemon_channels::context::SenderCeiling::Medium)
}

fn refused(message: &str) -> ProtocolError {
    ProtocolError::new(ErrorCode::Forbidden, message)
}

/// Whether `principal` may pair an e-ink reader: the owner's local token,
/// or a device that may answer every approval (Approve with no risk
/// ceiling). An e-ink reader never pairs another device, whatever its
/// scopes say. The live scope is checked again by the dispatcher.
pub(crate) fn may_pair(principal: &crate::pairing::Principal) -> Result<(), ProtocolError> {
    if principal.eink {
        return Err(refused("An e-ink device may not pair devices"));
    }
    if principal.is_owner() {
        return Ok(());
    }
    if principal.may(aidaemon_client_protocol::Scope::Approve) && principal.ceiling.is_none() {
        return Ok(());
    }
    Err(refused(
        "Pair an e-ink device from a device that may answer every approval",
    ))
}

/// Whether `principal` may unpair device `device_id`: whoever may pair
/// ([`may_pair`]), but never the device asking (removing itself from a
/// list by mistake would lock the owner out of that device).
pub(crate) fn may_revoke(
    principal: &crate::pairing::Principal,
    device_id: &str,
) -> Result<(), ProtocolError> {
    may_pair(principal)?;
    if principal.device_id.as_deref() == Some(device_id) {
        return Err(refused(
            "This device cannot remove itself: remove it from another device, or on the \
             computer AIDaemon runs on",
        ));
    }
    Ok(())
}

/// Whether an e-ink device at `level` may send `command`, and with these
/// arguments.
pub(crate) fn check(
    command: &Command,
    level: EinkAccess,
    home: Option<&Path>,
) -> Result<(), ProtocolError> {
    let reads = matches!(
        command,
        Command::ListSessions { .. }
            | Command::GetHistory { .. }
            | Command::GetRun { .. }
            | Command::GetGlance
            | Command::GetSnapshot
            | Command::GetChanges { .. }
            | Command::ListProjects
            | Command::ListModels
    );
    if level == EinkAccess::ViewOnly && !reads {
        return Err(refused(
            "This e-ink device may only look: raise its access in the app to do that",
        ));
    }
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

/// Folders right under home that start programs or hold the system's own
/// settings (login items, launch agents, the user's `PATH`): a project
/// there would let a run write what runs later, outside any ceiling.
const NOT_PROJECTS: [&str; 5] = ["Library", "AppData", "Applications", "bin", "Start Menu"];

/// A project folder an e-ink device may name: inside the home folder, with
/// no hidden part (`.ssh`, `.config`), no `..`, not one of
/// [`NOT_PROJECTS`], and still so once symbolic links are followed.
fn check_project_folder(path: &str, home: Option<&Path>) -> Result<(), ProtocolError> {
    let Some(home) = home else {
        return Err(refused("No home folder to make projects in"));
    };
    let full = check_named_folder(path, home)?;
    // Where it really is: the deepest part that exists, links followed.
    let mut existing = full.as_path();
    while !existing.exists() {
        match existing.parent() {
            Some(parent) => existing = parent,
            None => break,
        }
    }
    if let (Ok(real), Ok(real_home)) = (existing.canonicalize(), home.canonicalize()) {
        let rest = full.strip_prefix(existing).unwrap_or(Path::new(""));
        check_named_folder(&real.join(rest).to_string_lossy(), &real_home)?;
    }
    Ok(())
}

fn check_named_folder(path: &str, home: &Path) -> Result<PathBuf, ProtocolError> {
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
    if let Some(Component::Normal(first)) = inside.components().next()
        && NOT_PROJECTS
            .iter()
            .any(|name| first.to_string_lossy().eq_ignore_ascii_case(name))
    {
        return Err(refused(
            "A project folder from this device may not be a system folder (Library, AppData, bin)",
        ));
    }
    Ok(full)
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
    fn check_full(command: &Command, home: Option<&Path>) -> Result<(), ProtocolError> {
        check(command, EinkAccess::Full, home)
    }

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

    fn device(scopes: &[aidaemon_client_protocol::Scope]) -> crate::pairing::Principal {
        crate::pairing::Principal {
            device_id: Some("dv-synthetic-phone".into()),
            device_name: Some("Synthetic phone".into()),
            scopes: scopes.to_vec(),
            ..crate::pairing::Principal::owner()
        }
    }

    /// The owner's token and a device that may answer every approval pair a
    /// reader; a ceilinged device, an observer and a reader itself do not.
    #[test]
    fn who_may_pair_an_eink_reader() {
        use aidaemon_client_protocol::{Risk, Scope};
        assert!(may_pair(&crate::pairing::Principal::owner()).is_ok());
        assert!(may_pair(&device(&[Scope::Observe, Scope::Chat, Scope::Approve])).is_ok());
        let watch = crate::pairing::Principal {
            ceiling: Some(Risk::Low),
            ..device(&[Scope::Observe, Scope::Approve])
        };
        assert_eq!(may_pair(&watch).unwrap_err().code, ErrorCode::Forbidden);
        assert!(may_pair(&device(&[Scope::Observe])).is_err());
        assert!(may_pair(&device(&[Scope::Observe, Scope::Chat])).is_err());
        let reader = crate::pairing::Principal {
            eink: true,
            ..device(&[Scope::Observe, Scope::Chat, Scope::Approve])
        };
        assert!(may_pair(&reader).is_err());
        // Even a reader with no ceiling and every scope.
        let odd_reader = crate::pairing::Principal {
            eink: true,
            ..crate::pairing::Principal::owner()
        };
        assert!(may_pair(&odd_reader).is_err());
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
        assert!(check_full(&Command::ListDevices, None).is_err());
        assert!(
            check_full(
                &Command::PairEinkDevice {
                    code: "K7F29QXM4HTAPZ6CRVTPVXVT2".into()
                },
                None
            )
            .is_err(),
            "an e-ink device never pairs another"
        );
        assert!(
            check_full(
                &Command::ChangeSettings {
                    change: aidaemon_client_protocol::SettingsChange::Set {
                        key: "client_gateway.eink.bridge_url".into(),
                        value: Some("\"wss://bridge.example.test/v1/daemon\"".into()),
                    }
                },
                None
            )
            .is_err(),
            "an e-ink device never changes settings, e-ink access included"
        );
        assert!(
            check_full(
                &Command::DeleteSession {
                    session_id: "s".into()
                },
                None
            )
            .is_err()
        );
        assert!(check_full(&Command::GetSystem, None).is_err());
    }

    #[test]
    fn project_folders_stay_inside_home_and_unhidden() {
        let home = Path::new("/home/alice");
        assert!(check_full(&create("~/projects/garden"), Some(home)).is_ok());
        assert!(check_full(&create("/home/alice/projects/garden"), Some(home)).is_ok());
        for bad in [
            "~/.ssh",
            "~/projects/.git",
            "~/../bob",
            "/etc",
            "/home/alice",
            "~",
            "/home/alicex/projects",
            "~/Library/LaunchAgents",
            "~/bin",
            "/home/alice/AppData/Roaming",
        ] {
            assert!(
                check_full(&create(bad), Some(home)).is_err(),
                "{bad} passed"
            );
        }
        assert!(check_full(&create("~/projects/garden"), None).is_err());
    }

    /// A link inside home that leads out of it, or into a hidden folder,
    /// is judged where it leads.
    #[cfg(unix)]
    #[test]
    fn project_folders_are_judged_where_their_links_lead() {
        let home = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::create_dir(home.path().join(".ssh")).unwrap();
        std::os::unix::fs::symlink(outside.path(), home.path().join("away")).unwrap();
        std::os::unix::fs::symlink(home.path().join(".ssh"), home.path().join("keys")).unwrap();
        std::fs::create_dir(home.path().join("projects")).unwrap();
        let home_dir = home.path();
        assert!(check_full(&create("~/projects/garden"), Some(home_dir)).is_ok());
        for bad in ["~/away", "~/away/new", "~/keys", "~/keys/new"] {
            assert!(
                check_full(&create(bad), Some(home_dir)).is_err(),
                "{bad} passed"
            );
        }
    }

    #[test]
    fn uploads_must_be_small_pngs() {
        assert!(check_full(&upload(png(930, 240)), None).is_ok());
        assert!(check_full(&upload(png(100_000, 100_000)), None).is_err());
        assert!(check_full(&upload(png(4000, 4000)), None).is_err());
        assert!(check_full(&upload(png(0, 10)), None).is_err());
        let gif = base64::engine::general_purpose::STANDARD.encode(b"GIF89a................");
        assert!(check_full(&upload(gif), None).is_err());
        assert!(check_full(&upload("not base64!".into()), None).is_err());
    }

    fn reader(level: EinkAccess) -> crate::pairing::Principal {
        crate::pairing::Principal {
            device_id: Some("dv-synthetic-reader".into()),
            scopes: scopes_for(level).unwrap(),
            ceiling: Some(ceiling_for(level)),
            eink: true,
            ..crate::pairing::Principal::owner()
        }
    }

    /// Each level records scopes and a ceiling that read back as it; a
    /// reader paired before levels (Observe, Chat, Approve, Medium) is
    /// Limited; only Limited caps the runs it starts.
    #[test]
    fn levels_map_to_scopes_ceilings_and_run_caps() {
        use aidaemon_channels::context::SenderCeiling;
        for level in [EinkAccess::ViewOnly, EinkAccess::Limited, EinkAccess::Full] {
            assert_eq!(
                level_of(&scopes_for(level).unwrap(), Some(ceiling_for(level))),
                level
            );
        }
        assert_eq!(
            scopes_for(EinkAccess::ViewOnly).unwrap(),
            vec![Scope::Observe]
        );
        assert_eq!(ceiling_for(EinkAccess::Limited), Risk::Medium);
        assert_eq!(ceiling_for(EinkAccess::Full), Risk::High);
        assert!(scopes_for(EinkAccess::Unknown).is_none());
        let old = [Scope::Observe, Scope::Chat, Scope::Approve];
        assert_eq!(level_of(&old, Some(Risk::Medium)), EinkAccess::Limited);
        assert_eq!(level_of(&old, None), EinkAccess::Limited);
        assert_eq!(
            sender_ceiling(&reader(EinkAccess::Limited)),
            Some(SenderCeiling::Medium)
        );
        assert_eq!(sender_ceiling(&reader(EinkAccess::Full)), None);
        assert_eq!(sender_ceiling(&reader(EinkAccess::ViewOnly)), None);
    }

    /// View only reads; every level refuses settings, pairing and changing
    /// its own access.
    #[test]
    fn view_only_reads_and_no_level_manages_devices() {
        let command =
            |json: serde_json::Value| -> Command { serde_json::from_value(json).unwrap() };
        let send = command(serde_json::json!({"type": "send_message", "text": "hi"}));
        let resolve = command(serde_json::json!({
            "type": "resolve_approval", "approval_id": "a-1", "decision": "approve_once"
        }));
        let stop = command(serde_json::json!({"type": "stop_turn", "session_id": "s"}));
        for refused in [
            &send,
            &resolve,
            &stop,
            &create("~/projects/garden"),
            &upload(png(10, 10)),
        ] {
            assert!(
                check(
                    refused,
                    EinkAccess::ViewOnly,
                    Some(Path::new("/home/alice"))
                )
                .is_err(),
                "{refused:?}"
            );
        }
        for read in [
            Command::GetGlance,
            Command::GetSnapshot,
            Command::ListProjects,
            Command::ListModels,
        ] {
            assert!(check(&read, EinkAccess::ViewOnly, None).is_ok(), "{read:?}");
        }
        assert!(check(&send, EinkAccess::Limited, None).is_ok());
        assert!(check(&resolve, EinkAccess::Full, None).is_ok());
        let set = Command::SetEinkAccess {
            device_id: "dv-synthetic-reader".into(),
            level: EinkAccess::Full,
        };
        for level in [EinkAccess::ViewOnly, EinkAccess::Limited, EinkAccess::Full] {
            assert!(check(&set, level, None).is_err());
            assert!(may_pair(&reader(level)).is_err());
        }
    }

    #[test]
    fn who_may_remove_a_device() {
        use aidaemon_client_protocol::Risk;
        let owner = crate::pairing::Principal::owner();
        assert!(may_revoke(&owner, "dv-synthetic-phone").is_ok());
        let phone = device(&[Scope::Observe, Scope::Chat, Scope::Approve]);
        assert!(may_revoke(&phone, "dv-synthetic-other").is_ok());
        assert!(
            may_revoke(&phone, "dv-synthetic-phone").is_err(),
            "not itself"
        );
        let watch = crate::pairing::Principal {
            ceiling: Some(Risk::Low),
            ..device(&[Scope::Observe, Scope::Approve])
        };
        assert!(may_revoke(&watch, "dv-synthetic-other").is_err());
        assert!(may_revoke(&reader(EinkAccess::Full), "dv-synthetic-other").is_err());
        let remove = Command::RevokeDevice {
            device_id: "dv-synthetic-other".into(),
        };
        assert!(check(&remove, EinkAccess::Full, None).is_err());
    }
}
