//! The bridge's control messages (SPEC §3) and the strict text forms it
//! accepts: lowercase hex, Crockford Base32 nameplates, exact JSON shapes.

use serde::Deserialize;
use serde_json::json;

/// A daemon's login, the only message it may send before `ready`.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum LoginMessage {
    Login { login_pub: String, sig: String },
}

/// What a logged-in daemon may send as text. Anything else is refused.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum DaemonMessage {
    Promote { conn: u32 },
    Close { conn: u32 },
    Join { nameplate: String },
}

/// A nameplate: 4 characters of the Crockford Base32 alphabet (§3).
pub(crate) type Nameplate = [u8; 4];

const CROCKFORD: &[u8] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

pub(crate) fn nameplate(text: &str) -> Option<Nameplate> {
    let bytes: Nameplate = text.as_bytes().try_into().ok()?;
    bytes
        .iter()
        .all(|byte| CROCKFORD.contains(byte))
        .then_some(bytes)
}

/// Decode exactly `2 * N` lowercase hex characters (§1: hex is lowercase).
pub(crate) fn lower_hex<const N: usize>(text: &str) -> Option<[u8; N]> {
    if text.len() != 2 * N
        || !text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }
    let mut out = [0u8; N];
    hex::decode_to_slice(text, &mut out).ok()?;
    Some(out)
}

/// `(login_pub, sig)` from a login message, with their exact lengths
/// (130 and 128 hex characters); key and signature checks come after.
pub(crate) fn parse_login(text: &str) -> Option<([u8; 65], [u8; 64])> {
    let LoginMessage::Login { login_pub, sig } = serde_json::from_str(text).ok()?;
    Some((lower_hex(&login_pub)?, lower_hex(&sig)?))
}

pub(crate) fn parse_daemon(text: &str) -> Option<DaemonMessage> {
    serde_json::from_str(text).ok()
}

pub(crate) fn challenge(nonce: &[u8; 32]) -> String {
    json!({"type": "challenge", "nonce": hex::encode(nonce)}).to_string()
}

pub(crate) fn ready(route_id: &[u8; 32]) -> String {
    json!({"type": "ready", "route_id": hex::encode(route_id)}).to_string()
}

pub(crate) fn open(conn: u32) -> String {
    json!({"type": "open", "conn": conn}).to_string()
}

/// A browser that waited on `nameplate` joined: the daemon accepts only a
/// pairing handshake on this connection.
pub(crate) fn open_joined(conn: u32, nameplate: &str) -> String {
    json!({"type": "open", "conn": conn, "nameplate": nameplate}).to_string()
}

pub(crate) fn close(conn: u32) -> String {
    json!({"type": "close", "conn": conn}).to_string()
}

pub(crate) fn join_failed(nameplate: &Nameplate) -> String {
    // A valid nameplate is ASCII by construction.
    let text = std::str::from_utf8(nameplate).unwrap_or_default();
    json!({"type": "join_failed", "nameplate": text}).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nameplates_are_strict() {
        assert_eq!(nameplate("7K3Q"), Some(*b"7K3Q"));
        for bad in ["7k3q", "7K3", "7K3QQ", "7K3I", "7K3O", "7K3U", "7K3L", ""] {
            assert_eq!(nameplate(bad), None, "{bad}");
        }
    }

    #[test]
    fn hex_is_lowercase_and_exact() {
        assert_eq!(lower_hex::<2>("0aff"), Some([0x0a, 0xff]));
        assert_eq!(lower_hex::<2>("0AFF"), None);
        assert_eq!(lower_hex::<2>("0af"), None);
        assert_eq!(lower_hex::<2>("0aff00"), None);
    }

    #[test]
    fn daemon_messages_are_exact() {
        assert!(matches!(
            parse_daemon(r#"{"type":"promote","conn":3}"#),
            Some(DaemonMessage::Promote { conn: 3 })
        ));
        assert!(parse_daemon(r#"{"type":"promote","conn":3,"x":1}"#).is_none());
        assert!(parse_daemon(r#"{"type":"open","conn":3}"#).is_none());
        assert!(parse_daemon(r#"{"type":"login","login_pub":"","sig":""}"#).is_none());
        assert!(parse_daemon(r#"{"type":"close","conn":-1}"#).is_none());
    }
}
