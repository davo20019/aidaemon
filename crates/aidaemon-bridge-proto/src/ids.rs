//! Identifiers (SPEC §2) and strict P-256 point parsing (SPEC §1).

use p256::PublicKey;
use p256::ecdsa::VerifyingKey;
use p256::elliptic_curve::sec1::ToSec1Point;
use sha2::{Digest, Sha256};

/// Length of an uncompressed SEC1 P-256 point, the only form accepted.
pub const POINT_LEN: usize = 65;

/// Parse a 65-byte uncompressed P-256 point; refuse compressed points,
/// other lengths and points not on the curve.
pub fn parse_point(bytes: &[u8]) -> Option<VerifyingKey> {
    if bytes.len() != POINT_LEN || bytes[0] != 0x04 {
        return None;
    }
    VerifyingKey::from_sec1_bytes(bytes).ok()
}

/// [`parse_point`] as an ECDH public key.
pub fn parse_ecdh_point(bytes: &[u8]) -> Option<PublicKey> {
    if bytes.len() != POINT_LEN || bytes[0] != 0x04 {
        return None;
    }
    PublicKey::from_sec1_bytes(bytes).ok()
}

/// A public key as its 65-byte uncompressed SEC1 encoding.
pub fn point_bytes(key: &PublicKey) -> [u8; POINT_LEN] {
    key.to_sec1_point(false)
        .as_bytes()
        .try_into()
        .expect("uncompressed P-256 point is 65 bytes")
}

/// A verifying key as its 65-byte uncompressed SEC1 encoding.
pub fn verifying_key_bytes(key: &VerifyingKey) -> [u8; POINT_LEN] {
    key.to_sec1_point(false)
        .as_bytes()
        .try_into()
        .expect("uncompressed P-256 point is 65 bytes")
}

/// `H(x)`.
pub fn sha256(x: &[u8]) -> [u8; 32] {
    Sha256::digest(x).into()
}

/// The Crockford Base32 alphabet of nameplates and secrets (SPEC §3, §6).
pub const CROCKFORD: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
/// Characters in a nameplate.
pub const NAMEPLATE_LEN: usize = 4;
/// Characters in a pairing secret (65 bits).
pub const SECRET_LEN: usize = 13;

fn crockford_only(s: &[u8], len: usize) -> bool {
    s.len() == len && s.iter().all(|c| CROCKFORD.contains(c))
}

/// A nameplate is exactly 4 upper-case Crockford Base32 characters.
pub fn is_valid_nameplate(nameplate: &str) -> bool {
    crockford_only(nameplate.as_bytes(), NAMEPLATE_LEN)
}

/// A pairing secret is exactly 13 upper-case Crockford Base32 characters.
pub fn is_valid_secret(secret: &[u8]) -> bool {
    crockford_only(secret, SECRET_LEN)
}

fn tagged(label: &[u8], key: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(label);
    hasher.update(key);
    hasher.finalize().into()
}

/// `H("aidaemon-daemon-id-v1" ‖ identity_pub)`.
pub fn daemon_id(identity_pub: &[u8]) -> [u8; 32] {
    tagged(b"aidaemon-daemon-id-v1", identity_pub)
}

/// `H("aidaemon-device-id-v1" ‖ device_pub)`; also the gateway's peer id.
pub fn device_id(device_pub: &[u8]) -> [u8; 32] {
    tagged(b"aidaemon-device-id-v1", device_pub)
}

/// `H("aidaemon-bridge-route-v1" ‖ login_pub)`.
pub fn route_id(login_pub: &[u8]) -> [u8; 32] {
    tagged(b"aidaemon-bridge-route-v1", login_pub)
}
