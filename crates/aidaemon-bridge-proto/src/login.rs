//! The daemon's bridge login (SPEC §3): a signature by the login key over
//! a fixed prefix and the bridge's 32-byte nonce. The login key signs
//! nothing else, and the prefix keeps the signature from meaning anything
//! in another context.

use p256::ecdsa::signature::{Signer, Verifier};
use p256::ecdsa::{Signature, SigningKey};

use crate::ids::parse_point;

/// The prefix of every login signature.
pub const LOGIN_PREFIX: &[u8] = b"aidaemon-bridge-auth-v1:";

/// The bytes a login signature covers.
pub fn login_message(nonce: &[u8; 32]) -> Vec<u8> {
    let mut message = Vec::with_capacity(LOGIN_PREFIX.len() + 32);
    message.extend_from_slice(LOGIN_PREFIX);
    message.extend_from_slice(nonce);
    message
}

/// Sign a bridge challenge (64-byte `r ‖ s`).
pub fn sign_login(login_key: &SigningKey, nonce: &[u8; 32]) -> [u8; 64] {
    let signature: Signature = login_key.sign(&login_message(nonce));
    signature.to_bytes().into()
}

/// Check a login: the key must be a 65-byte uncompressed point and the
/// 64-byte signature must cover this nonce.
pub fn verify_login(login_pub: &[u8], nonce: &[u8; 32], signature: &[u8]) -> bool {
    let Some(key) = parse_point(login_pub) else {
        return false;
    };
    let Ok(signature) = Signature::from_slice(signature) else {
        return false;
    };
    key.verify(&login_message(nonce), &signature).is_ok()
}
