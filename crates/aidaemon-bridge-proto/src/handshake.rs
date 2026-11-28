//! The two handshakes (SPEC §5 session, §6 pairing) as explicit state
//! machines for the daemon and, as a reference and for tests, the page.
//!
//! Daemon, session mode:
//! `parse_hello` → [`SessionHello::respond`] (after looking up
//! [`SessionHello::device_id`]) → send HELLO_REPLY → [`DaemonSession::finish`]
//! → send `promote` to the bridge → send [`DaemonChannel::ready_frame`].
//!
//! Daemon, pairing mode:
//! `parse_hello` → [`PairingHello::respond`] (with the owner's secret and
//! device key hash) → send HELLO_REPLY → [`DaemonPairing::finish`] → create
//! the device → send `promote` → send [`DaemonPaired::paired_control`]
//! sealed as DATA (the page sends no DATA before it opens `paired`).
//!
//! Every state consumes itself, so a step cannot run twice; every check of a
//! peer's bytes returns a typed [`Error`].

use hkdf::Hkdf;
use hmac::{Hmac, KeyInit, Mac};
use p256::ecdh::diffie_hellman;
use p256::ecdsa::signature::{Signer, Verifier};
use p256::ecdsa::{Signature, SigningKey};
use p256::elliptic_curve::Generate;
use p256::{PublicKey, SecretKey};
use rand::Rng;
use rand::rand_core::UnwrapErr;
use rand::rngs::SysRng;
use sha2::Sha256;
use subtle::ConstantTimeEq;

use crate::error::{Error, Result};
use crate::frame::{self, FrameType, MAX_REQUEST, MAX_RESPONSE, Opener, Sealer};
use crate::ids::{
    self, POINT_LEN, daemon_id, device_id, is_valid_secret, parse_ecdh_point, parse_point,
    point_bytes, sha256,
};
use crate::message;
use crate::wire::{Reader, lp_concat, put_lp};

/// `lp()`'d first in every transcript.
pub const PROTOCOL_LABEL: &[u8] = b"aidaemon-bridge-v1";
/// Signature prefix in session mode.
pub const SESSION_SIG_PREFIX: &[u8] = b"aidaemon-bridge-hs-v1:";
/// Signature prefix in pairing mode.
pub const PAIRING_SIG_PREFIX: &[u8] = b"aidaemon-bridge-pair-v1:";
/// HKDF info for the daemon's pairing confirm key.
pub const INFO_CONFIRM_DAEMON: &[u8] = b"aidaemon-bridge-v1 confirm daemon";
/// HKDF info for the device's pairing confirm key.
pub const INFO_CONFIRM_DEVICE: &[u8] = b"aidaemon-bridge-v1 confirm device";
/// HKDF info for the daemon → page data key.
pub const INFO_KEY_D2P: &[u8] = b"aidaemon-bridge-v1 key daemon to page";
/// HKDF info for the page → daemon data key.
pub const INFO_KEY_P2D: &[u8] = b"aidaemon-bridge-v1 key page to daemon";

/// Session handshake (SPEC §5).
pub const MODE_SESSION: u8 = 1;
/// Pairing handshake (SPEC §6).
pub const MODE_PAIRING: u8 = 2;

/// Handshake nonce length.
pub const NONCE_LEN: usize = 32;
/// `pairing_id` length.
pub const PAIRING_ID_LEN: usize = 16;
/// Longest device name, in bytes.
pub const MAX_DEVICE_NAME: usize = 60;
/// Fewest leading bytes of `device_id` the owner's app may send. The spec
/// says only "the leading bytes from a typed code"; fewer than 4 bytes (32
/// bits) would let a guessed key pass, so they are refused.
pub const MIN_DEVICE_KEY_HASH: usize = 4;

/// Which side signs: hashed as `lp("daemon")` or `lp("device")`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Daemon,
    Device,
}

impl Role {
    fn label(self) -> &'static [u8] {
        match self {
            Role::Daemon => b"daemon",
            Role::Device => b"device",
        }
    }
}

// ---------------------------------------------------------------------------
// Transcripts, signatures and key derivation (shared by both sides).

/// Session transcript `T` (SPEC §5).
pub fn session_transcript(
    identity_pub: &[u8],
    device_pub: &[u8],
    e_k: &[u8],
    e_d: &[u8],
    n_k: &[u8],
    n_d: &[u8],
) -> Vec<u8> {
    lp_concat(&[
        PROTOCOL_LABEL,
        &[MODE_SESSION],
        identity_pub,
        device_pub,
        e_k,
        e_d,
        n_k,
        n_d,
    ])
}

/// Pairing transcript `T` (SPEC §6).
#[allow(clippy::too_many_arguments)]
pub fn pairing_transcript(
    device_pub: &[u8],
    device_name: &[u8],
    e_k: &[u8],
    n_k: &[u8],
    identity_pub: &[u8],
    e_d: &[u8],
    n_d: &[u8],
    pairing_id: &[u8],
) -> Vec<u8> {
    lp_concat(&[
        PROTOCOL_LABEL,
        &[MODE_PAIRING],
        device_pub,
        device_name,
        e_k,
        n_k,
        identity_pub,
        e_d,
        n_d,
        pairing_id,
    ])
}

/// The bytes a handshake signature covers: `prefix ‖ H(lp(role) ‖ T)`.
pub fn signed_message(prefix: &[u8], role: Role, transcript: &[u8]) -> Vec<u8> {
    let mut inner = Vec::with_capacity(4 + 6 + transcript.len());
    put_lp(&mut inner, role.label());
    inner.extend_from_slice(transcript);
    let mut out = Vec::with_capacity(prefix.len() + 32);
    out.extend_from_slice(prefix);
    out.extend_from_slice(&sha256(&inner));
    out
}

/// Sign with RFC 6979 (deterministic) ECDSA P-256/SHA-256, `r ‖ s`.
pub fn sign(key: &SigningKey, message: &[u8]) -> [u8; 64] {
    let signature: Signature = key.sign(message);
    signature.to_bytes().into()
}

/// Verify a 64-byte `r ‖ s` signature (high S accepted, SPEC §1).
pub fn verify(public: &[u8], message: &[u8], signature: &[u8]) -> Result<()> {
    let key = parse_point(public).ok_or(Error::BadPoint("signing key"))?;
    let signature = Signature::from_slice(signature).map_err(|_| Error::BadSignature)?;
    key.verify(message, &signature)
        .map_err(|_| Error::BadSignature)
}

/// The 32-byte ECDH shared secret (the X coordinate).
pub fn ecdh(secret: &SecretKey, peer: &PublicKey) -> [u8; 32] {
    let shared = diffie_hellman(secret.to_nonzero_scalar(), peer.as_affine());
    (*shared.raw_secret_bytes()).into()
}

fn expand(prk: &Hkdf<Sha256>, info: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    prk.expand(info, &mut out)
        .expect("32 bytes is a valid HKDF length");
    out
}

/// `HKDF-Extract(salt, ikm)` as a 32-byte PRK.
pub fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
    let (prk, _) = Hkdf::<Sha256>::extract(Some(salt), ikm);
    prk.into()
}

/// Both directions' data keys (SPEC §7).
#[derive(Clone)]
pub struct DataKeys {
    pub d2p: [u8; 32],
    pub p2d: [u8; 32],
}

impl DataKeys {
    /// Derive from a mode's `salt = H(T)` and `ikm`.
    pub fn derive(salt: &[u8; 32], ikm: &[u8]) -> Self {
        let prk = Hkdf::<Sha256>::new(Some(salt), ikm);
        Self {
            d2p: expand(&prk, INFO_KEY_D2P),
            p2d: expand(&prk, INFO_KEY_P2D),
        }
    }

    /// The daemon's channel: seals daemon → page, opens page → daemon.
    pub fn daemon_channel(&self) -> (Sealer, Opener) {
        (Sealer::new(&self.d2p), Opener::new(&self.p2d, MAX_REQUEST))
    }

    /// The page's channel: seals page → daemon, opens daemon → page.
    pub fn page_channel(&self) -> (Sealer, Opener) {
        (Sealer::new(&self.p2d), Opener::new(&self.d2p, MAX_RESPONSE))
    }
}

/// The pairing confirm keys (SPEC §6).
#[derive(Clone)]
pub struct ConfirmKeys {
    pub daemon: [u8; 32],
    pub device: [u8; 32],
}

impl ConfirmKeys {
    /// Derive from `salt = H(T)` and `ikm = ECDH ‖ secret`.
    pub fn derive(salt: &[u8; 32], ikm: &[u8]) -> Self {
        let prk = Hkdf::<Sha256>::new(Some(salt), ikm);
        Self {
            daemon: expand(&prk, INFO_CONFIRM_DAEMON),
            device: expand(&prk, INFO_CONFIRM_DEVICE),
        }
    }
}

/// `HMAC-SHA-256(key, message)`.
pub fn hmac(key: &[u8; 32], message: &[u8]) -> [u8; 32] {
    let mut mac = <Hmac<Sha256> as KeyInit>::new_from_slice(key).expect("any key length");
    mac.update(message);
    mac.finalize().into_bytes().into()
}

fn check_mac(key: &[u8; 32], th: &[u8; 32], got: &[u8; 32]) -> Result<()> {
    let want = hmac(key, th);
    if bool::from(want.ct_eq(got)) {
        Ok(())
    } else {
        Err(Error::BadMac)
    }
}

fn pairing_ikm(shared: &[u8; 32], secret: &[u8]) -> Vec<u8> {
    let mut ikm = Vec::with_capacity(32 + secret.len());
    ikm.extend_from_slice(shared);
    ikm.extend_from_slice(secret);
    ikm
}

fn check_device_name(name: &[u8]) -> Result<&str> {
    if name.is_empty() || name.len() > MAX_DEVICE_NAME {
        return Err(Error::BadDeviceName);
    }
    std::str::from_utf8(name).map_err(|_| Error::BadDeviceName)
}

fn point(bytes: [u8; POINT_LEN], field: &'static str) -> Result<PublicKey> {
    parse_ecdh_point(&bytes).ok_or(Error::BadPoint(field))
}

fn rng() -> UnwrapErr<SysRng> {
    UnwrapErr(SysRng)
}

/// A fresh ephemeral ECDH key.
pub fn random_ephemeral() -> SecretKey {
    SecretKey::generate_from_rng(&mut rng())
}

/// `N` random bytes from the OS.
pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut out = [0u8; N];
    rng().fill_bytes(&mut out);
    out
}

// ---------------------------------------------------------------------------
// HELLO parsing (daemon).

/// A parsed HELLO; which mode the page asked for.
pub enum Hello {
    Session(SessionHello),
    Pairing(PairingHello),
}

/// Parse a HELLO frame (1 KiB limit, ABORT refused as [`Error::Aborted`]).
pub fn parse_hello(frame_bytes: &[u8]) -> Result<Hello> {
    let body = frame::decode_handshake(frame_bytes, FrameType::Hello)?;
    let mut r = Reader::new(body);
    let [mode] = r.lp_fixed::<1>("mode")?;
    match mode {
        MODE_SESSION => {
            let e_k = r.lp_fixed::<POINT_LEN>("E_k")?;
            let n_k = r.lp_fixed::<NONCE_LEN>("N_k")?;
            let device_id = r.lp_fixed::<32>("device_id")?;
            r.finish()?;
            let e_k_key = point(e_k, "E_k")?;
            Ok(Hello::Session(SessionHello {
                e_k,
                e_k_key,
                n_k,
                device_id,
            }))
        }
        MODE_PAIRING => {
            let e_k = r.lp_fixed::<POINT_LEN>("E_k")?;
            let n_k = r.lp_fixed::<NONCE_LEN>("N_k")?;
            let device_pub = r.lp_fixed::<POINT_LEN>("device_pub")?;
            let name = r.lp()?;
            r.finish()?;
            let e_k_key = point(e_k, "E_k")?;
            parse_point(&device_pub).ok_or(Error::BadPoint("device_pub"))?;
            let device_name = check_device_name(name)?.to_owned();
            Ok(Hello::Pairing(PairingHello {
                e_k,
                e_k_key,
                n_k,
                device_pub,
                device_name,
            }))
        }
        other => Err(Error::WrongMode(other)),
    }
}

/// Parse a HELLO that must be session mode (a `/v1/connect` connection).
pub fn parse_session_hello(frame_bytes: &[u8]) -> Result<SessionHello> {
    match parse_hello(frame_bytes)? {
        Hello::Session(hello) => Ok(hello),
        Hello::Pairing(_) => Err(Error::WrongMode(MODE_PAIRING)),
    }
}

/// Parse a HELLO that must be pairing mode (a joined nameplate).
pub fn parse_pairing_hello(frame_bytes: &[u8]) -> Result<PairingHello> {
    match parse_hello(frame_bytes)? {
        Hello::Pairing(hello) => Ok(hello),
        Hello::Session(_) => Err(Error::WrongMode(MODE_SESSION)),
    }
}

// ---------------------------------------------------------------------------
// Session mode, daemon side.

/// A session HELLO; look up [`device_id`](Self::device_id), then respond.
pub struct SessionHello {
    e_k: [u8; POINT_LEN],
    e_k_key: PublicKey,
    n_k: [u8; NONCE_LEN],
    device_id: [u8; 32],
}

impl SessionHello {
    /// The device the page claims to be; look it up (refuse unknown or
    /// revoked) and pass its key to [`respond`](Self::respond).
    pub fn device_id(&self) -> &[u8; 32] {
        &self.device_id
    }

    /// Answer with HELLO_REPLY. `device_pub` is the key on record for
    /// [`device_id`](Self::device_id), or `None` when the device is unknown
    /// or revoked (refused with [`Error::UnknownDevice`]).
    pub fn respond(
        self,
        identity_key: &SigningKey,
        device_pub: Option<&[u8]>,
    ) -> Result<(DaemonSession, Vec<u8>)> {
        self.respond_with(
            identity_key,
            device_pub,
            random_ephemeral(),
            random_bytes::<NONCE_LEN>(),
        )
    }

    /// [`respond`](Self::respond) with given randomness. Only for tests and
    /// test vectors: reusing an ephemeral key or nonce breaks the protocol.
    pub fn respond_with(
        self,
        identity_key: &SigningKey,
        device_pub: Option<&[u8]>,
        ephemeral: SecretKey,
        n_d: [u8; NONCE_LEN],
    ) -> Result<(DaemonSession, Vec<u8>)> {
        let device_pub = device_pub.ok_or(Error::UnknownDevice)?;
        let device_pub: [u8; POINT_LEN] = device_pub
            .try_into()
            .map_err(|_| Error::BadPoint("device_pub"))?;
        parse_point(&device_pub).ok_or(Error::BadPoint("device_pub"))?;
        if device_id(&device_pub) != self.device_id {
            return Err(Error::DeviceIdMismatch);
        }
        let identity_pub = ids::verifying_key_bytes(identity_key.verifying_key());
        let e_d = point_bytes(&ephemeral.public_key());
        let transcript =
            session_transcript(&identity_pub, &device_pub, &self.e_k, &e_d, &self.n_k, &n_d);
        let sig_d = sign(
            identity_key,
            &signed_message(SESSION_SIG_PREFIX, Role::Daemon, &transcript),
        );
        let reply =
            frame::handshake_frame(FrameType::HelloReply, &lp_concat(&[&e_d, &n_d, &sig_d]));
        let shared = ecdh(&ephemeral, &self.e_k_key);
        Ok((
            DaemonSession {
                device_pub,
                device_id: self.device_id,
                transcript,
                shared,
            },
            reply,
        ))
    }
}

/// Daemon, session mode: HELLO_REPLY sent, waiting for FINISH.
pub struct DaemonSession {
    device_pub: [u8; POINT_LEN],
    device_id: [u8; 32],
    transcript: Vec<u8>,
    shared: [u8; 32],
}

impl DaemonSession {
    /// The device this handshake is for.
    pub fn device_id(&self) -> &[u8; 32] {
        &self.device_id
    }

    /// Check FINISH (`sig_k` by the device key); then the channel is open.
    pub fn finish(self, finish_frame: &[u8]) -> Result<DaemonChannel> {
        let body = frame::decode_handshake(finish_frame, FrameType::Finish)?;
        let mut r = Reader::new(body);
        let sig_k = r.lp_fixed::<64>("sig_k")?;
        r.finish()?;
        verify(
            &self.device_pub,
            &signed_message(SESSION_SIG_PREFIX, Role::Device, &self.transcript),
            &sig_k,
        )?;
        let th = sha256(&self.transcript);
        let keys = DataKeys::derive(&th, &self.shared);
        let (sealer, opener) = keys.daemon_channel();
        Ok(DaemonChannel {
            sealer,
            opener,
            device_id: self.device_id,
            device_pub: self.device_pub,
            transcript_hash: th,
        })
    }
}

/// The exact JSON of the session `ready` control message (SPEC §5).
pub const READY_JSON: &[u8] = br#"{"type":"ready"}"#;

/// The daemon's end of an established connection.
pub struct DaemonChannel {
    /// Seals daemon → page DATA.
    pub sealer: Sealer,
    /// Opens page → daemon DATA (16 MiB reassembly).
    pub opener: Opener,
    pub device_id: [u8; 32],
    pub device_pub: [u8; POINT_LEN],
    /// `H(T)`.
    pub transcript_hash: [u8; 32],
}

impl DaemonChannel {
    /// The sealed `ready` control message. In session mode the daemon sends
    /// `promote` to the bridge and then this frame, before anything else.
    pub fn ready_frame(&mut self) -> Result<Vec<u8>> {
        self.sealer
            .seal_fragment(&message::encode_control(READY_JSON), true)
    }
}

// ---------------------------------------------------------------------------
// Pairing mode, daemon side.

/// A pairing HELLO; respond with the owner's secret and device key hash.
pub struct PairingHello {
    e_k: [u8; POINT_LEN],
    e_k_key: PublicKey,
    n_k: [u8; NONCE_LEN],
    device_pub: [u8; POINT_LEN],
    device_name: String,
}

impl PairingHello {
    /// The page's device key.
    pub fn device_pub(&self) -> &[u8; POINT_LEN] {
        &self.device_pub
    }

    /// The device's name (1–60 bytes of UTF-8).
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Answer with HELLO_REPLY. `secret` is the 13-character code the
    /// owner's app sent, `device_key_hash` the leading 4–32 bytes of
    /// `device_id` it sent; a mismatch is [`Error::DeviceKeyHashMismatch`].
    pub fn respond(
        self,
        identity_key: &SigningKey,
        secret: &[u8],
        device_key_hash: &[u8],
    ) -> Result<(DaemonPairing, Vec<u8>)> {
        self.respond_with(
            identity_key,
            secret,
            device_key_hash,
            random_ephemeral(),
            random_bytes::<NONCE_LEN>(),
            random_bytes::<PAIRING_ID_LEN>(),
        )
    }

    /// [`respond`](Self::respond) with given randomness. Only for tests and
    /// test vectors.
    pub fn respond_with(
        self,
        identity_key: &SigningKey,
        secret: &[u8],
        device_key_hash: &[u8],
        ephemeral: SecretKey,
        n_d: [u8; NONCE_LEN],
        pairing_id: [u8; PAIRING_ID_LEN],
    ) -> Result<(DaemonPairing, Vec<u8>)> {
        if !is_valid_secret(secret) {
            return Err(Error::BadSecret);
        }
        if device_key_hash.len() < MIN_DEVICE_KEY_HASH || device_key_hash.len() > 32 {
            return Err(Error::BadDeviceKeyHash);
        }
        let id = device_id(&self.device_pub);
        if !bool::from(id[..device_key_hash.len()].ct_eq(device_key_hash)) {
            return Err(Error::DeviceKeyHashMismatch);
        }
        let identity_pub = ids::verifying_key_bytes(identity_key.verifying_key());
        let e_d = point_bytes(&ephemeral.public_key());
        let transcript = pairing_transcript(
            &self.device_pub,
            self.device_name.as_bytes(),
            &self.e_k,
            &self.n_k,
            &identity_pub,
            &e_d,
            &n_d,
            &pairing_id,
        );
        let th = sha256(&transcript);
        let shared = ecdh(&ephemeral, &self.e_k_key);
        let ikm = pairing_ikm(&shared, secret);
        let confirm = ConfirmKeys::derive(&th, &ikm);
        let mac_d = hmac(&confirm.daemon, &th);
        let sig_d = sign(
            identity_key,
            &signed_message(PAIRING_SIG_PREFIX, Role::Daemon, &transcript),
        );
        let reply = frame::handshake_frame(
            FrameType::HelloReply,
            &lp_concat(&[&e_d, &n_d, &identity_pub, &pairing_id, &mac_d, &sig_d]),
        );
        Ok((
            DaemonPairing {
                device_pub: self.device_pub,
                device_id: id,
                device_name: self.device_name,
                identity_pub,
                pairing_id,
                transcript,
                th,
                confirm_device: confirm.device,
                ikm,
            },
            reply,
        ))
    }
}

/// Daemon, pairing mode: HELLO_REPLY sent, waiting for FINISH.
pub struct DaemonPairing {
    device_pub: [u8; POINT_LEN],
    device_id: [u8; 32],
    device_name: String,
    identity_pub: [u8; POINT_LEN],
    pairing_id: [u8; PAIRING_ID_LEN],
    transcript: Vec<u8>,
    th: [u8; 32],
    confirm_device: [u8; 32],
    ikm: Vec<u8>,
}

impl DaemonPairing {
    /// Check FINISH: `mac_k` (constant time), then `sig_k`. On success the
    /// caller creates the device and sends [`DaemonPaired::paired_control`].
    pub fn finish(self, finish_frame: &[u8]) -> Result<DaemonPaired> {
        let body = frame::decode_handshake(finish_frame, FrameType::Finish)?;
        let mut r = Reader::new(body);
        let mac_k = r.lp_fixed::<32>("mac_k")?;
        let sig_k = r.lp_fixed::<64>("sig_k")?;
        r.finish()?;
        check_mac(&self.confirm_device, &self.th, &mac_k)?;
        verify(
            &self.device_pub,
            &signed_message(PAIRING_SIG_PREFIX, Role::Device, &self.transcript),
            &sig_k,
        )?;
        let keys = DataKeys::derive(&self.th, &self.ikm);
        let (sealer, opener) = keys.daemon_channel();
        Ok(DaemonPaired {
            channel: DaemonChannel {
                sealer,
                opener,
                device_id: self.device_id,
                device_pub: self.device_pub,
                transcript_hash: self.th,
            },
            device_name: self.device_name,
            pairing_id: self.pairing_id,
            daemon_id: daemon_id(&self.identity_pub),
        })
    }
}

/// A completed pairing on the daemon.
pub struct DaemonPaired {
    pub channel: DaemonChannel,
    pub device_name: String,
    pub pairing_id: [u8; PAIRING_ID_LEN],
    /// `daemon_id` of the identity key that signed.
    pub daemon_id: [u8; 32],
}

impl DaemonPaired {
    /// The control message (SPEC §6, §8) to seal and send once the device
    /// is created: `{"type":"paired","daemon_id":…,"route_id":…,"device":…}`,
    /// where `device` is the id the daemon gave the device (`dv-…`), which
    /// the page names when it signs in.
    pub fn paired_control(&self, route_id: &[u8; 32], device: &str) -> Vec<u8> {
        message::encode_control(&paired_json(&self.daemon_id, route_id, device))
    }
}

/// The exact JSON of the `paired` control message: keys in this order,
/// lowercase hex, no whitespace.
pub fn paired_json(daemon_id: &[u8; 32], route_id: &[u8; 32], device: &str) -> Vec<u8> {
    debug_assert!(is_device_ref(device));
    format!(
        "{{\"type\":\"paired\",\"daemon_id\":\"{}\",\"route_id\":\"{}\",\"device\":\"{}\"}}",
        hex::encode(daemon_id),
        hex::encode(route_id),
        device
    )
    .into_bytes()
}

/// A daemon's id for a device as `paired` carries it: `dv-` and 1–40
/// lowercase hex digits.
pub fn is_device_ref(device: &str) -> bool {
    device.strip_prefix("dv-").is_some_and(|hex| {
        (1..=40).contains(&hex.len()) && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    })
}

// ---------------------------------------------------------------------------
// Page side (reference implementation; the real page is js/bridge-proto.js).

/// The page's end of an established connection.
pub struct PageChannel {
    /// Seals page → daemon DATA.
    pub sealer: Sealer,
    /// Opens daemon → page DATA (4 MiB reassembly).
    pub opener: Opener,
    /// `H(T)`.
    pub transcript_hash: [u8; 32],
}

/// Page, session mode: HELLO sent, waiting for HELLO_REPLY.
pub struct PageSession {
    device_key: SigningKey,
    device_pub: [u8; POINT_LEN],
    identity_pub: [u8; POINT_LEN],
    ephemeral: SecretKey,
    e_k: [u8; POINT_LEN],
    n_k: [u8; NONCE_LEN],
}

impl PageSession {
    /// Start a session with the pinned `identity_pub`; returns HELLO.
    pub fn start(device_key: &SigningKey, identity_pub: &[u8]) -> Result<(Self, Vec<u8>)> {
        Self::start_with(
            device_key,
            identity_pub,
            random_ephemeral(),
            random_bytes::<NONCE_LEN>(),
        )
    }

    /// [`start`](Self::start) with given randomness (tests and vectors).
    pub fn start_with(
        device_key: &SigningKey,
        identity_pub: &[u8],
        ephemeral: SecretKey,
        n_k: [u8; NONCE_LEN],
    ) -> Result<(Self, Vec<u8>)> {
        let identity_pub: [u8; POINT_LEN] = identity_pub
            .try_into()
            .map_err(|_| Error::BadPoint("identity_pub"))?;
        parse_point(&identity_pub).ok_or(Error::BadPoint("identity_pub"))?;
        let device_pub = ids::verifying_key_bytes(device_key.verifying_key());
        let e_k = point_bytes(&ephemeral.public_key());
        let hello = frame::handshake_frame(
            FrameType::Hello,
            &lp_concat(&[&[MODE_SESSION], &e_k, &n_k, &device_id(&device_pub)]),
        );
        Ok((
            Self {
                device_key: device_key.clone(),
                device_pub,
                identity_pub,
                ephemeral,
                e_k,
                n_k,
            },
            hello,
        ))
    }

    /// Check HELLO_REPLY (`sig_d` by the pinned identity key); returns the
    /// state waiting for `ready` and FINISH to send. The page sends no DATA
    /// until [`PageSessionAwaitReady::on_ready`] succeeds (SPEC §5).
    pub fn on_reply(self, reply_frame: &[u8]) -> Result<(PageSessionAwaitReady, Vec<u8>)> {
        let body = frame::decode_handshake(reply_frame, FrameType::HelloReply)?;
        let mut r = Reader::new(body);
        let e_d = r.lp_fixed::<POINT_LEN>("E_d")?;
        let n_d = r.lp_fixed::<NONCE_LEN>("N_d")?;
        let sig_d = r.lp_fixed::<64>("sig_d")?;
        r.finish()?;
        let e_d_key = point(e_d, "E_d")?;
        let transcript = session_transcript(
            &self.identity_pub,
            &self.device_pub,
            &self.e_k,
            &e_d,
            &self.n_k,
            &n_d,
        );
        verify(
            &self.identity_pub,
            &signed_message(SESSION_SIG_PREFIX, Role::Daemon, &transcript),
            &sig_d,
        )?;
        let sig_k = sign(
            &self.device_key,
            &signed_message(SESSION_SIG_PREFIX, Role::Device, &transcript),
        );
        let finish = frame::handshake_frame(FrameType::Finish, &lp_concat(&[&sig_k]));
        let th = sha256(&transcript);
        let keys = DataKeys::derive(&th, &ecdh(&self.ephemeral, &e_d_key));
        let (sealer, opener) = keys.page_channel();
        Ok((
            PageSessionAwaitReady {
                channel: PageChannel {
                    sealer,
                    opener,
                    transcript_hash: th,
                },
            },
            finish,
        ))
    }
}

/// Page, session mode: FINISH sent, waiting for the daemon's `ready`.
pub struct PageSessionAwaitReady {
    channel: PageChannel,
}

impl PageSessionAwaitReady {
    /// Open the first daemon DATA frame, which must be the complete
    /// `ready` control message (exactly [`READY_JSON`]); then the page may
    /// send DATA.
    pub fn on_ready(mut self, data_frame: &[u8]) -> Result<PageChannel> {
        let Some(message) = self.channel.opener.open(data_frame)? else {
            return Err(Error::BadMessage("ready must be one frame"));
        };
        match message::decode_from_daemon(&message)? {
            message::DaemonMessage::Control(json) if json == READY_JSON => Ok(self.channel),
            _ => Err(Error::BadMessage("expected ready control")),
        }
    }
}

/// Page, pairing mode: HELLO sent, waiting for HELLO_REPLY.
pub struct PagePairing {
    device_key: SigningKey,
    device_pub: [u8; POINT_LEN],
    device_name: String,
    secret: Vec<u8>,
    ephemeral: SecretKey,
    e_k: [u8; POINT_LEN],
    n_k: [u8; NONCE_LEN],
}

/// A pairing the page has confirmed; it still waits for the `paired`
/// control message before storing anything (SPEC §6).
pub struct PagePaired {
    pub channel: PageChannel,
    pub identity_pub: [u8; POINT_LEN],
    pub daemon_id: [u8; 32],
    pub pairing_id: [u8; PAIRING_ID_LEN],
}

impl PagePairing {
    /// Start pairing; returns HELLO.
    pub fn start(
        device_key: &SigningKey,
        device_name: &str,
        secret: &[u8],
    ) -> Result<(Self, Vec<u8>)> {
        Self::start_with(
            device_key,
            device_name,
            secret,
            random_ephemeral(),
            random_bytes::<NONCE_LEN>(),
        )
    }

    /// [`start`](Self::start) with given randomness (tests and vectors).
    pub fn start_with(
        device_key: &SigningKey,
        device_name: &str,
        secret: &[u8],
        ephemeral: SecretKey,
        n_k: [u8; NONCE_LEN],
    ) -> Result<(Self, Vec<u8>)> {
        check_device_name(device_name.as_bytes())?;
        if !is_valid_secret(secret) {
            return Err(Error::BadSecret);
        }
        let device_pub = ids::verifying_key_bytes(device_key.verifying_key());
        let e_k = point_bytes(&ephemeral.public_key());
        let hello = frame::handshake_frame(
            FrameType::Hello,
            &lp_concat(&[
                &[MODE_PAIRING],
                &e_k,
                &n_k,
                &device_pub,
                device_name.as_bytes(),
            ]),
        );
        Ok((
            Self {
                device_key: device_key.clone(),
                device_pub,
                device_name: device_name.to_owned(),
                secret: secret.to_vec(),
                ephemeral,
                e_k,
                n_k,
            },
            hello,
        ))
    }

    /// Check HELLO_REPLY (`mac_d` in constant time, then `sig_d`); returns
    /// the confirmed pairing and FINISH to send.
    pub fn on_reply(self, reply_frame: &[u8]) -> Result<(PagePaired, Vec<u8>)> {
        let body = frame::decode_handshake(reply_frame, FrameType::HelloReply)?;
        let mut r = Reader::new(body);
        let e_d = r.lp_fixed::<POINT_LEN>("E_d")?;
        let n_d = r.lp_fixed::<NONCE_LEN>("N_d")?;
        let identity_pub = r.lp_fixed::<POINT_LEN>("identity_pub")?;
        let pairing_id = r.lp_fixed::<PAIRING_ID_LEN>("pairing_id")?;
        let mac_d = r.lp_fixed::<32>("mac_d")?;
        let sig_d = r.lp_fixed::<64>("sig_d")?;
        r.finish()?;
        let e_d_key = point(e_d, "E_d")?;
        parse_point(&identity_pub).ok_or(Error::BadPoint("identity_pub"))?;
        let transcript = pairing_transcript(
            &self.device_pub,
            self.device_name.as_bytes(),
            &self.e_k,
            &self.n_k,
            &identity_pub,
            &e_d,
            &n_d,
            &pairing_id,
        );
        let th = sha256(&transcript);
        let ikm = pairing_ikm(&ecdh(&self.ephemeral, &e_d_key), &self.secret);
        let confirm = ConfirmKeys::derive(&th, &ikm);
        check_mac(&confirm.daemon, &th, &mac_d)?;
        verify(
            &identity_pub,
            &signed_message(PAIRING_SIG_PREFIX, Role::Daemon, &transcript),
            &sig_d,
        )?;
        let mac_k = hmac(&confirm.device, &th);
        let sig_k = sign(
            &self.device_key,
            &signed_message(PAIRING_SIG_PREFIX, Role::Device, &transcript),
        );
        let finish = frame::handshake_frame(FrameType::Finish, &lp_concat(&[&mac_k, &sig_k]));
        let keys = DataKeys::derive(&th, &ikm);
        let (sealer, opener) = keys.page_channel();
        Ok((
            PagePaired {
                channel: PageChannel {
                    sealer,
                    opener,
                    transcript_hash: th,
                },
                identity_pub,
                daemon_id: daemon_id(&identity_pub),
                pairing_id,
            },
            finish,
        ))
    }
}

#[cfg(test)]
mod tests;
