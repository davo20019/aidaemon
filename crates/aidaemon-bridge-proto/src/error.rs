//! One error type for everything a peer can make us refuse.
//!
//! Every variant ends the connection (SPEC §4: "close the connection. Never
//! retry or skip."); the variant only says why, for logs and tests.

use std::fmt;

/// Why a frame, handshake or message was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The frame is shorter than its 11-byte header.
    Truncated,
    /// The frame's version byte is not 1.
    Version(u8),
    /// The frame's type byte is not one the spec lists.
    UnknownType(u8),
    /// Reserved flag bits were set, or a handshake/ABORT frame had flags.
    Flags(u8),
    /// A handshake or ABORT frame had a nonzero counter.
    HandshakeCounter(u64),
    /// A DATA frame's counter was not the next one.
    Counter { expected: u64, got: u64 },
    /// This side's counter would wrap; the connection must end.
    CounterExhausted,
    /// A frame was larger than the limit in force (SPEC §7).
    FrameTooLarge { len: usize, max: usize },
    /// A reassembled message exceeded its limit (SPEC §7).
    MessageTooLarge { max: usize },
    /// A frame of a type not expected in the current state (including any
    /// handshake frame after the handshake, or DATA during it).
    UnexpectedFrame(u8),
    /// The peer sent ABORT.
    Aborted,
    /// A body or message ended before a field it declares.
    Short,
    /// Bytes followed the last field.
    TrailingBytes,
    /// A fixed-size field had the wrong length.
    FieldLength { field: &'static str, len: usize },
    /// The handshake mode was not the one this side expects.
    WrongMode(u8),
    /// A point was not a 65-byte uncompressed P-256 point on the curve.
    BadPoint(&'static str),
    /// AEAD authentication failed.
    Decrypt,
    /// The device is unknown or revoked (session mode).
    UnknownDevice,
    /// The device key the lookup returned does not hash to the device id.
    DeviceIdMismatch,
    /// The device key does not match the hash the owner's app sent (pairing).
    DeviceKeyHashMismatch,
    /// The expected device key hash is not 4–32 bytes.
    BadDeviceKeyHash,
    /// The device name is not 1–60 bytes of UTF-8.
    BadDeviceName,
    /// The pairing secret is not 13 Crockford Base32 characters.
    BadSecret,
    /// A pairing MAC did not match.
    BadMac,
    /// A handshake signature did not verify.
    BadSignature,
    /// A message (SPEC §8) broke a rule; the string names it.
    BadMessage(&'static str),
    /// A header not in the allowlist for its direction.
    HeaderNotAllowed(String),
    /// The sealer/opener already failed; the connection is over.
    Closed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated => write!(f, "frame shorter than its header"),
            Error::Version(v) => write!(f, "unsupported frame version {v}"),
            Error::UnknownType(t) => write!(f, "unknown frame type {t:#04x}"),
            Error::Flags(b) => write!(f, "reserved frame flags set ({b:#04x})"),
            Error::HandshakeCounter(c) => write!(f, "handshake frame with counter {c}"),
            Error::Counter { expected, got } => {
                write!(f, "frame counter {got}, expected {expected}")
            }
            Error::CounterExhausted => write!(f, "frame counter exhausted"),
            Error::FrameTooLarge { len, max } => write!(f, "frame of {len} bytes over {max}"),
            Error::MessageTooLarge { max } => write!(f, "message over {max} bytes"),
            Error::UnexpectedFrame(t) => write!(f, "unexpected frame type {t:#04x}"),
            Error::Aborted => write!(f, "peer aborted"),
            Error::Short => write!(f, "field runs past the end"),
            Error::TrailingBytes => write!(f, "trailing bytes"),
            Error::FieldLength { field, len } => write!(f, "{field} has wrong length {len}"),
            Error::WrongMode(m) => write!(f, "wrong handshake mode {m}"),
            Error::BadPoint(field) => write!(f, "{field} is not an uncompressed P-256 point"),
            Error::Decrypt => write!(f, "decryption failed"),
            Error::UnknownDevice => write!(f, "unknown or revoked device"),
            Error::DeviceIdMismatch => write!(f, "device key does not match device id"),
            Error::DeviceKeyHashMismatch => write!(f, "device key does not match pairing code"),
            Error::BadDeviceKeyHash => write!(f, "device key hash must be 4-32 bytes"),
            Error::BadDeviceName => write!(f, "device name must be 1-60 bytes of UTF-8"),
            Error::BadSecret => write!(f, "pairing secret must be 13 Crockford Base32 characters"),
            Error::BadMac => write!(f, "pairing MAC mismatch"),
            Error::BadSignature => write!(f, "handshake signature invalid"),
            Error::BadMessage(why) => write!(f, "bad message: {why}"),
            Error::HeaderNotAllowed(name) => write!(f, "header not allowed: {name}"),
            Error::Closed => write!(f, "connection already failed"),
        }
    }
}

impl std::error::Error for Error {}

/// Result with this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
