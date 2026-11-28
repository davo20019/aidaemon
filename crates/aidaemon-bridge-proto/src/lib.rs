//! The aidaemon bridge protocol, v1: see `SPEC.md` in this crate.
//!
//! A browser that cannot run iroh reaches its owner's daemon through a
//! public relay that only forwards bytes. Everything a peer can hand us is
//! untrusted: every parser here refuses what the spec does not allow.
//!
//! - [`frame`]: frames, DATA [`frame::Sealer`]/[`frame::Opener`] with
//!   counters, fragmentation and reassembly limits.
//! - [`handshake`]: session and pairing handshakes for the daemon (and the
//!   page, as a reference; the real page is `js/bridge-proto.js`).
//! - [`message`]: request, response and control messages inside DATA.
//! - [`ids`], [`login`]: identifiers, point parsing, the bridge login.

pub mod error;
pub mod frame;
pub mod handshake;
pub mod ids;
pub mod login;
pub mod message;
mod wire;

pub use error::{Error, Result};
