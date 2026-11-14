//! # aidaemon-client-core
//!
//! What every AIdaemon client shares, with no rendering: the connection to
//! the Client Gateway ([`link`]: hello, resume from the last `seq`, pings,
//! reconnect with backoff, unanswered commands resent under their own
//! `request_id`), the event reducer ([`state`]: the client's view built only
//! from server frames), and sign-in ([`signin`]: the owner's local token, or
//! a paired device key that trades a signed challenge for a short-lived
//! grant, kept in a [`keystore::KeyStore`]).
//!
//! The terminal UI and the nsh app (`clients/app`) link it; the web client keeps a
//! TypeScript reducer held to the same behaviour by shared fixtures.
//!
//! With default features off, what is left has no I/O and builds for
//! `wasm32-unknown-unknown`: the reducer, the link's rules ([`machine`]),
//! device keys, and sign-in over any [`signin::Http`]. Feature `native`
//! (on by default) adds the tokio link and reqwest.

pub mod keystore;
#[cfg(feature = "native")]
pub mod link;
pub mod machine;
pub mod signin;
pub mod state;

pub use keystore::{FileKeyStore, KeyStore, StoredDevice};
#[cfg(feature = "native")]
pub use link::{Auth, LinkConfig};
pub use machine::{ClientIdentity, Disconnect, DropKind, LinkMachine, Update};
pub use signin::DeviceKey;
pub use state::State;
