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
//! The terminal UI and the desktop app link it; the web client keeps a
//! TypeScript reducer held to the same behaviour by shared fixtures.

pub mod keystore;
pub mod link;
pub mod signin;
pub mod state;

pub use keystore::{FileKeyStore, KeyStore, StoredDevice};
pub use link::{Auth, ClientIdentity, Disconnect, DropKind, LinkConfig, Update};
pub use signin::DeviceKey;
pub use state::State;
