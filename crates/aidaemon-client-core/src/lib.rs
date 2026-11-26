//! # aidaemon-client-core
//!
//! What every AIDaemon client shares, with no rendering: the connection to
//! the Client Gateway ([`link`]: hello, resume from the last `seq`, pings,
//! reconnect with backoff, unanswered commands resent under their own
//! `request_id`), the event reducer ([`state`]: the client's view built only
//! from server frames), and sign-in ([`signin`]: the owner's local token, or
//! a paired device key that trades a signed challenge for a short-lived
//! grant, kept in a [`keystore::KeyStore`]).
//!
//! [`client`] is the whole client without rendering: the reducer, the
//! connection's state, the reads asked for and their answers, turns'
//! activity, uploads and the other state a UI must not own.
//!
//! The terminal UI and the nsh app (`clients/app`, desktop, web and terminal)
//! link it.
//!
//! With default features off, what is left has no I/O and builds for
//! `wasm32-unknown-unknown`: the reducer, the link's rules ([`machine`]),
//! device keys, and sign-in over any [`signin::Http`]. Feature `native`
//! (on by default) adds the tokio link and reqwest; feature `dial` alone
//! dials a daemon by key over iroh ([`remote`]) and also builds there.

#[cfg(feature = "remote")]
pub mod bridge;
pub mod client;
pub mod keystore;
#[cfg(feature = "native")]
pub mod link;
#[cfg(feature = "native")]
pub mod local;
pub mod machine;
/// A conversation's participants, derived the same way everywhere.
pub use aidaemon_client_protocol::participants;
#[cfg(feature = "dial")]
pub mod remote;
#[cfg(feature = "native")]
pub mod screen_share;
pub mod shells;
pub mod signin;
pub mod state;
pub mod surfaces;
pub mod text;
#[cfg(feature = "native")]
mod ticketed;
#[cfg(feature = "native")]
mod tls;
pub mod tools;

pub use client::{
    Client, ComposerTarget, ComposerTargetView, Connection, Context, Effect, Felt, Read,
};
#[cfg(target_os = "ios")]
pub use keystore::KeychainKeyStore;
pub use keystore::{FileKeyStore, KeyStore, StoredDevice};
#[cfg(feature = "native")]
pub use link::{Auth, LinkConfig};
pub use machine::{ClientIdentity, Disconnect, DropKind, LinkMachine, Update};
pub use signin::DeviceKey;
pub use state::State;
