//! First-class AIdaemon Node domain and protocol.
//!
//! A Node is a persistent participant with server-issued identity, observed
//! hardware capabilities, explicit authorization, policy, and one or more
//! conversational Channels. Transport adapters only expose the narrow Node
//! Gateway; they never establish domain authority.

pub mod announcement;
pub mod auth;
pub mod channel;
pub mod cli;
pub mod domain;
pub mod gateway;
pub mod monitoring;
pub mod ota;
pub mod protocol;
pub mod service;
pub mod simulator;
pub mod speech;
pub mod store;
pub mod tool;

pub use domain::{AuthenticatedNodeContext, NodeAction, NodeRecord};
pub use service::{NodeConversationIngress, NodeService};
pub use store::NodeStore;
