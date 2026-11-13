//! Health probes and their alert delivery live in `aidaemon_work::health`
//! (the channel hub is an `AlertDelivery`); the daemon keeps its old paths.

pub use aidaemon_work::health::*;
