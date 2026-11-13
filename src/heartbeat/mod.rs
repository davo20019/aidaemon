//! The daemon's background work lives in `aidaemon_runtime::heartbeat`
//! (the coordinator) and `aidaemon_work` (the goal cycle, telemetry and
//! health gate); the daemon keeps its old paths.

pub use aidaemon_runtime::heartbeat::*;
#[allow(unused_imports)]
pub(crate) use aidaemon_work::scheduler::*;

#[allow(unused_imports)]
pub use aidaemon_work::heartbeat::{DbHealthGate, HeartbeatTelemetry, JobObserver};

#[cfg(test)]
mod tests;
