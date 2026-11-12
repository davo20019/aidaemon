//! The boundary between a turn's kernel run and its host.
//!
//! A turn runs on the `aidaemon-core` kernel; whatever the host owns (tool
//! dispatch, the provider stack, budgets, the run ledger's closeout) the
//! kernel reaches through [`calls::HostCall`]s on the turn's channel, put
//! there by the kernel-side adapters in [`hooks`] and [`provider`]. The host
//! serves each call on the turn's own task.

use std::time::Duration;

pub mod calls;
pub mod hooks;
pub mod provider;

/// Host tools enforce their own timeouts (terminal backgrounding, CLI
/// agents); this only stops a wedged call from holding the turn forever.
pub const TOOL_TIMEOUT: Duration = Duration::from_secs(30 * 60);
