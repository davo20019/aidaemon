//! The boundary between a turn's kernel run and its host.
//!
//! A turn runs on the `aidaemon-core` kernel; whatever the host owns (tool
//! dispatch, the provider stack, budgets, the run ledger's closeout) the
//! kernel reaches through [`calls::HostCall`]s on the turn's channel, put
//! there by the kernel-side adapters in [`hooks`] and [`provider`]. The host
//! serves each call on the turn's own task.

use std::time::Duration;

pub mod budgets;
pub mod calls;
pub mod compaction;
pub mod hooks;
pub mod mandate_run;
pub mod outcome;
pub mod provider;
pub mod recovery;
pub mod scope;
pub mod settings;
pub mod stop;
pub mod tool_calls;
pub mod tool_exec;
pub mod tool_result_fit;
pub mod work_scope;

/// Host tools enforce their own timeouts (terminal backgrounding, CLI
/// agents); this only stops a wedged call from holding the turn forever.
pub const TOOL_TIMEOUT: Duration = Duration::from_secs(30 * 60);
