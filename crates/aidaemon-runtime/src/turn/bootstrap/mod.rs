//! The start of a turn: the stop command, resume, the turn's context,
//! digest and task relationship, its contract, the tool roster and model,
//! and the system prompt, before the agent loop runs.

mod run;
mod shortcuts;
mod types;

pub use run::{run_bootstrap, GateFilterStage};
pub use types::{BootstrapCtx, BootstrapData, BootstrapOutcome};
