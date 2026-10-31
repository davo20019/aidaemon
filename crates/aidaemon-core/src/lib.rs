//! # aidaemon-core (experimental sketch)
//!
//! A small agent-loop kernel. It owns the loop, the tool contract, typed
//! outcomes, receipts, budgets, stuck detection, and interrupt/resume. Every
//! *policy* (approvals, completion checks, context strategies, memory) plugs
//! in through [`Hook`]. Product features (channels, goals, scheduling,
//! specialists) live outside this crate.
//!
//! The loop, in one sentence: build the request, call the model, and if it
//! asked for tools run them (read-only ones concurrently, mutating ones one
//! at a time) and go around again; otherwise ask the stop hooks whether the
//! answer is acceptable and finish with a typed [`RunOutcome`].
//!
//! See `DESIGN.md` next to this crate for the reasoning behind each choice,
//! and `examples/quickstart.rs` for a runnable example.

mod agent;
mod error;
mod event;
mod hook;
mod limits;
mod message;
mod outcome;
mod provider;
mod receipt;
mod run_loop;
mod state;
mod store;
mod stuck;
mod tool;

#[cfg(feature = "testing")]
pub mod testing;

pub use agent::{Agent, AgentBuilder, Resolution};
pub use error::{BuildError, CoreError};
pub use event::{Event, EventSink};
pub use hook::{Hook, HookContext, StopDecision, StuckDecision, ToolDecision};
pub use limits::{LimitKind, Limits, RunSnapshot, StopCondition, HARD_TURN_CAP};
pub use message::{Message, Role, ToolCall};
pub use outcome::{RunOutcome, RunStatus};
pub use provider::{
    ModelRequest, ModelResponse, Provider, ProviderError, Retry, ToolChoice, ToolSpec, Usage,
};
pub use receipt::{InvocationStage, Receipt, ReceiptStatus};
pub use state::{Interruption, Pending, RunId, RunState};
pub use store::{MemoryStore, RunStore, StoreError};
pub use stuck::{StuckConfig, StuckSignal};
pub use tool::{FnTool, Tool, ToolAnnotations, ToolContext, ToolError, ToolOutput, ToolResult};

/// Re-exported so tools can observe cancellation without adding the dependency.
pub use tokio_util::sync::CancellationToken;
