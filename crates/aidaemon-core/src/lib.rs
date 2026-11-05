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
mod approval;
mod authority;
mod error;
mod event;
mod failover;
mod hook;
mod limits;
mod message;
mod outcome;
mod provider;
mod receipt;
mod run_loop;
mod safeguard;
mod state;
mod store;
mod stuck;
mod tool;
mod untrusted;

#[cfg(feature = "testing")]
pub mod conformance;
#[cfg(feature = "testing")]
pub mod testing;

pub use agent::{Agent, AgentBuilder, InterruptResolver, Resolution};
pub use approval::{
    ApprovalRisk, CallEffects, EffectClass, EffectKey, InterruptKind, OwnerApprovalRequest,
    EGRESS_BLOCKED, OWNER_DECLINED,
};
pub use authority::{AccessLevel, Authority, AuthorityCeiling, AuthorityHook};
pub use error::{BuildError, CoreError};
pub use event::{Event, EventSink};
pub use failover::{
    DefaultFailover, Failover, FailoverAction, FailoverPolicy, FailoverStep, ProviderSwitch,
    SwitchReason,
};
pub use hook::{Hook, HookContext, StopDecision, StuckDecision, ToolDecision};
pub use limits::{LimitKind, Limits, RunSnapshot, StopCondition, HARD_TURN_CAP};
pub use message::{Audio, Image, Message, MessageOrigin, Role, ToolCall};
pub use outcome::{FinishOutcome, RunOutcome, RunStatus};
pub use provider::{
    collect_stream, ModelRequest, ModelResponse, Provider, ProviderError, RequestOptions,
    ResponseFormat, Retry, StreamEvent, ToolChoice, ToolSpec, Usage,
};
pub use receipt::{InvocationStage, Receipt, ReceiptStatus};
pub use safeguard::Safeguard;
pub use state::{InFlight, Interruption, Pending, RunId, RunState, RUN_STATE_SCHEMA};
pub use store::{MemoryStore, RunStore, StoreError};
pub use stuck::{StuckConfig, StuckSignal};
pub use tool::{
    AccessManifest, Extensions, Finish, FnTool, SourceTrust, Tool, ToolAnnotations, ToolContext,
    ToolError, ToolInfo, ToolOutput, ToolResult,
};

pub use untrusted::{
    is_untrusted_egress, UntrustedHook, UntrustedMode, UNTRUSTED_EGRESS_INTERRUPT,
};

/// Re-exported so tools can observe cancellation without adding the dependency.
pub use tokio_util::sync::CancellationToken;
