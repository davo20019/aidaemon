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
//! See the SDK guide at <https://docs.aidaemon.ai/sdk> for the concepts,
//! `examples/kernel_tour.rs` for a runnable tour of the kernel, and
//! `aidaemon-sdk`'s `examples/quickstart.rs` for the shortest working agent.

mod agent;
mod approval;
mod audience;
mod authority;
mod error;
mod event;
mod failover;
mod hook;
mod limits;
pub mod mandate_card;
mod message;
mod outcome;
mod provider;
mod quota;
mod receipt;
mod run_loop;
mod safeguard;
mod service;
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
    ApprovalCard, ApprovalRisk, AskReason, CallEffects, CardAction, EffectClass, EffectKey,
    HighImpact, InterruptKind, OwnerApprovalRequest, SharedDeclines, AUTHORITY_BLOCKED,
    EGRESS_BLOCKED, OWNER_DECLINED, OWNER_UNANSWERED, UNTRUSTED_BLOCKED,
};
pub use audience::{ChannelVisibility, UserRole};
pub use authority::{AccessLevel, Authority, AuthorityCeiling, AuthorityHook};
pub use error::{BuildError, CoreError};
pub use event::{Event, EventSink};
pub use failover::{
    DefaultFailover, Failover, FailoverAction, FailoverPolicy, FailoverStep, ProviderSwitch,
    SwitchReason,
};
pub use hook::{
    Hook, HookContext, HookFailure, HookPoint, LimitDecision, ModelDecision, ModelErrorDecision,
    StopDecision, StuckDecision, ToolDecision,
};
#[doc(hidden)]
pub use limits::LimitGrants;
pub use limits::{LimitKind, Limits, RunSnapshot, StopCondition, HARD_TURN_CAP};
pub use message::{Audio, Image, Message, MessageOrigin, Role, ToolCall};
pub use outcome::{FinishOutcome, RunOutcome, RunStatus};
pub use provider::{
    collect_stream, ModelRequest, ModelResponse, Provider, ProviderError, RequestOptions,
    ResponseFormat, Retry, StreamEvent, ToolChoice, ToolSpec, Usage,
};
pub use quota::{Cooldown, QuotaGuard};
pub use receipt::{InvocationStage, Receipt, ReceiptStatus};
pub use safeguard::Safeguard;
pub use service::{
    FnService, Schedule, Service, ServiceContext, ServiceError, ServiceEvent, ServiceGate,
    ServiceHost, ServiceObserver, ServiceWaker,
};
/// Kernel bookkeeping types inside [`RunState`]; not part of the stable API.
#[doc(hidden)]
pub use state::{InFlight, PendingFinish};
pub use state::{Interruption, Pending, RunId, RunState, RUN_STATE_SCHEMA};
pub use store::{MemoryStore, RunStore, StoreError};
pub use stuck::{StuckConfig, StuckSignal};
pub use tool::{
    AccessManifest, Extensions, Finish, FnTool, Outbound, SameToolStreak, SourceTrust, Tool,
    ToolAnnotations, ToolContext, ToolError, ToolInfo, ToolOutput, ToolResult,
};

pub use untrusted::{
    is_untrusted_egress, UntrustedHook, UntrustedMode, UNTRUSTED_EGRESS_INTERRUPT,
};

/// Re-exported so tools can observe cancellation without adding the dependency.
pub use tokio_util::sync::CancellationToken;
