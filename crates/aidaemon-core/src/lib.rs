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
mod binding;
pub mod card;
mod endpoint;
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
pub mod redaction;
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
    AUTHORITY_BLOCKED, ApprovalCard, ApprovalRisk, AskReason, CallEffects, CardAction,
    EGRESS_BLOCKED, EffectClass, EffectKey, HighImpact, InterruptKind, OWNER_DECLINED,
    OWNER_UNANSWERED, OwnerApprovalRequest, SharedDeclines, UNTRUSTED_BLOCKED,
};
pub use audience::{ChannelVisibility, UserRole};
pub use authority::{AccessLevel, Authority, AuthorityCeiling, AuthorityHook};
pub use endpoint::{EndpointPublisher, EndpointSource};
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
pub use limits::{HARD_TURN_CAP, LimitKind, Limits, RunSnapshot, StopCondition};
pub use message::{Audio, Image, Message, MessageOrigin, Role, ToolCall};
pub use outcome::{CallResult, FinishOutcome, RunOutcome, RunStatus};
pub use provider::{
    ModelRequest, ModelResponse, Provider, ProviderError, RequestOptions, ResponseFormat, Retry,
    StreamEvent, ToolChoice, ToolSpec, Usage, collect_stream,
};
pub use quota::{Cooldown, QuotaGuard};
pub use receipt::{InvocationStage, Receipt, ReceiptStatus};
pub use safeguard::Safeguard;
pub use service::{
    FnService, Schedule, Service, ServiceContext, ServiceError, ServiceEvent, ServiceGate,
    ServiceHost, ServiceObserver, ServiceRunEnd, ServiceWaker,
};
/// Kernel bookkeeping types inside [`RunState`]; not part of the stable API.
#[doc(hidden)]
pub use state::{InFlight, PendingFinish};
pub use state::{Interruption, Pending, RUN_STATE_SCHEMA, RunId, RunState};
pub use store::{MemoryStore, RunStore, StoreError};
pub use stuck::{StuckConfig, StuckSignal};
pub use tool::{
    AccessManifest, Extensions, Finish, FnTool, Outbound, SameToolStreak, SourceTrust, Tool,
    ToolAnnotations, ToolContext, ToolError, ToolInfo, ToolOutput, ToolResult,
    check_argument_types, check_arguments, strip_reserved_arguments,
};

pub use untrusted::{
    UNTRUSTED_EGRESS_INTERRUPT, UntrustedHook, UntrustedMode, is_untrusted_egress,
};

/// Re-exported so tools can observe cancellation without adding the dependency.
pub use tokio_util::sync::CancellationToken;
