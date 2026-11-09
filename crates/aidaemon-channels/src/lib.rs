//! # aidaemon-channels (experimental)
//!
//! Chat channels (Telegram, Slack, Discord, or your own) connected to
//! `aidaemon_core` agents. This crate defines the [`Channel`] contract and
//! the runner that turns channel events into agent runs: one session per
//! conversation, one run at a time per session, streamed progress, and
//! approvals as buttons that resume interrupted runs.
//!
//! Adapters live in their own crates (`aidaemon-channel-telegram`,
//! `aidaemon-channel-slack`, `aidaemon-channel-discord`) and depend only on
//! the types in [`channel`].
//!
//! The runner is [`ChannelRunner`]; see [`runner`] for how to wire an agent,
//! an [`EventRouter`] and channels together.

mod approvals;
pub mod channel;
#[cfg(feature = "conformance")]
pub mod conformance;
pub mod latex;
pub mod policy;
mod render;
pub mod router;
pub mod runner;
pub mod session;
pub mod split;

#[cfg(feature = "testing")]
pub mod testing;

pub use approvals::{ApprovalStore, MemoryApprovalStore, StoredApproval};
pub use channel::{
    ApprovalAnswer, ApprovalChoice, ApprovalPrompt, Attachment, Channel, ChannelCapabilities,
    ChannelError, ChannelEvent, CommandInvocation, CommandSpec, Conversation, ConversationKind,
    InboundMessage, MessageRef, OutboundFile, OutboundMessage, Sender,
};
pub use policy::{AccessPolicy, Admission, SenderRole, DEFAULT_REFUSAL};
pub use router::EventRouter;
pub use runner::{
    ChannelRunner, ChannelRunnerBuilder, RunnerBuildError, RunnerConfig, APPROVE_CHOICE,
    DENY_CHOICE,
};
pub use session::{session_id, MemorySessionIndex, SessionIndex};
pub use split::split_message;
