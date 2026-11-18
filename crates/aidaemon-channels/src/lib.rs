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

pub mod admin_commands;
#[cfg(feature = "conformance")]
pub mod approval_conformance;
pub mod approval_render;
mod approvals;
pub mod attachments;
pub mod bot;
pub mod channel;
pub mod commands;
#[cfg(feature = "conformance")]
pub mod conformance;
pub mod context;
#[cfg(feature = "http")]
pub mod download;
pub mod hub;
mod inbound_files;
pub mod latex;
pub mod live_status;
pub mod mid_run;
pub mod names;
pub mod outbound;
pub mod owner_wait;
pub mod policy;
pub mod ports;
pub mod presentation;
pub mod prompts;
pub mod queue;
#[cfg(feature = "http")]
pub mod rate_limit;
mod render;
pub mod router;
pub mod runner;
pub mod send_file;
pub mod session;
pub mod slash;
pub mod split;
pub mod stop;
pub mod tasks;
pub mod terminal_lite;
pub mod turns;

#[cfg(feature = "testing")]
pub mod testing;

pub use approvals::{ApprovalStore, MemoryApprovalStore, StoredApproval};
pub use channel::{
    ApprovalAnswer, ApprovalChoice, ApprovalDetails, ApprovalPrompt, Attachment, AttachmentKind,
    Channel, ChannelCapabilities, ChannelError, ChannelEvent, CommandInvocation, CommandSpec,
    Conversation, ConversationKind, DEFAULT_MAX_DOWNLOAD_BYTES, InboundMessage, MessageRef,
    OutboundFile, OutboundMessage, ProgressPhase, RunEnding, RunProgress, Sender, ToolProgress,
    ToolState,
};
pub use commands::{CommandContext, CommandHandler, CommandReply, builtin_commands};
pub use names::DisplayNameCache;
pub use policy::{AccessPolicy, Admission, DEFAULT_REFUSAL, SenderRole};
pub use router::EventRouter;
pub use runner::{
    APPROVE_CHOICE, AttachmentConfig, ChannelRunner, ChannelRunnerBuilder, DENY_CHOICE,
    RunnerBuildError, RunnerConfig,
};
pub use session::{MemorySessionIndex, SessionIndex, session_id};
pub use split::split_message;
