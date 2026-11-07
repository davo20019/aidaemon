//! # aidaemon-policies (experimental)
//!
//! The daemon's policies re-implemented as [`aidaemon_core::Hook`]s and an
//! [`aidaemon_core::EventSink`], using only the kernel's public API.
//!
//! | Policy | Hook point | What it decides from |
//! |---|---|---|
//! | [`ApprovalPolicy`] | `before_tool` | annotations, tool names, tokenized commands, [`ApprovalRules`], [`ApprovalGrants`] |
//! | [`EgressGuard`] | `before_tool` | secret shapes and literal secrets in outbound arguments |
//! | [`ScopeGuard`] | `before_tool` | the call's access manifest |
//! | [`ReceiptCompletionCheck`] | `on_stop` | receipts since the last user message |
//! | [`Redactor`] | `after_tool` | secret shapes and literal secrets in output |
//! | [`UntrustedOutput`] | `after_tool` | tool name / prefix configuration |
//! | [`ProjectInstructions`] | `before_model` | `AGENTS.md` hierarchy on disk |
//! | [`ContextWindow`] | `before_model` | estimated request size |
//! | [`MemoryInjection`] | `before_model` | a [`Recall`] backend |
//! | [`TracingSink`] | `EventSink` | the event stream |
//!
//! Every decision comes from typed state (annotations, access manifests,
//! receipts, statuses, message origins) or from structural parsing, never
//! from matching words in model prose.
//!
//! Order matters: hooks run in registration order. Register request
//! injectors (`ProjectInstructions`, `MemoryInjection`) before
//! `ContextWindow` so the size estimate includes what they add, and
//! `Redactor` before `UntrustedOutput` so redaction sees raw output.
//! Before tools run, the order is pinned: authority and scope, then
//! [`EgressGuard`], then [`aidaemon_core::UntrustedHook`], then
//! [`ApprovalPolicy`] last, so the owner is only asked about calls every
//! other hook allows. (The kernel refuses a call the owner already declined
//! in the run before any hook sees it.)

mod approval;
mod completion;
mod context_window;
mod conversation;
mod egress;
mod grants;
mod instructions;
mod loop_guard;
mod memory;
mod redact;
mod scope;
mod shell;
mod tracing_sink;
mod untrusted;

pub use approval::{ApprovalGrants, ApprovalMode, ApprovalPolicy, ApprovalRules, RuleDecision};
pub use completion::{ReceiptCompletionCheck, Requirement};
pub use context_window::{estimate_tokens, ContextWindow};
pub use conversation::{
    compaction_threshold, elide_middle, render_summary_note, turns_to_summarize, verbatim_tokens,
    wait_threshold, CompactionJobs, CompactionLimits, Compactor, ConversationCompaction,
    ConversationSource, ConversationSummary, ConversationTurn, InMemorySummaries, PendingTurns,
    ProviderSummarizer, Summarizer, SummaryNote, SummaryStore, COMPACTED_STATE_HEADINGS,
};
pub use egress::EgressGuard;
pub use grants::SessionGrants;
pub use instructions::{InstructionSource, ProjectInstructions};
pub use loop_guard::{same_tool_loop, SameToolStreakGuard, StreakAllowance, StreakLimits};
pub use memory::{InMemoryRecall, MemoryInjection, Recall};
pub use redact::{secret_patterns, Redactor};
pub use scope::{ScopeGuard, UndeclaredAccess};
pub use shell::{catastrophic_reason, command_segments};
pub use tracing_sink::TracingSink;
pub use untrusted::{wrap_untrusted, UntrustedOutput, UNTRUSTED_PREFIX, UNTRUSTED_SUFFIX};

use aidaemon_core::{Message, Role};

/// True for a user message the harness or a hook injected rather than the
/// user. Reads the typed [`aidaemon_core::MessageOrigin`], not the text.
pub fn is_harness_message(message: &Message) -> bool {
    message.role == Role::User && message.is_injected()
}

/// Index of the last user message that the user (not the harness) wrote.
pub fn last_user_message_index(messages: &[Message]) -> Option<usize> {
    messages
        .iter()
        .rposition(|m| m.role == Role::User && !is_harness_message(m))
}

/// Insert `message` after the leading run of system messages.
pub(crate) fn insert_after_system(messages: &mut Vec<Message>, message: Message) {
    let at = messages
        .iter()
        .position(|m| m.role != Role::System)
        .unwrap_or(messages.len());
    messages.insert(at, message);
}
