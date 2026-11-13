//! # aidaemon-runtime
//!
//! aidaemon's turn runtime: what runs one agent turn on top of the
//! `aidaemon-core` kernel, with aidaemon's behavior as the default.
//!
//! - [`llm`]: the model runtime a host builds from its providers (primary,
//!   router, failover chain) and the side-call gateway, the one door for
//!   model calls made outside the agent loop ([`llm::side_calls`],
//!   [`llm::gateway`]), with the optional decision backend
//!   ([`llm::decide`]).
//! - [`engine`]: the boundary between a turn's kernel run and its host:
//!   the host calls, the forwarding hooks and the host provider.
//! - [`chat_bridge`]: calling an SDK provider with chat-completions JSON,
//!   and the classified provider errors recovery decisions read.
//! - [`prompt`]: the system prompt's cacheable core, its fixed rule blocks
//!   and the prefix fingerprints.
//! - [`policy`]: execution policies, turn signals, recall guardrails and
//!   model trust tiers.
//! - [`markers`]: marker constants shared by prompts and sanitizers.
//! - [`media`]: received images and audio as content blocks, and the
//!   speech-to-text fallback.
//! - [`message_content`]: chat content helpers.
//! - [`context_budget`]: token estimates and context-window budgets.
//! - [`turn`]: system directives, tool-result notices and task outcomes.
//! - [`settings`]: what the runtime takes from its host's configuration.
//! - [`approvals`]: owner approvals on the kernel: the approval rules over
//!   the turn's tools, the owner's session grants and the card resolver.
//! - [`authority`]: tool authority resolved once per turn, session modes and
//!   what a spawned child inherits.
//! - [`coordination`]: what one request's tree of sub-agents knows about each
//!   other, and the hook that judges their file writes.
//! - [`known_resources`]: the context that crosses a handoff.
//! - [`spawn`]: the running sub-agents registry, a child's progress and
//!   end as its parent sees them, and the plain child's prompt.
//! - [`untrusted`]: the untrusted-content rule's taint facts and verifier.

pub mod approvals;
pub mod authority;
pub mod chat_bridge;
pub mod context_budget;
pub mod coordination;
pub mod engine;
pub mod host;
pub mod known_resources;
pub mod llm;
pub mod markers;
pub mod media;
pub mod message_content;
pub mod mid_run;
pub mod policy;
pub mod prompt;
pub mod settings;
pub mod spawn;
pub mod turn;
pub mod untrusted;
