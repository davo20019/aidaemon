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
//! - [`chat_bridge`]: calling an SDK provider with chat-completions JSON,
//!   and the classified provider errors recovery decisions read.
//! - [`context_budget`]: token estimates and context-window budgets.
//! - [`settings`]: what the runtime takes from its host's configuration.

pub mod chat_bridge;
pub mod context_budget;
pub mod llm;
pub mod settings;
