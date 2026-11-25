//! # aidaemon-runtime-core
//!
//! The core of aidaemon's turn runtime: what runs one agent turn on top of
//! the `aidaemon-core` kernel, with aidaemon's behavior as the default. The
//! built-in tools and the agent are `aidaemon-runtime`, which re-exports
//! every module here at its root; depend on that crate.
//!
//! - [`llm`]: the model runtime a host builds from its providers (primary,
//!   router, failover chain) and the side-call gateway, the one door for
//!   model calls made outside the agent loop ([`llm::side_calls`],
//!   [`llm::gateway`]), with the optional decision backend
//!   ([`llm::decide`]).
//! - [`engine`]: the boundary between a turn's kernel run and its host:
//!   the host calls, the forwarding hooks and the host provider.
//! - [`eval`]: the per-task harness effectiveness evaluation.
//! - [`background_task_lead`]: a goal's task lead run in the background,
//!   its executors, recovery and result delivery.
//! - [`goal_dispatch`]: wait prefixes, scheduled-run provenance and state,
//!   and the user-facing goal and task result summaries.
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
//! - [`run_state`]: one typed record per background helper run (running,
//!   succeeded, failed or stopped; measured times; evidence) that every
//!   surface showing a helper renders from, and whose end pushes its result.
//! - [`spawn`]: the running sub-agents registry, a child's progress and
//!   end as its parent sees them, and the plain child's prompt.
//! - [`turn_runtime`]: one agent's turn settings and caches.
//! - [`specialists`]: the specialist registry (bundled and owner-overridden
//!   expert profiles a child is made from).
//! - [`spawn_tool`]: the `spawn_agent` tool.
//! - [`untrusted`]: the untrusted-content rule's taint facts and verifier.

// Feature unification only (cargo-hakari); keeps the unused-dependency lint quiet.
use workspace_hack as _;

pub mod approvals;
pub mod authority;
pub mod background_run;
pub mod background_task_lead;
pub mod budget_policy;
pub mod call_args;
pub mod chat_bridge;
pub mod consolidation;
pub mod context_budget;
pub mod coordination;
pub mod decisions;
pub mod engine;
pub mod eval;
pub mod goal_dispatch;
pub mod heartbeat;
pub mod helper_run;
pub mod history;
pub mod host;
pub mod known_resources;
pub mod llm;
pub mod markers;
pub mod media;
pub mod memory;
pub mod memory_port;
pub mod message_content;
pub mod mid_run;
pub mod owed_reports;
pub mod owner_level;
pub mod permissions;
pub mod policy;
pub mod prompt;
pub mod render;
pub mod run_state;
pub mod session_model;
pub mod settings;
pub mod spawn;
pub mod spawn_tool;
pub mod specialists;
pub mod state;
pub mod steering;
pub mod stores;
pub mod terminal_grants;
pub mod text_cut;
pub mod tool_manifest;
pub mod turn;
pub mod turn_runtime;
pub mod untrusted;
pub mod untrusted_writes;
