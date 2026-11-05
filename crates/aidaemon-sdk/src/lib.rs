//! # aidaemon-sdk (experimental)
//!
//! One dependency for building agents on the aidaemon kernel. The kernel
//! ([`aidaemon_core`], re-exported at the crate root) owns the loop, typed
//! outcomes, receipts, limits, events and interrupt/resume. Everything else
//! is an optional module, enabled by a feature of the same name:
//!
//! | Module | Feature | What it adds |
//! |---|---|---|
//! | [`providers`] | `providers` (default) | OpenAI-compatible and Anthropic models, streaming |
//! | [`tools`] | `tools` (default) | Workspace file tools, search, sandboxed shell, web fetch, checklist |
//! | [`policies`] | `policies` (default) | Approvals, scope guard, completion check, redaction, instructions, context window, conversation compaction, memory, tracing |
//! | [`agents`] | `agents` (default) | Agents as tools, child budgets |
//! | [`store`] | `sqlite` (default) | Durable runs on SQLite, crash recovery queries |
//! | [`mcp`] | `mcp` | MCP servers as tools |
//! | [`eval`] | `eval` | YAML scenarios, N-of-M pass rates, baseline comparison |
//! | [`memory`] | `memory` (+ `fastembed`) | Long-term memory: facts, episodes, hybrid recall; every part swappable |
//! | [`channels`] | `channels` | The `Channel` contract and the runner that connects channels to agents |
//! | [`channels::telegram`], [`channels::slack`], [`channels::discord`] | `telegram`, `slack`, `discord` | Built-in channel adapters |
//! | [`testing`] | `testing` | Scripted provider for tests |
//!
//! ```no_run
//! # #[cfg(all(feature = "providers", feature = "tools", feature = "policies"))]
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! use aidaemon_sdk::prelude::*;
//!
//! let workspace = tools::Workspace::new(".")?;
//! let mut builder = Agent::builder("helper")
//!     .model("claude-sonnet-5-5")
//!     .instructions("You maintain this repository. Verify changes with tests.")
//!     .provider(providers::Anthropic::builder().api_key("<key>").build()?)
//!     .hook(policies::ApprovalPolicy::new(policies::ApprovalMode::DestructiveOnly))
//!     .hook(policies::ScopeGuard::new([workspace.root()]))
//!     .hook(policies::UntrustedOutput::default());
//! for tool in tools::default_tools(workspace) {
//!     builder = builder.tool_arc(tool);
//! }
//! let agent = builder.build()?;
//!
//! let mut outcome = agent.run("cli:synthetic-user-1", "Fix the failing test").await?;
//! while let RunStatus::Interrupted { interruption } = &outcome.status {
//!     println!("approve {}? (auto-approving in this example)", interruption.call.name);
//!     outcome = agent.resume(outcome.state, Resolution::Approve).await?;
//! }
//! println!("{:?}: {}", outcome.status, outcome.final_text.unwrap_or_default());
//! # Ok(())
//! # }
//! ```
//!
//! See `crates/aidaemon-core/DESIGN.md` for the design and
//! `crates/aidaemon-core/PLAN.md` for the roadmap.

pub use aidaemon_core::*;

#[cfg(feature = "providers")]
pub use aidaemon_providers as providers;

#[cfg(feature = "tools")]
pub use aidaemon_tools as tools;

#[cfg(feature = "policies")]
pub use aidaemon_policies as policies;

#[cfg(feature = "agents")]
pub use aidaemon_agents as agents;

#[cfg(feature = "sqlite")]
pub use aidaemon_store_sqlite as store;

#[cfg(feature = "mcp")]
pub use aidaemon_mcp as mcp;

#[cfg(feature = "eval")]
pub use aidaemon_eval as eval;

#[cfg(feature = "memory")]
pub use aidaemon_memory as memory;

/// Chat channels: the contract, the runner, and the built-in adapters.
#[cfg(feature = "channels")]
pub mod channels {
    pub use aidaemon_channels::*;

    #[cfg(feature = "discord")]
    pub use aidaemon_channel_discord as discord;
    #[cfg(feature = "slack")]
    pub use aidaemon_channel_slack as slack;
    #[cfg(feature = "telegram")]
    pub use aidaemon_channel_telegram as telegram;
}

/// The names most agents need, plus the enabled modules.
pub mod prelude {
    pub use aidaemon_core::{
        Agent, AgentBuilder, Event, Hook, HookContext, Limits, Message, Provider, Resolution,
        RunOutcome, RunState, RunStatus, StopDecision, Tool, ToolAnnotations, ToolContext,
        ToolDecision, ToolInfo, ToolOutput, ToolResult, ToolSpec,
    };

    #[cfg(feature = "agents")]
    pub use crate::agents;
    #[cfg(feature = "channels")]
    pub use crate::channels;
    #[cfg(feature = "eval")]
    pub use crate::eval;
    #[cfg(feature = "mcp")]
    pub use crate::mcp;
    #[cfg(feature = "memory")]
    pub use crate::memory;
    #[cfg(feature = "policies")]
    pub use crate::policies;
    #[cfg(feature = "providers")]
    pub use crate::providers;
    #[cfg(feature = "sqlite")]
    pub use crate::store;
    #[cfg(feature = "tools")]
    pub use crate::tools;
}
