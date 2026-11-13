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
//! | [`policies`] | `policies` (default) | Approvals, scope and egress guards, streak guard, redaction, instructions, context window, conversation compaction, memory, tracing |
//! | [`secrets`] | `secrets` (default; `keychain` for the OS store) | Secret stores whose values come out as `Secret` and are redacted from every run |
//! | [`agents`] | `agents` (default) | Agents as tools, child budgets |
//! | [`store`] | `sqlite` (default) | Durable runs on SQLite, crash recovery queries, upkeep (`store::maintenance`: planner statistics, live snapshots) |
//! | [`mcp`] | `mcp` | MCP servers as tools |
//! | [`eval`] | `eval` | YAML scenarios, N-of-M pass rates, baseline comparison |
//! | [`memory`] | `memory` (+ `fastembed`) | Long-term memory: facts, episodes, hybrid recall; every part swappable |
//! | [`events`] | `events` | The immutable event store, its payloads and the run aggregate |
//! | [`work`] | `work` | **Internal, unstable:** the daemon's goals, mandates, tasks and notifications. Not for applications; schedule your own work with [`ServiceHost`] |
//! | [`channels`] | `channels` | The `Channel` contract and the runner that connects channels to agents |
//! | [`channels::telegram`], [`channels::slack`], [`channels::discord`] | `telegram`, `slack`, `discord` | Built-in channel adapters |
//! | `channels::conformance` | `conformance` | The channel conformance kit, for your own adapter's tests |
//! | [`secrets`] | `secrets` | Where an agent keeps its secrets: the `SecretStore` trait, the OS keychain, an env file, an in-memory store |
//! | [`testing`] | `testing` | Scripted provider for tests (and Tokio's `test-util`, for paused time); the shared behavior suite (`testing::behavior`) |
//! | [`tool`] | `macros` (default) | `#[tool]`: a tool from a plain function, schema derived |
//! | [`mod@preset`] | `providers` + `policies` (default) | A short start: provider, safeguards, store and default tools in a few lines |
//! | [`structured`] | always | Typed output: a model call or an agent's final answer as your type, one repair retry |
//!
//! Background work is in the kernel, at the crate root: a [`Service`]
//! states its [`Schedule`] (every so often, every tick, continuously, or
//! daily at a UTC time) and a [`ServiceHost`] runs it, with backoff after
//! failures, one run at a time per service, and one [`ServiceEvent::Ended`]
//! per run (its number, outcome and duration) for a [`ServiceObserver`].
//! The `scheduled_worker` example drives an agent from one.
//!
//! ```no_run
//! # #[cfg(all(feature = "providers", feature = "tools", feature = "policies"))]
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! use aidaemon_sdk::prelude::*;
//! use aidaemon_sdk::{Authority, AuthorityHook};
//!
//! let workspace = tools::Workspace::new(".")?;
//! let mut builder = Agent::builder("helper")
//!     .model("claude-sonnet-5-5")
//!     .instructions("You maintain this repository. Verify changes with tests.")
//!     .provider(providers::Anthropic::builder().api_key("<key>").build()?)
//!     .hook(AuthorityHook::new(Authority::full()))
//!     .hook(policies::ScopeGuard::new([workspace.root()]))
//!     .hook(policies::EgressGuard::new())
//!     .hook(policies::ApprovalPolicy::new(policies::ApprovalMode::DestructiveOnly))
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
//! See the SDK guide at <https://docs.aidaemon.ai/sdk>.

pub use aidaemon_core::*;

/// Test helpers: the kernel's scripted provider and response builders and,
/// with the `macros`, `agents` and `channels` features, the
/// [`behavior`](testing::behavior) suite every assembled agent should pass.
#[cfg(feature = "testing")]
pub mod testing {
    pub use aidaemon_core::testing::*;

    #[cfg(all(
        feature = "macros",
        feature = "agents",
        feature = "channels",
        feature = "providers",
        feature = "policies"
    ))]
    pub mod behavior;
}

pub mod schema;
pub mod structured;

#[cfg(all(feature = "providers", feature = "policies"))]
pub mod preset;
#[cfg(all(feature = "providers", feature = "policies"))]
pub use preset::{Preset, PresetError, preset};

#[cfg(feature = "runtime")]
pub mod runtime;

#[doc(hidden)]
#[path = "macro_support.rs"]
pub mod __private;

/// Write a tool as a plain function: `#[tool]` derives its JSON schema from
/// the parameters (with `schemars`) and emits a unit struct, named after the
/// function in `PascalCase`, that implements [`Tool`].
///
/// ```
/// use aidaemon_sdk::{tool, Tool};
///
/// /// Look up an order: customer, item and total.
/// #[tool(read_only)]
/// async fn lookup_order(
///     /// The order id, like `ORD-1001`.
///     order_id: String,
/// ) -> String {
///     format!("{order_id}: Acme Kettle, 45.00 (synthetic)")
/// }
///
/// let spec = LookupOrder.spec();
/// assert_eq!(spec.name, "lookup_order");
/// assert_eq!(spec.parameters["additionalProperties"], false);
/// ```
///
/// - **Description:** the function's doc comment (required; or pass
///   `description = "..."`). A parameter's doc comment becomes its
///   description in the schema.
/// - **Name:** the function's name, or `name = "..."`.
/// - **Arguments:** each parameter is one key of the argument object
///   (owned types implementing `Deserialize` and `JsonSchema`; `Option<T>`
///   is optional). `#[serde(...)]` and `#[schemars(...)]` attributes on a
///   parameter are kept. Or mark one parameter `#[args]` to take the whole
///   object as your own type (`#[derive(Deserialize, JsonSchema)]`). A
///   `&ToolContext` parameter, anywhere, receives the call's context
///   (session, cancellation, the application's `extensions`).
/// - **Returns:** `String`, `&str`, `ToolOutput`, `serde_json::Value`, or a
///   `Result` of one of them whose error displays (an error reaches the
///   model as a failed call). The function may be sync or async.
/// - **Annotations** are explicit, never guessed. With no preset a tool is
///   [`ToolAnnotations::DESTRUCTIVE`], the same conservative default as a
///   hand-written tool. `read_only`, `write` and `destructive` pick
///   [`ToolAnnotations::READ_ONLY`], [`ToolAnnotations::WRITE`] or
///   [`ToolAnnotations::DESTRUCTIVE`]; `idempotent = bool` and
///   `egress = bool` (alias `open_world`) then override that one field. A
///   read-only tool that contacts another host must say `egress = true`.
/// - **Trust:** output is untrusted unless you pass `trusted` (the result
///   is the owner's own data, not content someone else wrote).
/// - **Effects and access:** `effects = path` names a
///   `fn(&Args) -> CallEffects`, `access = path` a
///   `fn(&Args) -> AccessManifest` (with `#[args]`, `Args` is your type).
///   Without them a tool gets the kernel's defaults: no effects for a
///   read-only call that is not egress, unknown otherwise.
/// - `timeout_secs = N` overrides `Limits::tool_timeout`; `crate = path`
///   points the generated code at a re-export of `aidaemon_sdk`.
///
/// Arguments are checked before anything runs: a call that is not an
/// object, has a key the schema does not declare, or does not deserialize
/// is refused by `validate` and answered with an error result by `call`.
#[cfg(feature = "macros")]
pub use aidaemon_macros::tool;

#[cfg(feature = "providers")]
pub use aidaemon_providers as providers;

#[cfg(feature = "tools")]
pub use aidaemon_tools as tools;

#[cfg(feature = "policies")]
pub use aidaemon_policies as policies;

#[cfg(feature = "secrets")]
pub use aidaemon_secrets as secrets;

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

#[cfg(feature = "events")]
pub use aidaemon_events as events;

#[cfg(feature = "work")]
pub use aidaemon_work as work;

/// Chat channels: the contract, the runner, and the built-in adapters.
#[cfg(feature = "channels")]
pub mod channels {
    pub use aidaemon_channels::*;

    /// The typed status a channel can render natively: helper runs, plan
    /// cards and steps ([`card`]), drawn as text by
    /// [`live_status::render_card`].
    pub use aidaemon_core::card;

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
        Agent, AgentBuilder, CallResult, Event, Hook, HookContext, Limits, Message, Provider,
        Receipt, ReceiptStatus, Resolution, Role, RunOutcome, RunState, RunStatus, SourceTrust,
        StopDecision, Tool, ToolAnnotations, ToolCall, ToolContext, ToolDecision, ToolError,
        ToolInfo, ToolOutput, ToolResult, ToolSpec,
    };

    #[cfg(feature = "macros")]
    pub use crate::tool;

    #[cfg(all(feature = "providers", feature = "policies"))]
    pub use crate::preset::{Preset, PresetError, preset};

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
