//! Deferred / circular wiring performed during startup.
//!
//! Several runtime components hold references to each other in cycles that
//! cannot be expressed with plain owning `Arc`s during construction:
//!
//! - **Agent ↔ SpawnAgentTool**: the agent owns the spawn tool (via its tool
//!   list), and the spawn tool needs to spawn sub-agents through the agent.
//!   The spawn tool therefore holds a `Weak<Agent>`, installed *after* the
//!   agent is constructed via `set_agent`.
//! - **Agent ↔ self**: the agent spawns background tasks that re-enter the
//!   agent loop, so it keeps a `Weak<Agent>` self-reference (`set_self_ref`).
//! - **Tools / Agent ↔ ChannelHub**: the `ChannelHub` is built *after* the
//!   agent and tools (it needs the agent for delivery notes), but the spawn
//!   tool, terminal tool, CLI-agent tool, and the agent itself all need to
//!   push notifications/progress through the hub. They each hold a
//!   `Weak<ChannelHub>`, installed once the hub exists.
//!
//! Using `Weak` references breaks the reference cycles so the graph can be
//! dropped cleanly, but it forces the wiring to be *deferred* until both
//! endpoints exist. This module centralizes those deferred calls so the
//! ordering and arguments live in one place rather than being scattered
//! through `core.rs`. The two phases mirror the two construction moments:
//! [`wire_agent_cycles`] runs right after the agent is built (before the hub
//! exists), and [`wire_hub_cycles`] runs right after the `ChannelHub` is built.

use std::sync::Arc;

use crate::agent::Agent;
use crate::channels::ChannelHub;

/// Phase 1: wiring that only needs the constructed [`Agent`] (no hub yet).
///
/// Must be called immediately after the agent is constructed, in the same
/// order as the original inline calls:
/// 1. give the spawn tool a weak agent reference,
/// 2. give the agent a weak self-reference for background task spawning.
pub async fn wire_agent_cycles(
    agent: &Arc<Agent>,
    spawn_tool: Option<&Arc<crate::tools::SpawnAgentTool>>,
) {
    // Close the loop: give the spawn tool a weak reference to the agent.
    if let Some(st) = spawn_tool {
        st.set_agent(Arc::downgrade(agent));
    }

    // Give the agent a weak self-reference for background task spawning.
    agent.set_self_ref(Arc::downgrade(agent)).await;
}

/// Phase 2: wiring that needs the constructed [`ChannelHub`].
///
/// Must be called immediately after the hub is constructed, in the same order
/// as the original inline calls:
/// 1. spawn tool → hub (background mode notifications),
/// 2. terminal tool → hub + agent (background command progress/completion and
///    loop re-engagement),
/// 3. CLI-agent tool → hub,
/// 4. agent → hub (background task notifications).
///
/// Takes ownership of the optional tool handles since they are not needed after
/// this point in startup (matching the original inline consumption).
pub async fn wire_hub_cycles(
    agent: &Arc<Agent>,
    hub: &Arc<ChannelHub>,
    spawn_tool: Option<Arc<crate::tools::SpawnAgentTool>>,
    terminal_tool: Option<Arc<crate::tools::TerminalTool>>,
    cli_agent_tool: Option<Arc<crate::tools::CliAgentTool>>,
    plan_store: Arc<crate::plans::PlanStore>,
) {
    // Give the spawn tool a reference to the hub for background mode notifications.
    if let Some(st) = spawn_tool {
        st.set_hub(Arc::downgrade(hub));
    }

    // Give terminal a reference to the hub so background command progress/completion
    // can be delivered immediately (not only via heartbeat queue polling).
    // Also give it a weak agent reference so background completions can re-engage
    // the agent loop to process the output and continue the original task.
    if let Some(tt) = terminal_tool {
        tt.set_hub(Arc::downgrade(hub));
        tt.set_agent(Arc::downgrade(agent));
        // Durable plan store for completion-time deliverable attribution and the
        // conservative delivery-step write-back.
        tt.set_plan_store(plan_store);
    }
    if let Some(cat) = cli_agent_tool {
        cat.set_hub(Arc::downgrade(hub));
    }

    // Give the agent a reference to the hub for background task notifications.
    agent.set_hub(Arc::downgrade(hub)).await;
}
