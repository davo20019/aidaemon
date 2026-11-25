//! A memory system's part in a turn: the scope it gets ([`MemoryTurn`]),
//! the hooks and tools it adds ([`MemoryParts`]) through [`MemoryPlugin`],
//! and how the daemon's own memory tools are offered beside it.

pub use crate::memory_port::{MemoryParts, MemoryPlugin, MemoryTurn, MemoryWarmup};
use aidaemon_channels::context::ChannelContext;
use aidaemon_tools::semantics::UserRole;

/// The daemon's built-in memory tools. Offered only while the active plugin
/// lists them in [`MemoryParts::owned_daemon_tools`].
pub const DAEMON_MEMORY_TOOLS: &[&str] = &["remember_fact", "manage_memories", "manage_people"];

/// `manage_memories` actions that manage goals, not memory. They stay with
/// the daemon whatever the plugin: under a plugin that does not own
/// `manage_memories`, the tool is offered narrowed to these actions.
pub const MANAGE_MEMORIES_GOAL_ACTIONS: &[&str] = &[
    "create_personal_goal",
    "list_goals",
    "complete_goal",
    "abandon_goal",
];

/// How a daemon tool is offered under the active plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonToolOffer {
    Full,
    /// `manage_memories` with only [`MANAGE_MEMORIES_GOAL_ACTIONS`].
    GoalActions,
    Hidden,
}

/// How the daemon tool `name` is offered while the active plugin owns the
/// daemon memory tools `owned`.
pub fn daemon_tool_offer(name: &str, owned: &[&str]) -> DaemonToolOffer {
    if !DAEMON_MEMORY_TOOLS.contains(&name) || owned.contains(&name) {
        DaemonToolOffer::Full
    } else if name == "manage_memories" {
        DaemonToolOffer::GoalActions
    } else {
        DaemonToolOffer::Hidden
    }
}

/// `manage_memories`' function definition (bare or wrapped in `function`)
/// narrowed to its goal actions.
pub fn goal_actions_view(definition: &serde_json::Value) -> serde_json::Value {
    let mut definition = definition.clone();
    let function = match definition.get_mut("function") {
        Some(function) => function,
        None => &mut definition,
    };
    function["description"] =
        serde_json::Value::String("Personal goals: create, list, complete, abandon.".to_string());
    if let Some(actions) = function
        .pointer_mut("/parameters/properties/action/enum")
        .and_then(serde_json::Value::as_array_mut)
    {
        actions.retain(|action| {
            action
                .as_str()
                .is_some_and(|action| MANAGE_MEMORIES_GOAL_ACTIONS.contains(&action))
        });
    }
    definition
}

/// Rejects a `manage_memories` call outside the goal-actions view.
pub fn check_goal_action(arguments: &serde_json::Value) -> Result<(), String> {
    let action = arguments
        .get("action")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if MANAGE_MEMORIES_GOAL_ACTIONS.contains(&action) {
        Ok(())
    } else {
        Err(format!(
            "manage_memories action `{action}` is not available here: it only manages goals. \
             Use the memory tools you were given for facts and preferences."
        ))
    }
}

/// The project whose memory a turn reads and writes: the turn's project,
/// only on the owner's own turns in a private or internal channel, never
/// under a mandate. Guests, groups and public channels keep today's memory.
pub fn memory_project(
    project: Option<&aidaemon_core::ProjectRef>,
    user_role: UserRole,
    channel_ctx: &ChannelContext,
    mandate: bool,
) -> Option<aidaemon_core::ProjectRef> {
    memory_project_allowed(user_role, channel_ctx, mandate)
        .then(|| project.cloned())
        .flatten()
}

/// Whether a turn may keep project memory at all (see [`memory_project`]).
pub fn memory_project_allowed(
    user_role: UserRole,
    channel_ctx: &ChannelContext,
    mandate: bool,
) -> bool {
    user_role == UserRole::Owner && channel_ctx.should_inject_personal_memory() && !mandate
}

impl MemoryTurn {
    /// Worker sessions keep summaries but never learn owner facts.
    pub fn memory_isolated(&self) -> bool {
        self.depth > 0 || aidaemon_events::is_memory_isolated_worker_session(&self.session_id)
    }
}

/// The parts `plugin` adds to `turn`. A mandate worker gets none, whatever
/// the plugin: its authority envelope excludes owner memory.
pub async fn memory_parts(plugin: &dyn MemoryPlugin, turn: &MemoryTurn) -> MemoryParts {
    if turn.mandate {
        return MemoryParts::default();
    }
    MemoryParts {
        owned_daemon_tools: plugin.owned_daemon_tools(),
        ..plugin.turn(turn).await
    }
}
