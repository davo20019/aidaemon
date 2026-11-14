//! A memory system's part in a turn: the scope it gets ([`MemoryTurn`]),
//! the hooks and tools it adds ([`MemoryParts`]) through [`MemoryPlugin`],
//! and how the daemon's own memory tools are offered beside it.

use std::sync::Arc;

use aidaemon_channels::context::ChannelContext;
use aidaemon_core as sdk;
use aidaemon_tools::semantics::UserRole;
use async_trait::async_trait;

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

/// Everything a plugin needs to scope memory for one turn.
#[derive(Clone)]
pub struct MemoryTurn {
    pub session_id: String,
    pub task_id: String,
    /// The conversation turn this run answers. It is over when the run ends,
    /// before its `task_end` is appended.
    pub turn_id: String,
    /// The turn's event emitter (decision points, memory telemetry).
    pub emitter: aidaemon_events::EventEmitter,
    /// The persisted user message for the turn.
    pub user_text: String,
    pub user_role: UserRole,
    pub channel_ctx: ChannelContext,
    /// The compiled policy allows reading memory.
    pub memory_pipeline_allowed: bool,
    /// The compiled policy allows writing memory.
    pub memory_persistence_allowed: bool,
    /// A mandate worker: no memory at all.
    pub mandate: bool,
    pub depth: usize,
    pub model: String,
    /// The run's learning context; the dispatcher adds its tool calls.
    pub learning: Arc<std::sync::Mutex<crate::turn::post_task::LearningContext>>,
    /// How the turn judges the run's final status.
    pub verdict: Arc<crate::engine::outcome::TurnVerdict>,
    /// Whether the turn may be learned from ("don't remember this"), shared
    /// with whatever memory system serves the turn.
    pub learning_consent: aidaemon_policies::TurnLearning,
}

impl MemoryTurn {
    /// Worker sessions keep summaries but never learn owner facts.
    pub fn memory_isolated(&self) -> bool {
        self.depth > 0 || aidaemon_events::is_memory_isolated_worker_session(&self.session_id)
    }
}

/// What a plugin adds to one turn.
#[derive(Default)]
pub struct MemoryParts {
    pub hooks: Vec<Arc<dyn sdk::Hook>>,
    pub tools: Vec<Arc<dyn sdk::Tool>>,
    /// Daemon tools this plugin owns; offered only while it is the active
    /// plugin. Filled from [`MemoryPlugin::owned_daemon_tools`].
    pub owned_daemon_tools: &'static [&'static str],
}

#[async_trait]
pub trait MemoryPlugin: Send + Sync {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts;

    /// The daemon memory tools ([`DAEMON_MEMORY_TOOLS`]) this plugin owns.
    fn owned_daemon_tools(&self) -> &'static [&'static str] {
        &[]
    }

    /// This is the daemon's own memory, so the daemon's prompts about facts
    /// and its learning from finished runs apply. Another memory system may
    /// have neither.
    fn is_daemons_own(&self) -> bool {
        false
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
