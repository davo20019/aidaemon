use async_trait::async_trait;

pub use aidaemon_events::provider::TokenUsageStore;
pub use aidaemon_memory::stores::{EpisodeStore, FactStore, PeopleStore};
pub use aidaemon_tools::services::{OAuthGatewayStore, OAuthStore, SettingsStore};
#[allow(unused_imports)]
pub use aidaemon_work::stores::*;

/// Session message storage and context retrieval.
#[async_trait]
pub trait MessageStore: Send + Sync {
    /// Append a message to the session history hot window.
    /// Canonical persistence is handled via emitted events.
    async fn append_message(&self, msg: &super::Message) -> anyhow::Result<()>;

    /// Add `annotation` to a stored user message (identified by its id,
    /// which is also its turn id), in the hot window and in the canonical
    /// event. A store that cannot annotate ignores it: annotations added
    /// this way are advisory rendering hints.
    async fn annotate_user_message(
        &self,
        _session_id: &str,
        _message_id: &str,
        _annotation: super::MessageAnnotation,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get recent messages for a session from working memory.
    async fn get_history(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<super::Message>>;

    /// Get context using Tri-Hybrid retrieval (Recency + Vector + Salience).
    /// Default implementation just calls `get_history`.
    ///
    /// Pillar B (Task 7): the last production caller (`load_initial_history`)
    /// was removed when the turn-anchored fetch took over history retention.
    /// Retained as part of the `MessageStore` contract (still exercised by the
    /// sqlite store tests); allow dead_code until a future caller or removal.
    #[allow(dead_code)]
    async fn get_context(
        &self,
        session_id: &str,
        _query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<super::Message>> {
        self.get_history(session_id, limit).await
    }

    /// HARD-clear a session: delete its canonical events + conversation summary
    /// (working memory too). Destructive and irreversible — this is `/wipe`.
    /// Facts already extracted into memory are preserved.
    async fn clear_session(&self, session_id: &str) -> anyhow::Result<()>;

    /// Reset a session's conversation CONTEXT without deleting anything. The
    /// next turn starts fresh (a durable boundary hides prior messages from
    /// context retrieval), but the event history remains for the memory
    /// pipeline and audit. This is what `/clear` should do. The default falls
    /// back to the destructive clear for stores that don't implement a
    /// boundary (keeps simple/test stores functional).
    async fn clear_session_context(&self, session_id: &str) -> anyhow::Result<()> {
        self.clear_session(session_id).await
    }

    /// Advance the durable context floor without hiding the current turn.
    ///
    /// This is the lifecycle equivalent of starting a new conversation thread:
    /// earlier events remain queryable for audit and explicit history lookup,
    /// while ordinary prompt/context hydration begins after
    /// `cleared_after_event_id`. Stores without a durable boundary may keep the
    /// default no-op; production stores should also prune their hot context to
    /// `retained_turn_id` so an in-memory cache cannot bypass the boundary.
    async fn advance_session_context_boundary(
        &self,
        _session_id: &str,
        _cleared_after_event_id: i64,
        _retained_turn_id: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Durable projection of a session's open request/question state.
#[async_trait]
pub trait DialogueStateStore: Send + Sync {
    async fn get_dialogue_state(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<super::DialogueState>>;

    async fn upsert_dialogue_state(&self, state: &super::DialogueState) -> anyhow::Result<()>;

    #[allow(dead_code)]
    async fn delete_dialogue_state(&self, session_id: &str) -> anyhow::Result<()>;
}

/// Learning storage lives in the SDK memory crate.
pub use aidaemon_memory::LearningStore;

/// Skills storage (deprecated dynamic skills + skill drafts).
#[async_trait]
pub trait SkillStore: Send + Sync {
    /// Store a dynamically added skill.
    /// Deprecated: use filesystem skills instead.
    #[allow(dead_code)]
    async fn add_dynamic_skill(&self, _skill: &super::DynamicSkill) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all dynamic skills.
    /// Deprecated: use filesystem skills instead.
    async fn get_dynamic_skills(&self) -> anyhow::Result<Vec<super::DynamicSkill>> {
        Ok(vec![])
    }

    /// Delete a dynamic skill by ID.
    /// Deprecated: use filesystem skills instead.
    #[allow(dead_code)]
    async fn delete_dynamic_skill(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Update the enabled flag of a dynamic skill.
    /// Deprecated: file existence = active, no enable/disable needed.
    #[allow(dead_code)]
    async fn update_dynamic_skill_enabled(&self, _id: i64, _enabled: bool) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get procedures eligible for skill promotion (success_count >= min_success, success rate >= min_rate).
    async fn get_promotable_procedures(
        &self,
        _min_success: i32,
        _min_rate: f32,
    ) -> anyhow::Result<Vec<super::Procedure>> {
        Ok(vec![])
    }

    /// Store a skill draft from auto-promotion. Returns the draft ID.
    async fn add_skill_draft(&self, _draft: &super::SkillDraft) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all pending skill drafts.
    async fn get_pending_skill_drafts(&self) -> anyhow::Result<Vec<super::SkillDraft>> {
        Ok(vec![])
    }

    /// Get a skill draft by ID.
    async fn get_skill_draft(&self, _id: i64) -> anyhow::Result<Option<super::SkillDraft>> {
        Ok(None)
    }

    /// Update a skill draft's status ("approved" or "dismissed").
    async fn update_skill_draft_status(&self, _id: i64, _status: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Check if any draft record already exists for a given procedure name
    /// (pending, approved, or dismissed).
    async fn skill_draft_exists_for_procedure(
        &self,
        _procedure_name: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }
}

/// Dynamic bots (runtime-managed) persistence.
#[async_trait]
pub trait DynamicBotStore: Send + Sync {
    /// Store a dynamically added bot configuration.
    async fn add_dynamic_bot(&self, _bot: &super::DynamicBot) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all dynamically added bots.
    async fn get_dynamic_bots(&self) -> anyhow::Result<Vec<super::DynamicBot>> {
        Ok(vec![])
    }

    /// Update the allowed_user_ids for a dynamic bot identified by its token.
    #[allow(dead_code)]
    async fn update_dynamic_bot_allowed_users(
        &self,
        _bot_token: &str,
        _allowed_user_ids: &[String],
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Delete a dynamic bot by ID.
    #[allow(dead_code)]
    async fn delete_dynamic_bot(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Session → channel mapping persistence.
#[async_trait]
pub trait SessionChannelStore: Send + Sync {
    /// Persist a session_id → channel_name mapping so it survives restarts.
    async fn save_session_channel(
        &self,
        _session_id: &str,
        _channel_name: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Load all persisted session → channel mappings (for populating session_map on startup).
    async fn load_session_channels(&self) -> anyhow::Result<Vec<(String, String)>> {
        Ok(vec![])
    }
}

/// Runtime-managed MCP servers persistence.
#[async_trait]
pub trait DynamicMcpServerStore: Send + Sync {
    /// Store a dynamically added MCP server.
    async fn save_dynamic_mcp_server(
        &self,
        _server: &super::DynamicMcpServer,
    ) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all dynamic MCP servers.
    async fn list_dynamic_mcp_servers(&self) -> anyhow::Result<Vec<super::DynamicMcpServer>> {
        Ok(vec![])
    }

    /// Delete a dynamic MCP server by ID.
    async fn delete_dynamic_mcp_server(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Update a dynamic MCP server.
    async fn update_dynamic_mcp_server(
        &self,
        _server: &super::DynamicMcpServer,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Runtime-managed CLI agents persistence + invocation logs.
#[async_trait]
pub trait DynamicCliAgentStore: Send + Sync {
    /// Store a dynamically added CLI agent.
    async fn save_dynamic_cli_agent(&self, _agent: &super::DynamicCliAgent) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all dynamic CLI agents.
    async fn list_dynamic_cli_agents(&self) -> anyhow::Result<Vec<super::DynamicCliAgent>> {
        Ok(vec![])
    }

    /// Delete a dynamic CLI agent by ID.
    async fn delete_dynamic_cli_agent(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Update a dynamic CLI agent.
    async fn update_dynamic_cli_agent(
        &self,
        _agent: &super::DynamicCliAgent,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Log the start of a CLI agent invocation. Returns the invocation ID.
    async fn log_cli_agent_start(
        &self,
        _task_id: Option<&str>,
        _session_id: &str,
        _agent_name: &str,
        _prompt_summary: &str,
        _working_dir: Option<&str>,
    ) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Log the completion of a CLI agent invocation.
    async fn log_cli_agent_complete(
        &self,
        _id: i64,
        _exit_code: Option<i32>,
        _output_summary: &str,
        _success: bool,
        _duration_secs: f64,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get recent CLI agent invocations (most recent first).
    async fn get_cli_agent_invocations(
        &self,
        _limit: usize,
    ) -> anyhow::Result<Vec<super::CliAgentInvocation>> {
        Ok(vec![])
    }

    /// Auto-close stale CLI agent invocations that never completed (e.g. crashed worker).
    ///
    /// Implementations should mark rows with `completed_at IS NULL` and older than
    /// `max_age_hours` as completed with `success=false`.
    async fn cleanup_stale_cli_agent_invocations(
        &self,
        _max_age_hours: i64,
    ) -> anyhow::Result<u64> {
        Ok(0)
    }
}

/// Sliding-window conversation summaries.
#[async_trait]
pub trait ConversationSummaryStore: Send + Sync {
    /// Get the conversation summary for a session.
    async fn get_conversation_summary(
        &self,
        _session_id: &str,
    ) -> anyhow::Result<Option<super::ConversationSummary>> {
        Ok(None)
    }

    /// Create or update a conversation summary for a session.
    async fn upsert_conversation_summary(
        &self,
        _summary: &super::ConversationSummary,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Database health check — verifies the connection is alive.
#[async_trait]
pub trait HealthCheckStore: Send + Sync {
    async fn health_check(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Rendered system-prompt snapshots, deduplicated by content hash.
///
/// Enables exact replay of past LLM calls: the `instructions_snapshot`
/// decision-point event records the core prompt's `core_hash` (plus the
/// volatile context tail inline); this store maps that hash back to the full
/// rendered core prompt text. Rows are written insert-or-ignore, so storage
/// grows only when the rendered prompt actually changes (deploys, config or
/// memory shape changes) — not per interaction.
#[async_trait]
pub trait PromptSnapshotStore: Send + Sync {
    /// Persist a rendered prompt keyed by its hash. Must be idempotent.
    async fn save_prompt_snapshot(&self, _hash: &str, _content: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Fetch a stored prompt snapshot by exact hash.
    #[allow(dead_code)] // Read path is db_probe; kept on the trait for tests/tools.
    async fn get_prompt_snapshot(&self, _hash: &str) -> anyhow::Result<Option<String>> {
        Ok(None)
    }
}

/// Focused persistence contract used by goal-trace tooling.
pub trait GoalTraceStore:
    GoalStore + GoalScheduleStore + MandateStore + TaskStore + WorkCoordinationStore + ScheduledRunStore
{
}

impl<T> GoalTraceStore for T where
    T: GoalStore
        + GoalScheduleStore
        + MandateStore
        + TaskStore
        + WorkCoordinationStore
        + ScheduledRunStore
{
}

/// Focused persistence contract used by mandate management. Includes the
/// schedule and scheduled-run lifecycle so the mandate surface can project
/// its controller goal's authoritative objective status.
pub trait MandateToolStore:
    MandateStore
    + GoalStore
    + WorkCoordinationStore
    + SessionChannelStore
    + GoalBudgetStore
    + GoalScheduleStore
    + ScheduledRunStore
{
}

impl<T> MandateToolStore for T where
    T: MandateStore
        + GoalStore
        + WorkCoordinationStore
        + SessionChannelStore
        + GoalBudgetStore
        + GoalScheduleStore
        + ScheduledRunStore
{
}

/// Focused persistence contract used by the dynamic MCP registry.
pub trait McpRegistryStore: DynamicMcpServerStore {}

impl<T> McpRegistryStore for T where T: DynamicMcpServerStore {}

/// Focused persistence contract used by skill startup and management.
pub trait SkillsStore: SkillStore + SettingsStore {}

impl<T> SkillsStore for T where T: SkillStore + SettingsStore {}

/// Focused persistence contract used by blocker reporting.
pub trait ReportBlockerStore:
    TaskStore + GoalStore + NotificationStore + WorkCoordinationStore
{
}

impl<T> ReportBlockerStore for T where
    T: TaskStore + GoalStore + NotificationStore + WorkCoordinationStore
{
}

/// Focused persistence contract used by scheduled-goal tooling.
pub trait ScheduledGoalRunsStore:
    GoalStore
    + GoalScheduleStore
    + ScheduledRunStore
    + GoalBudgetStore
    + TaskStore
    + WorkCoordinationStore
    + SettingsStore
    + MandateStore
{
}

impl<T> ScheduledGoalRunsStore for T where
    T: GoalStore
        + GoalScheduleStore
        + ScheduledRunStore
        + GoalBudgetStore
        + TaskStore
        + WorkCoordinationStore
        + SettingsStore
        + MandateStore
{
}

/// Facade trait kept for backwards compatibility.
///
/// This lets call sites keep using `Arc<dyn StateStore>`, while new code can
/// depend on focused store traits like `FactStore` or `PeopleStore`.
pub trait StateStore:
    Send
    + Sync
    + MessageStore
    + DialogueStateStore
    + super::HistorySearchStore
    + GoalTraceStore
    + MandateToolStore
    + McpRegistryStore
    + SkillsStore
    + ReportBlockerStore
    + ScheduledGoalRunsStore
    + OAuthGatewayStore
    + TokenUsageStore
    + SkillStore
    + DynamicBotStore
    + SessionChannelStore
    + DynamicMcpServerStore
    + DynamicCliAgentStore
    + SettingsStore
    + OAuthStore
    + GoalStore
    + MandateStore
    + TaskStore
    + GoalScheduleStore
    + GoalBudgetStore
    + ScheduledRunStore
    + TaskDispatchStore
    + WorkCoordinationStore
    + GoalNotificationStore
    + ConversationSummaryStore
    + HealthCheckStore
    + NotificationStore
    + PromptSnapshotStore
{
}

impl<T> StateStore for T where
    T: Send
        + Sync
        + MessageStore
        + DialogueStateStore
        + super::HistorySearchStore
        + GoalTraceStore
        + MandateToolStore
        + McpRegistryStore
        + SkillsStore
        + ReportBlockerStore
        + ScheduledGoalRunsStore
        + OAuthGatewayStore
        + TokenUsageStore
        + SkillStore
        + DynamicBotStore
        + SessionChannelStore
        + DynamicMcpServerStore
        + DynamicCliAgentStore
        + SettingsStore
        + OAuthStore
        + GoalStore
        + MandateStore
        + TaskStore
        + GoalScheduleStore
        + GoalBudgetStore
        + ScheduledRunStore
        + TaskDispatchStore
        + WorkCoordinationStore
        + GoalNotificationStore
        + ConversationSummaryStore
        + HealthCheckStore
        + NotificationStore
        + PromptSnapshotStore
{
}
