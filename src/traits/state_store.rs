use async_trait::async_trait;

pub use aidaemon_events::provider::TokenUsageStore;
pub use aidaemon_events::stores::{
    ConversationSummaryStore, DialogueStateStore, MessageStore, PromptSnapshotStore,
};
pub use aidaemon_memory::stores::{EpisodeStore, FactStore, PeopleStore};
pub use aidaemon_tools::services::{
    DynamicCliAgentStore, OAuthGatewayStore, OAuthStore, SettingsStore,
};
#[allow(unused_imports)]
pub use aidaemon_work::stores::*;

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

/// Database health check — verifies the connection is alive.
#[async_trait]
pub trait HealthCheckStore: Send + Sync {
    async fn health_check(&self) -> anyhow::Result<()> {
        Ok(())
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
