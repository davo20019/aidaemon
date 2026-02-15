use async_trait::async_trait;

use crate::types::{ChannelVisibility, FactPrivacy};

/// Session message storage and context retrieval.
#[async_trait]
pub trait MessageStore: Send + Sync {
    /// Append a message to the session history (both DB and working memory).
    async fn append_message(&self, msg: &super::Message) -> anyhow::Result<()>;

    /// Get recent messages for a session from working memory.
    async fn get_history(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<super::Message>>;

    /// Get context using Tri-Hybrid retrieval (Recency + Vector + Salience).
    /// Default implementation just calls `get_history`.
    async fn get_context(
        &self,
        session_id: &str,
        _query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<super::Message>> {
        self.get_history(session_id, limit).await
    }

    /// Clear conversation history for a session (working memory + DB messages).
    /// Facts are preserved.
    async fn clear_session(&self, session_id: &str) -> anyhow::Result<()>;
}

/// Layer-2 facts storage and retrieval (including privacy + channel provenance).
#[async_trait]
pub trait FactStore: Send + Sync {
    /// Upsert a fact with channel provenance and privacy level.
    async fn upsert_fact(
        &self,
        category: &str,
        key: &str,
        value: &str,
        source: &str,
        channel_id: Option<&str>,
        privacy: FactPrivacy,
    ) -> anyhow::Result<()>;

    /// Get all facts, optionally filtered by category.
    async fn get_facts(&self, category: Option<&str>) -> anyhow::Result<Vec<super::Fact>>;

    /// Get facts semantically relevant to a query, falling back to `get_facts` on error.
    async fn get_relevant_facts(
        &self,
        _query: &str,
        max: usize,
    ) -> anyhow::Result<Vec<super::Fact>> {
        // Default: return all facts (capped). Implementations can override with semantic filtering.
        let mut facts = self.get_facts(None).await?;
        facts.truncate(max);
        Ok(facts)
    }

    /// Get facts for a specific channel context, respecting privacy levels.
    async fn get_relevant_facts_for_channel(
        &self,
        query: &str,
        max: usize,
        _channel_id: Option<&str>,
        _visibility: ChannelVisibility,
    ) -> anyhow::Result<Vec<super::Fact>> {
        self.get_relevant_facts(query, max).await
    }

    /// Get cross-channel hints: channel-scoped facts from OTHER channels relevant to the query.
    async fn get_cross_channel_hints(
        &self,
        _query: &str,
        _current_channel_id: &str,
        _max: usize,
    ) -> anyhow::Result<Vec<super::Fact>> {
        Ok(vec![])
    }

    /// Update a fact's privacy level (e.g., channel → global after approval).
    async fn update_fact_privacy(
        &self,
        _fact_id: i64,
        _privacy: FactPrivacy,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Soft-delete a fact by superseding it.
    async fn delete_fact(&self, _fact_id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get all active facts with provenance info for memory management display.
    async fn get_all_facts_with_provenance(&self) -> anyhow::Result<Vec<super::Fact>> {
        self.get_facts(None).await
    }
}

/// Episodic memory storage and retrieval.
#[async_trait]
pub trait EpisodeStore: Send + Sync {
    /// Get episodes relevant to a query.
    async fn get_relevant_episodes(
        &self,
        _query: &str,
        _limit: usize,
    ) -> anyhow::Result<Vec<super::Episode>> {
        Ok(vec![])
    }

    /// Get episodes for a specific channel context.
    async fn get_relevant_episodes_for_channel(
        &self,
        _query: &str,
        _limit: usize,
        _channel_id: Option<&str>,
    ) -> anyhow::Result<Vec<super::Episode>> {
        Ok(vec![])
    }
}

/// Token usage persistence.
#[async_trait]
pub trait TokenUsageStore: Send + Sync {
    /// Record token usage from an LLM call.
    async fn record_token_usage(
        &self,
        _session_id: &str,
        _usage: &super::TokenUsage,
    ) -> anyhow::Result<()> {
        Ok(()) // default no-op
    }

    /// Get token usage records since a given datetime string (ISO 8601).
    async fn get_token_usage_since(
        &self,
        _since: &str,
    ) -> anyhow::Result<Vec<super::TokenUsageRecord>> {
        Ok(vec![]) // default no-op
    }

    /// Get token usage grouped by session_id since a given datetime.
    /// Returns Vec of (session_id, total_input_tokens, total_output_tokens, request_count).
    #[allow(dead_code)] // Used by token usage tooling when that tool is enabled.
    async fn get_token_usage_by_session(
        &self,
        _since: &str,
    ) -> anyhow::Result<Vec<(String, i64, i64, i64)>> {
        Ok(vec![]) // default no-op
    }
}

/// Learning system: procedures, expertise, behavior patterns, and error solutions.
#[async_trait]
pub trait LearningStore: Send + Sync {
    /// Get behavior patterns above a confidence threshold.
    async fn get_behavior_patterns(
        &self,
        _min_confidence: f32,
    ) -> anyhow::Result<Vec<super::BehaviorPattern>> {
        Ok(vec![])
    }

    /// Get procedures relevant to a query.
    async fn get_relevant_procedures(
        &self,
        _query: &str,
        _limit: usize,
    ) -> anyhow::Result<Vec<super::Procedure>> {
        Ok(vec![])
    }

    /// Get error solutions relevant to an error message.
    async fn get_relevant_error_solutions(
        &self,
        _error: &str,
        _limit: usize,
    ) -> anyhow::Result<Vec<super::ErrorSolution>> {
        Ok(vec![])
    }

    /// Get all expertise records.
    async fn get_all_expertise(&self) -> anyhow::Result<Vec<super::Expertise>> {
        Ok(vec![])
    }

    /// Get the user profile.
    async fn get_user_profile(&self) -> anyhow::Result<Option<super::UserProfile>> {
        Ok(None)
    }

    /// Get trusted command patterns for AI context.
    /// Returns patterns with 3+ approvals, ordered by approval count.
    async fn get_trusted_command_patterns(&self) -> anyhow::Result<Vec<(String, i32)>> {
        Ok(vec![])
    }

    /// Increment expertise counters and update level for a domain.
    async fn increment_expertise(
        &self,
        _domain: &str,
        _success: bool,
        _error: Option<&str>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Insert or update a procedure.
    async fn upsert_procedure(&self, _procedure: &super::Procedure) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Update procedure outcome after execution.
    #[allow(dead_code)] // Reserved for procedure feedback loop
    async fn update_procedure_outcome(
        &self,
        _procedure_id: i64,
        _success: bool,
        _duration: Option<f32>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Insert a new error-solution pair.
    async fn insert_error_solution(&self, _solution: &super::ErrorSolution) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Update error solution outcome.
    #[allow(dead_code)] // Reserved for error solution feedback loop
    async fn update_error_solution_outcome(
        &self,
        _solution_id: i64,
        _success: bool,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

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

    /// Check if a draft already exists for a given procedure name.
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

/// Generic key/value settings persistence.
#[async_trait]
pub trait SettingsStore: Send + Sync {
    /// Get a setting value by key. Returns None if unset.
    async fn get_setting(&self, _key: &str) -> anyhow::Result<Option<String>> {
        Ok(None)
    }

    /// Set a setting value. Creates or updates the key.
    async fn set_setting(&self, _key: &str, _value: &str) -> anyhow::Result<()> {
        Ok(())
    }
}

/// People persistence (social graph).
#[async_trait]
pub trait PeopleStore: Send + Sync {
    /// Create or update a person record. Returns the person ID.
    async fn upsert_person(&self, _person: &super::Person) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get a person by their database ID.
    async fn get_person(&self, _id: i64) -> anyhow::Result<Option<super::Person>> {
        Ok(None)
    }

    /// Look up a person by a platform-qualified sender ID (e.g., "slack:U123").
    async fn get_person_by_platform_id(
        &self,
        _platform_id: &str,
    ) -> anyhow::Result<Option<super::Person>> {
        Ok(None)
    }

    /// Find a person by name or alias (case-insensitive).
    async fn find_person_by_name(&self, _name: &str) -> anyhow::Result<Option<super::Person>> {
        Ok(None)
    }

    /// Get all people.
    async fn get_all_people(&self) -> anyhow::Result<Vec<super::Person>> {
        Ok(vec![])
    }

    /// Delete a person and all their facts (cascade).
    async fn delete_person(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Link a platform identity to a person.
    async fn link_platform_id(
        &self,
        _person_id: i64,
        _platform_id: &str,
        _display_name: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Update interaction tracking for a person.
    async fn touch_person_interaction(&self, _person_id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Create or update a fact about a person.
    async fn upsert_person_fact(
        &self,
        _person_id: i64,
        _category: &str,
        _key: &str,
        _value: &str,
        _source: &str,
        _confidence: f32,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get facts about a person, optionally filtered by category.
    async fn get_person_facts(
        &self,
        _person_id: i64,
        _category: Option<&str>,
    ) -> anyhow::Result<Vec<super::PersonFact>> {
        Ok(vec![])
    }

    /// Delete a person fact by ID.
    async fn delete_person_fact(&self, _fact_id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Confirm an auto-extracted person fact (set confidence to 1.0).
    async fn confirm_person_fact(&self, _fact_id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get people with upcoming dates (birthdays, important dates) within N days.
    async fn get_people_with_upcoming_dates(
        &self,
        _within_days: i32,
    ) -> anyhow::Result<Vec<(super::Person, super::PersonFact)>> {
        Ok(vec![])
    }

    /// Delete stale auto-extracted person facts older than N days with confidence < 1.0.
    async fn prune_stale_person_facts(&self, _retention_days: u32) -> anyhow::Result<u64> {
        Ok(0)
    }

    /// Get people who haven't interacted in more than N days.
    async fn get_people_needing_reconnect(
        &self,
        _inactive_days: u32,
    ) -> anyhow::Result<Vec<super::Person>> {
        Ok(vec![])
    }
}

/// OAuth-connected external services persistence.
#[async_trait]
pub trait OAuthStore: Send + Sync {
    /// Save an OAuth connection. Returns the connection ID.
    async fn save_oauth_connection(&self, _conn: &super::OAuthConnection) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get an OAuth connection by service name.
    async fn get_oauth_connection(
        &self,
        _service: &str,
    ) -> anyhow::Result<Option<super::OAuthConnection>> {
        Ok(None)
    }

    /// List all OAuth connections.
    async fn list_oauth_connections(&self) -> anyhow::Result<Vec<super::OAuthConnection>> {
        Ok(vec![])
    }

    /// Delete an OAuth connection by service name.
    async fn delete_oauth_connection(&self, _service: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Update token expiry for an OAuth connection.
    async fn update_oauth_token_expiry(
        &self,
        _service: &str,
        _expires_at: Option<&str>,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Goal persistence (goals/tasks, schedules, and dispatch bookkeeping).
#[async_trait]
pub trait GoalStore: Send + Sync {
    /// Create a new goal.
    async fn create_goal(&self, _goal: &super::Goal) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get a goal by ID.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_goal(&self, _id: &str) -> anyhow::Result<Option<super::Goal>> {
        Ok(None)
    }

    /// Update a goal (full replacement).
    #[allow(dead_code)] // Used in Phase 2
    async fn update_goal(&self, _goal: &super::Goal) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get all active orchestration goals (status = "active" or "pending").
    #[allow(dead_code)] // Used in Phase 2
    async fn get_active_goals(&self) -> anyhow::Result<Vec<super::Goal>> {
        Ok(vec![])
    }

    /// Get active personal goals (tracked, never dispatched).
    async fn get_active_personal_goals(&self, _limit: i64) -> anyhow::Result<Vec<super::Goal>> {
        Ok(vec![])
    }

    /// Update a personal goal's status and/or append a progress note.
    async fn update_personal_goal(
        &self,
        _goal_id: &str,
        _status: Option<&str>,
        _progress_note: Option<&str>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get orchestration goals for a specific session.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_goals_for_session(&self, _session_id: &str) -> anyhow::Result<Vec<super::Goal>> {
        Ok(vec![])
    }

    /// Get scheduled goals awaiting confirmation in a session.
    async fn get_pending_confirmation_goals(
        &self,
        _session_id: &str,
    ) -> anyhow::Result<Vec<super::Goal>> {
        Ok(vec![])
    }

    /// Activate a pending-confirmation goal.
    /// Returns true when the status transition was applied.
    async fn activate_goal(&self, _goal_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Create a new task within a goal.
    #[allow(dead_code)] // Used in Phase 2
    async fn create_task(&self, _task: &super::Task) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get a task by ID.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_task(&self, _id: &str) -> anyhow::Result<Option<super::Task>> {
        Ok(None)
    }

    /// Update a task (full replacement).
    #[allow(dead_code)] // Used in Phase 2
    async fn update_task(&self, _task: &super::Task) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get all tasks for a goal.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_tasks_for_goal(&self, _goal_id: &str) -> anyhow::Result<Vec<super::Task>> {
        Ok(vec![])
    }

    /// Count completed/skipped tasks for a goal (used by progress-based circuit breaker).
    async fn count_completed_tasks_for_goal(&self, _goal_id: &str) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Atomically claim a pending task for an executor.
    #[allow(dead_code)] // Used in Phase 2
    async fn claim_task(&self, _task_id: &str, _agent_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Log an activity entry for a task.
    #[allow(dead_code)] // Used in Phase 2
    async fn log_task_activity(&self, _activity: &super::TaskActivity) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get activity log for a task.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_task_activities(
        &self,
        _task_id: &str,
    ) -> anyhow::Result<Vec<super::TaskActivity>> {
        Ok(vec![])
    }

    /// Create a new schedule for a goal.
    async fn create_goal_schedule(&self, _schedule: &super::GoalSchedule) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get a schedule by ID.
    async fn get_goal_schedule(
        &self,
        _schedule_id: &str,
    ) -> anyhow::Result<Option<super::GoalSchedule>> {
        Ok(None)
    }

    /// List schedules for a goal.
    async fn get_schedules_for_goal(
        &self,
        _goal_id: &str,
    ) -> anyhow::Result<Vec<super::GoalSchedule>> {
        Ok(vec![])
    }

    /// Get due schedules for active orchestration goals.
    async fn get_due_goal_schedules(
        &self,
        _limit: i64,
    ) -> anyhow::Result<Vec<super::GoalSchedule>> {
        Ok(vec![])
    }

    /// Update a schedule (full replacement).
    async fn update_goal_schedule(&self, _schedule: &super::GoalSchedule) -> anyhow::Result<()> {
        Ok(())
    }

    /// Delete a schedule by ID. Returns true if a row was deleted.
    async fn delete_goal_schedule(&self, _schedule_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Cancel pending-confirmation goals older than max_age_secs.
    async fn cancel_stale_pending_confirmation_goals(
        &self,
        _max_age_secs: i64,
    ) -> anyhow::Result<u64> {
        Ok(0)
    }

    /// Get all orchestration goals that have schedules or are awaiting confirmation.
    async fn get_scheduled_goals(&self) -> anyhow::Result<Vec<super::Goal>> {
        Ok(vec![])
    }

    /// Reset tokens_used_today to 0 for all active goals.
    async fn reset_daily_token_budgets(&self) -> anyhow::Result<u64> {
        Ok(0)
    }

    /// Atomically add tokens to a goal's daily usage counter and return budget status.
    ///
    /// Use `delta_tokens = 0` to read the latest counters without modifying them.
    async fn add_goal_tokens_and_get_budget_status(
        &self,
        _goal_id: &str,
        _delta_tokens: i64,
    ) -> anyhow::Result<Option<super::GoalTokenBudgetStatus>> {
        Ok(None)
    }

    /// Get pending tasks ordered by priority, filtering out those with unmet dependencies.
    async fn get_pending_tasks_by_priority(&self, _limit: i64) -> anyhow::Result<Vec<super::Task>> {
        Ok(vec![])
    }

    /// Get tasks stuck in running/claimed state longer than timeout_secs.
    async fn get_stuck_tasks(&self, _timeout_secs: i64) -> anyhow::Result<Vec<super::Task>> {
        Ok(vec![])
    }

    /// Get tasks completed after a given timestamp.
    #[allow(dead_code)]
    async fn get_recently_completed_tasks(&self, _since: &str) -> anyhow::Result<Vec<super::Task>> {
        Ok(vec![])
    }

    /// Mark a running/claimed task as interrupted (e.g., after crash or timeout).
    async fn mark_task_interrupted(&self, _task_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Count active evergreen (continuous) goals.
    async fn count_active_evergreen_goals(&self) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get goals that completed/failed but haven't been notified to the user yet.
    async fn get_goals_needing_notification(&self) -> anyhow::Result<Vec<super::Goal>> {
        Ok(vec![])
    }

    /// Mark a goal as notified (set notified_at timestamp).
    async fn mark_goal_notified(&self, _goal_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Mark stale active goals as abandoned/failed.
    ///
    /// - Finite orchestration goals: active goals with no update in `stale_hours` → failed
    /// - Continuous orchestration goals: skipped (they have their own idle detection)
    /// - Personal goals: skipped
    ///
    /// Returns the number of goals cleaned up.
    async fn cleanup_stale_goals(&self, _stale_hours: i64) -> anyhow::Result<u64> {
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

/// Notification delivery queue persistence.
#[async_trait]
pub trait NotificationStore: Send + Sync {
    /// Enqueue a notification for delivery.
    async fn enqueue_notification(&self, _entry: &super::NotificationEntry) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get pending notifications ordered by priority (critical first), then creation time.
    async fn get_pending_notifications(
        &self,
        _limit: i64,
    ) -> anyhow::Result<Vec<super::NotificationEntry>> {
        Ok(vec![])
    }

    /// Mark a notification as delivered.
    async fn mark_notification_delivered(&self, _notification_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Increment the attempt counter for a notification.
    async fn increment_notification_attempt(&self, _notification_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Delete expired status_update notifications (past their expires_at).
    async fn cleanup_expired_notifications(&self) -> anyhow::Result<i64> {
        Ok(0)
    }
}

/// Facade trait kept for backwards compatibility.
///
/// This lets call sites keep using `Arc<dyn StateStore>`, while new code can
/// depend on focused store traits like `FactStore` or `PeopleStore`.
pub trait StateStore:
    Send
    + Sync
    + MessageStore
    + FactStore
    + EpisodeStore
    + TokenUsageStore
    + LearningStore
    + SkillStore
    + DynamicBotStore
    + SessionChannelStore
    + DynamicMcpServerStore
    + DynamicCliAgentStore
    + SettingsStore
    + PeopleStore
    + OAuthStore
    + GoalStore
    + ConversationSummaryStore
    + HealthCheckStore
    + NotificationStore
{
}

impl<T> StateStore for T where
    T: Send
        + Sync
        + MessageStore
        + FactStore
        + EpisodeStore
        + TokenUsageStore
        + LearningStore
        + SkillStore
        + DynamicBotStore
        + SessionChannelStore
        + DynamicMcpServerStore
        + DynamicCliAgentStore
        + SettingsStore
        + PeopleStore
        + OAuthStore
        + GoalStore
        + ConversationSummaryStore
        + HealthCheckStore
        + NotificationStore
{
}
