use async_trait::async_trait;

use crate::types::{ChannelVisibility, FactPrivacy};

/// Session message storage and context retrieval.
#[async_trait]
pub trait MessageStore: Send + Sync {
    /// Append a message to the session history hot window.
    /// Canonical persistence is handled via emitted events.
    async fn append_message(&self, msg: &super::Message) -> anyhow::Result<()>;

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

/// Layer-2 facts storage and retrieval (including privacy + channel provenance).
#[async_trait]
pub trait FactStore: Send + Sync {
    /// Reconcile an entity-aware personal-memory write atomically.
    async fn reconcile_personal_memory(
        &self,
        _write: &super::PersonalMemoryWrite,
        _source: &str,
        _source_excerpt: Option<&str>,
        _channel_id: Option<&str>,
        _privacy: FactPrivacy,
    ) -> anyhow::Result<super::PersonalMemoryWriteResult> {
        anyhow::bail!("structured personal memory is not supported by this store")
    }

    /// Compatibility projection of active canonical facts and relationships.
    async fn get_canonical_memory_facts(&self) -> anyhow::Result<Vec<super::Fact>> {
        Ok(Vec::new())
    }

    /// Upsert a fact with channel provenance and privacy level.
    async fn upsert_fact(
        &self,
        category: &str,
        key: &str,
        value: &str,
        source: &str,
        channel_id: Option<&str>,
        privacy: FactPrivacy,
    ) -> anyhow::Result<()> {
        self.upsert_fact_with_provenance(
            category, key, value, source, channel_id, privacy, None, None,
        )
        .await
    }

    /// Upsert a fact with full provenance data.
    #[allow(clippy::too_many_arguments)]
    async fn upsert_fact_with_provenance(
        &self,
        category: &str,
        key: &str,
        value: &str,
        source: &str,
        channel_id: Option<&str>,
        privacy: FactPrivacy,
        first_seen_at: Option<chrono::DateTime<chrono::Utc>>,
        source_excerpt: Option<&str>,
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
    ///
    /// `requester_is_owner` controls the DM short-circuit: only the owner sees the
    /// full unfiltered graph (incl. Private + other-channel facts) in a 1:1 DM. A
    /// non-owner (an allowlisted Guest) gets the same privacy filtering as a group
    /// channel — Global + same-channel facts only, never Private or other-channel —
    /// so owner secrets can't leak into a guest's prompt context.
    async fn get_relevant_facts_for_channel(
        &self,
        query: &str,
        max: usize,
        _channel_id: Option<&str>,
        _visibility: ChannelVisibility,
        _requester_is_owner: bool,
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

    /// Soft-delete a fact by category and key. Returns true if a fact was found and deleted.
    async fn delete_fact_by_key(&self, _category: &str, _key: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Get all active facts with provenance info for memory management display.
    async fn get_all_facts_with_provenance(&self) -> anyhow::Result<Vec<super::Fact>> {
        self.get_facts(None).await
    }

    /// Pure semantic (vector) search over active facts: returns `(fact, score)`
    /// pairs whose embedding similarity clears the relevance threshold, ranked by
    /// score, with NO recency padding. Unlike [`get_relevant_facts`] (which is
    /// tuned for context injection and pads sparse results with recent facts),
    /// this returns only genuine matches — suitable for supplementing the
    /// keyword-based memory search tool. Default returns empty (stores without an
    /// embedding index simply contribute nothing).
    async fn search_facts_semantic(
        &self,
        _query: &str,
        _max: usize,
    ) -> anyhow::Result<Vec<(super::Fact, f32)>> {
        Ok(vec![])
    }

    /// Persist a model-extracted semantic graph for an already-written fact.
    /// Implementations must validate the graph and retain claim provenance.
    async fn project_extracted_fact_graph(
        &self,
        _category: &str,
        _key: &str,
        _source_excerpt: &str,
        _graph: &super::ExtractedMemoryGraph,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Refresh canonical claim/vector projections for the active fact at this key.
    async fn refresh_fact_memory(&self, _category: &str, _key: &str) -> anyhow::Result<()> {
        Ok(())
    }

    async fn memory_health_report(&self) -> anyhow::Result<super::MemoryHealthReport> {
        Ok(super::MemoryHealthReport::default())
    }

    async fn repair_memory_projections(&self) -> anyhow::Result<super::MemoryHealthReport> {
        self.memory_health_report().await
    }

    /// Assemble the neighborhood of facts for the given resolved entity names and
    /// initial seed fact IDs (e.g. from an embedding search hit). Expands the set
    /// via namespace, co-mention, and owner-relationship cluster rules.
    ///
    /// The default no-op keeps non-SQLite stores compiling without change.
    /// Wired into production in Task 6 (`get_relevant_facts`).
    #[allow(dead_code)]
    async fn assemble_neighborhood(
        &self,
        _entities: &[String],
        _initial_ids: &std::collections::HashSet<i64>,
    ) -> anyhow::Result<Vec<super::Fact>> {
        Ok(vec![])
    }
}

/// Episodic memory storage and retrieval.
#[async_trait]
pub trait EpisodeStore: Send + Sync {
    /// Refresh derived canonical-memory projections for an episode. Stores that
    /// do not maintain secondary indexes may keep the default no-op.
    async fn project_episode_memory(&self, _episode_id: i64) -> anyhow::Result<()> {
        Ok(())
    }

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

    /// Get episodes only from one canonical session (used by untrusted
    /// internal/sub-agent origins).
    async fn get_relevant_episodes_for_session(
        &self,
        _query: &str,
        _limit: usize,
        _session_id: &str,
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
        _call_id: Option<&str>,
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

    /// Create (once) and read the immutable token budget for one mandate
    /// decision run. Implementations must validate the exact active mandate
    /// authority epoch and running goal-run fence. Returning success without
    /// durable enforcement would make autonomous execution fail open, so the
    /// default is deliberately unsupported.
    async fn ensure_mandate_run_token_budget(
        &self,
        _goal_run_id: &str,
        _mandate_id: &str,
        _mandate_version: i64,
        _budget_per_cycle: i64,
    ) -> anyhow::Result<(i64, i64)> {
        anyhow::bail!("durable mandate run token budgets are not supported by this store")
    }

    /// Try to serialize one model call across every worker in a mandate run.
    /// Returns `(acquired, tokens_used, budget_per_cycle)`. `acquired = false`
    /// means either another call holds the lease or the immutable cycle budget
    /// is exhausted; callers distinguish those states from the counters.
    async fn try_acquire_mandate_run_token_lease(
        &self,
        _goal_run_id: &str,
        _mandate_id: &str,
        _mandate_version: i64,
        _lease_token: &str,
        _lease_secs: i64,
    ) -> anyhow::Result<(bool, i64, i64)> {
        anyhow::bail!("durable mandate run token budgets are not supported by this store")
    }

    /// Mark the exact acquired lease as physically dispatched and
    /// pessimistically reserve its entire remaining cycle balance. This must
    /// commit before provider I/O begins.
    async fn mark_mandate_run_token_lease_dispatched(
        &self,
        _goal_run_id: &str,
        _mandate_id: &str,
        _mandate_version: i64,
        _lease_token: &str,
    ) -> anyhow::Result<bool> {
        anyhow::bail!("durable mandate run token budgets are not supported by this store")
    }

    /// Atomically replace a dispatched call's pessimistic reservation with
    /// trusted actual usage and release the exact call lease. Returns
    /// `(tokens_used, budget_per_cycle)`.
    async fn settle_mandate_run_token_lease(
        &self,
        _goal_run_id: &str,
        _lease_token: &str,
        _delta_tokens: i64,
    ) -> anyhow::Result<(i64, i64)> {
        anyhow::bail!("durable mandate run token budgets are not supported by this store")
    }

    /// Release a call lease only when the caller proves provider dispatch never
    /// began. A stale/mismatched lease must not release a newer worker's call.
    async fn release_mandate_run_token_lease(
        &self,
        _goal_run_id: &str,
        _lease_token: &str,
    ) -> anyhow::Result<bool> {
        anyhow::bail!("durable mandate run token budgets are not supported by this store")
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

    /// Insert/update a behavior pattern occurrence.
    async fn record_behavior_pattern(
        &self,
        _pattern_type: &str,
        _description: &str,
        _trigger_context: Option<&str>,
        _action: Option<&str>,
        _confidence_hint: f32,
        _occurrence_delta: i32,
    ) -> anyhow::Result<()> {
        Ok(())
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

    /// Save or replace a pending interactive OAuth flow.
    async fn save_pending_oauth_flow(&self, _flow: &super::PendingOAuthFlow) -> anyhow::Result<()> {
        Ok(())
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

    /// Persist or clear the verified remote account bound to an OAuth service.
    async fn update_oauth_account_id(
        &self,
        _service: &str,
        _account_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Get a pending OAuth flow by state parameter.
    async fn get_pending_oauth_flow(
        &self,
        _state: &str,
    ) -> anyhow::Result<Option<super::PendingOAuthFlow>> {
        Ok(None)
    }

    /// List all pending OAuth flows.
    async fn list_pending_oauth_flows(&self) -> anyhow::Result<Vec<super::PendingOAuthFlow>> {
        Ok(vec![])
    }

    /// Delete an OAuth connection by service name.
    async fn delete_oauth_connection(&self, _service: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Delete a pending OAuth flow by state parameter.
    async fn delete_pending_oauth_flow(&self, _state: &str) -> anyhow::Result<()> {
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

/// Goal lifecycle and confirmation-flow persistence. Task, schedule, budget,
/// scheduled-run, dispatch, and notification concerns live in the sibling
/// traits: [`TaskStore`], [`GoalScheduleStore`], [`GoalBudgetStore`],
/// [`ScheduledRunStore`], [`TaskDispatchStore`], and [`GoalNotificationStore`].
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
    #[allow(dead_code)]
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
}

/// Owner mandates, agent deliberation cycles, and intentions.
#[async_trait]
pub trait MandateStore: Send + Sync {
    /// Atomically create the continuous controller goal and its mandate.
    /// Adaptive review timing lives on the mandate, not in `goal_schedules`.
    async fn create_mandate_controller(
        &self,
        _goal: &super::Goal,
        _mandate: &super::Mandate,
    ) -> anyhow::Result<()> {
        anyhow::bail!("mandate controllers are not supported by this store")
    }

    async fn get_mandate(&self, _id: &str) -> anyhow::Result<Option<super::Mandate>> {
        Ok(None)
    }

    async fn get_mandate_for_goal(&self, _goal_id: &str) -> anyhow::Result<Option<super::Mandate>> {
        Ok(None)
    }

    async fn list_mandates(
        &self,
        _session_id: Option<&str>,
        _include_terminal: bool,
    ) -> anyhow::Result<Vec<super::Mandate>> {
        Ok(Vec::new())
    }

    /// Optimistic owner-policy update. `mandate.version` must be exactly one
    /// greater than the currently stored version.
    async fn update_mandate(&self, _mandate: &super::Mandate) -> anyhow::Result<()> {
        anyhow::bail!("mandates are not supported by this store")
    }

    /// Atomically consume an explicit owner confirmation and activate both the
    /// mandate and its continuous controller. This is the only path that may
    /// turn an unconfirmed mandate into active authority.
    async fn confirm_mandate(
        &self,
        _mandate_id: &str,
        _expected_version: i64,
        _activation_duration_secs: Option<i64>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Atomic lifecycle/authority-epoch transition. Implementations must
    /// invalidate open decision runs and stale ACT authority before returning.
    async fn transition_mandate_status(
        &self,
        _mandate_id: &str,
        _from_status: super::MandateStatus,
        _to_status: super::MandateStatus,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Resume a paused/awaiting-input mandate and update its controller
    /// context in the same expected-version transaction. This prevents owner
    /// guidance from overwriting a concurrent cancel/pause lifecycle state.
    async fn resume_mandate_with_context(
        &self,
        _mandate_id: &str,
        _from_status: super::MandateStatus,
        _expected_version: i64,
        _controller_context: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Atomically create one mandate review run and its root task under the
    /// caller's current review lease. A failed root insert must
    /// leave no open run behind.
    async fn create_mandate_review_run(
        &self,
        _mandate_id: &str,
        _review_lease_token: &str,
        _goal_run_id: &str,
        _root_task: &super::Task,
    ) -> anyhow::Result<super::GoalRun> {
        anyhow::bail!("mandate review runs are not supported by this store")
    }

    /// Repair controller runtime state only while the exact mandate authority
    /// epoch is still active. The state check and goal update must be one
    /// atomic database statement so a racing owner pause/cancel wins.
    async fn keep_mandate_controller_active(
        &self,
        _mandate_id: &str,
        _expected_version: i64,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Claim due reviews with a durable lease. Concurrent callers must never
    /// receive the same mandate before the lease expires.
    async fn claim_due_mandates(
        &self,
        _limit: i64,
        _lease_owner: &str,
        _lease_secs: i64,
    ) -> anyhow::Result<Vec<super::Mandate>> {
        Ok(Vec::new())
    }

    async fn release_mandate_review_lease(
        &self,
        _mandate_id: &str,
        _lease_token: &str,
        _retry_at: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Insert one non-root action task into an exact mandate run while the
    /// dispatcher-owned root attempt remains current. Implementations must
    /// validate the mandate/run/attempt fence and task cap atomically and must
    /// never create or rebind a goal run by goal id.
    async fn create_mandate_task_from_attempt(
        &self,
        _task: &super::Task,
        _mandate_id: &str,
        _mandate_version: i64,
        _goal_run_id: &str,
        _root_task_attempt_id: &str,
        _max_non_root_tasks: i64,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Claim one pending non-root task inside an exact running mandate cycle.
    /// The root task-attempt remains the control-plane lease; the returned
    /// child attempt deliberately carries no worker-profile/workspace binding.
    /// The independently fenced identifiers remain explicit because wrapping
    /// them would churn the stable store API without reducing validation work.
    #[allow(clippy::too_many_arguments)]
    async fn claim_mandate_task_from_attempt(
        &self,
        _task_id: &str,
        _worker_instance_id: &str,
        _mandate_id: &str,
        _mandate_version: i64,
        _goal_run_id: &str,
        _root_task_attempt_id: &str,
        _lease_secs: i64,
    ) -> anyhow::Result<Option<super::TaskAttempt>> {
        Ok(None)
    }

    /// Insert exactly one deliberation result for a goal run, plus its ACT
    /// intention when present. Implementations must commit both atomically.
    async fn record_mandate_decision(
        &self,
        _decision: &super::MandateDecisionCycle,
        _intention: Option<&super::Intention>,
        _task_attempt_id: Option<&str>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("mandate decisions are not supported by this store")
    }

    async fn record_mandate_decision_with_updates(
        &self,
        decision: &super::MandateDecisionCycle,
        intention: Option<&super::Intention>,
        operating_updates: Option<&super::MandateOperatingUpdates>,
        task_attempt_id: Option<&str>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            operating_updates.is_none_or(|updates| {
                updates.learning_note.is_none() && updates.strategy_revisions.is_empty()
            }),
            "adaptive mandate operating updates are not supported by this store"
        );
        self.record_mandate_decision(decision, intention, task_attempt_id)
            .await
    }

    async fn get_mandate_decision_for_run(
        &self,
        _goal_run_id: &str,
    ) -> anyhow::Result<Option<super::MandateDecisionCycle>> {
        Ok(None)
    }

    async fn list_mandate_decisions(
        &self,
        _mandate_id: &str,
        _limit: i64,
    ) -> anyhow::Result<Vec<super::MandateDecisionCycle>> {
        Ok(Vec::new())
    }

    async fn list_intentions(
        &self,
        _mandate_id: &str,
        _limit: i64,
    ) -> anyhow::Result<Vec<super::Intention>> {
        Ok(Vec::new())
    }

    /// Persist one bounded, advisory learning note only after all cited
    /// receipts are proven to belong to this mandate.
    #[allow(dead_code)]
    async fn record_mandate_learning_note(
        &self,
        _note: &super::MandateLearningNote,
    ) -> anyhow::Result<()> {
        anyhow::bail!("mandate learning is not supported by this store")
    }

    async fn list_mandate_learning_notes(
        &self,
        _mandate_id: &str,
        _limit: i64,
    ) -> anyhow::Result<Vec<super::MandateLearningNote>> {
        Ok(Vec::new())
    }

    /// Return the latest adaptive strategy node for each key. Retired nodes
    /// remain visible for audit but must not be treated as active guidance.
    async fn list_current_mandate_strategy(
        &self,
        _mandate_id: &str,
        _limit: i64,
    ) -> anyhow::Result<Vec<super::MandateStrategyRevision>> {
        Ok(Vec::new())
    }

    /// Deduplicate and route one content-free external signal to matching
    /// Autopilot mandates. Matching is structural and cannot widen authority.
    async fn wake_mandates_for_signal(
        &self,
        _signal: &super::MandateWakeSignal,
    ) -> anyhow::Result<Vec<String>> {
        Ok(Vec::new())
    }

    /// Resolve an awaiting-input state with a typed owner action. Question
    /// answers and external-effect reconciliation are intentionally distinct.
    #[allow(clippy::too_many_arguments)]
    async fn resolve_mandate_suspension(
        &self,
        _mandate_id: &str,
        _expected_version: i64,
        _expected_kind: super::MandateSuspensionKind,
        _controller_context: Option<&str>,
        _reconciliation_resolution: Option<super::MandateReconciliationResolution>,
        _owner_guidance: &str,
        _owner_session: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Atomically reserve the exact next mutation attempt under the current
    /// cycle, rolling-24h, cooldown, mandate, run, intention, task, and lease
    /// fences. Implementations must insert the durable ledger row and advance
    /// the cycle counter in one transaction.
    async fn reserve_mandate_action_attempt(
        &self,
        _reservation: &super::MandateMutationReservation,
    ) -> anyhow::Result<Option<super::MandateMutationAttempt>> {
        Ok(None)
    }

    /// Atomically claim one exact reserved mutation at the last common
    /// dispatcher. A claim is one-use and must fail if any authority, run,
    /// task-attempt, tool-call, digest, or reservation field is stale.
    async fn claim_mandate_mutation_dispatch(
        &self,
        _claim: &super::MandateMutationDispatchClaim,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Read the typed, cross-cycle mutation quota projection at one instant.
    /// This is advisory for deliberation; reservation remains authoritative.
    async fn get_mandate_mutation_quota_state(
        &self,
        _mandate_id: &str,
        _as_of: &str,
    ) -> anyhow::Result<Option<super::MandateMutationQuotaState>> {
        Ok(None)
    }

    /// Project one strict common-dispatch receipt into an existing reservation.
    /// Only exact grant/task/call matches may transition a reserved row.
    async fn project_mandate_mutation_outcome(
        &self,
        _projection: &super::MandateMutationOutcomeProjection,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn list_mandate_mutation_attempts_for_run(
        &self,
        _goal_run_id: &str,
    ) -> anyhow::Result<Vec<super::MandateMutationAttempt>> {
        Ok(Vec::new())
    }

    /// Atomically prove and close a mandate run. Satisfied outcomes close the
    /// goal run (and ACT intention) inside the proof transaction; reconciliation
    /// results leave lifecycle state untouched for the caller's explicit path.
    async fn finalize_mandate_run_from_proof(
        &self,
        _request: &super::MandateRunFinalizationRequest,
    ) -> anyhow::Result<super::MandateRunFinalizationResult> {
        Ok(super::MandateRunFinalizationResult::Rejected {
            reason: super::MandateFinalizationRejectReason::InvalidRequest,
        })
    }
}

/// Task persistence and task activity logs.
#[async_trait]
pub trait TaskStore: Send + Sync {
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
    #[allow(dead_code)] // Retained for stores and diagnostics; run-scoped code filters task rows.
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

    /// Record a self-correction attempt (durable audit + repeat/K bookkeeping).
    #[allow(dead_code)] // Used in Phase 2
    async fn record_self_correction_attempt(
        &self,
        _attempt: &super::SelfCorrectionAttempt,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// All self-correction attempts for a subject, oldest first.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_self_correction_attempts(
        &self,
        _subject_id: &str,
    ) -> anyhow::Result<Vec<super::SelfCorrectionAttempt>> {
        Ok(vec![])
    }
}

/// Goal schedule and scheduled-run persistence.
#[async_trait]
pub trait GoalScheduleStore: Send + Sync {
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
}

/// Goal token budget persistence and accounting.
#[async_trait]
pub trait GoalBudgetStore: Send + Sync {
    /// Reset tokens_used_today only for active goals whose usage belongs to a
    /// prior UTC day. Safe to call repeatedly or at process startup.
    async fn reset_daily_token_budgets(&self) -> anyhow::Result<u64> {
        Ok(0)
    }

    /// Update budget columns only. `None` = keep current value (COALESCE), NOT "clear to NULL".
    /// This is a targeted UPDATE that avoids the race with `add_goal_tokens_and_get_budget_status()`
    /// which atomically increments `tokens_used_today` — a full `update_goal()` would clobber that.
    async fn set_goal_budgets(
        &self,
        _goal_id: &str,
        _budget_per_check: Option<i64>,
        _budget_daily: Option<i64>,
    ) -> anyhow::Result<()> {
        Ok(())
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
}

/// Scheduled run runtime-state persistence.
#[async_trait]
pub trait ScheduledRunStore: Send + Sync {
    /// Persist runtime state for an active scheduled run.
    async fn upsert_scheduled_run_state(
        &self,
        _state: &super::ScheduledRunState,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Load persisted runtime state for an active scheduled run.
    async fn get_scheduled_run_state(
        &self,
        _goal_id: &str,
    ) -> anyhow::Result<Option<super::ScheduledRunState>> {
        Ok(None)
    }

    /// Delete persisted runtime state for an active scheduled run.
    async fn delete_scheduled_run_state(&self, _goal_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }
}

/// Task dispatch bookkeeping for executor scheduling and recovery.
#[async_trait]
pub trait TaskDispatchStore: Send + Sync {
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
}

/// Durable coordination for goal runs, fenced task attempts, collaboration
/// records, named worker policies, project scope, and attempt workspaces.
#[async_trait]
pub trait WorkCoordinationStore: Send + Sync {
    async fn create_work_project(
        &self,
        _name: &str,
        _description: Option<&str>,
    ) -> anyhow::Result<super::WorkProject> {
        anyhow::bail!("work projects are not supported by this store")
    }

    async fn list_work_projects(&self) -> anyhow::Result<Vec<super::WorkProject>> {
        Ok(vec![])
    }

    async fn get_session_work_project(&self, _session_id: &str) -> anyhow::Result<String> {
        Ok(super::DEFAULT_PROJECT_ID.to_string())
    }

    async fn set_session_work_project(
        &self,
        _session_id: &str,
        _project_id: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn start_goal_run(
        &self,
        _goal_id: &str,
        _trigger_type: &str,
        _schedule_id: Option<&str>,
        _root_task_id: Option<&str>,
    ) -> anyhow::Result<super::GoalRun> {
        anyhow::bail!("goal runs are not supported by this store")
    }

    async fn get_current_goal_run(&self, _goal_id: &str) -> anyhow::Result<Option<super::GoalRun>> {
        Ok(None)
    }

    async fn get_goal_runs(&self, _goal_id: &str) -> anyhow::Result<Vec<super::GoalRun>> {
        Ok(vec![])
    }

    async fn get_tasks_for_goal_run(&self, _run_id: &str) -> anyhow::Result<Vec<super::Task>> {
        Ok(vec![])
    }

    async fn finish_goal_run(
        &self,
        _run_id: &str,
        _status: &str,
        _summary: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn claim_task_with_lease(
        &self,
        _task_id: &str,
        _worker_instance_id: &str,
        _worker_profile_id: Option<&str>,
        _lease_secs: i64,
    ) -> anyhow::Result<Option<super::TaskAttempt>> {
        Ok(None)
    }

    async fn get_current_task_attempt(
        &self,
        _task_id: &str,
    ) -> anyhow::Result<Option<super::TaskAttempt>> {
        Ok(None)
    }

    async fn bind_task_attempt_worker(
        &self,
        _attempt_id: &str,
        _lease_token: &str,
        _worker_instance_id: &str,
        _worker_profile_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn heartbeat_task_attempt(
        &self,
        _attempt_id: &str,
        _lease_token: &str,
        _lease_secs: i64,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn patch_task_from_attempt(
        &self,
        _attempt_id: &str,
        _lease_token: &str,
        _patch: &super::TaskAttemptPatch,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Expire elapsed leases and return the affected task IDs. Idempotent tasks
    /// with retry budget are re-queued; ambiguous writes require verification.
    async fn recover_expired_task_attempts(&self) -> anyhow::Result<Vec<String>> {
        Ok(vec![])
    }

    async fn append_task_journal(&self, _entry: &super::TaskJournalEntry) -> anyhow::Result<()> {
        Ok(())
    }

    async fn get_task_journal(
        &self,
        _task_id: &str,
        _limit: i64,
    ) -> anyhow::Result<Vec<super::TaskJournalEntry>> {
        Ok(vec![])
    }

    async fn unblock_task(
        &self,
        _task_id: &str,
        _resolution: &str,
        _actor_id: &str,
        _source_channel: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn retry_work_task(
        &self,
        _task_id: &str,
        _actor_id: &str,
        _source_channel: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn cancel_work_task(
        &self,
        _task_id: &str,
        _actor_id: &str,
        _source_channel: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn upsert_worker_profile(&self, _profile: &super::WorkerProfile) -> anyhow::Result<()> {
        Ok(())
    }

    async fn get_worker_profile(
        &self,
        _profile_id: &str,
    ) -> anyhow::Result<Option<super::WorkerProfile>> {
        Ok(None)
    }

    async fn list_worker_profiles(
        &self,
        _project_id: Option<&str>,
    ) -> anyhow::Result<Vec<super::WorkerProfile>> {
        Ok(vec![])
    }

    async fn assign_task_worker_profile(
        &self,
        _task_id: &str,
        _profile_id: &str,
        _actor_id: &str,
        _source_channel: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn set_task_workspace_policy(
        &self,
        _task_id: &str,
        _policy: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn get_task_workspace_policy(&self, _task_id: &str) -> anyhow::Result<String> {
        Ok("shared".to_string())
    }

    async fn create_task_workspace(&self, _workspace: &super::TaskWorkspace) -> anyhow::Result<()> {
        Ok(())
    }

    async fn update_task_workspace(&self, _workspace: &super::TaskWorkspace) -> anyhow::Result<()> {
        Ok(())
    }

    async fn get_task_workspace(
        &self,
        _task_id: &str,
    ) -> anyhow::Result<Option<super::TaskWorkspace>> {
        Ok(None)
    }

    async fn get_latest_task_handoff(
        &self,
        _task_id: &str,
    ) -> anyhow::Result<Option<super::TaskHandoff>> {
        Ok(None)
    }

    async fn list_work_goals(
        &self,
        _project_id: &str,
        _include_terminal: bool,
        _limit: i64,
    ) -> anyhow::Result<Vec<super::WorkGoalSummary>> {
        Ok(vec![])
    }

    async fn list_work_tasks(
        &self,
        _project_id: &str,
        _lane: Option<&str>,
        _limit: i64,
    ) -> anyhow::Result<Vec<super::WorkTaskSummary>> {
        Ok(vec![])
    }
}

/// Goal lifecycle cleanup and user notification bookkeeping.
#[async_trait]
pub trait GoalNotificationStore: Send + Sync {
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

    /// Atomically claim a goal's notification slot and enqueue its notification.
    ///
    /// Returns `false` when another writer has already claimed the goal. Stores
    /// that do not support transactions may fall back to enqueueing only.
    async fn enqueue_goal_notification(
        &self,
        entry: &super::NotificationEntry,
    ) -> anyhow::Result<bool> {
        self.enqueue_notification(entry).await?;
        Ok(true)
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

/// Facade trait kept for backwards compatibility.
///
/// This lets call sites keep using `Arc<dyn StateStore>`, while new code can
/// depend on focused store traits like `FactStore` or `PeopleStore`.
pub trait StateStore:
    Send
    + Sync
    + MessageStore
    + DialogueStateStore
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
