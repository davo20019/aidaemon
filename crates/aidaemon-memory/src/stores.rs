//! The daemon-shaped memory storage contract: facts (with privacy, channel
//! provenance and the extraction graph), episodes, exact history and
//! people. A store implements the slices it supports; every method has a
//! conservative default.

use std::collections::HashSet;

use aidaemon_core::ChannelVisibility;
use async_trait::async_trait;

use crate::records::*;

/// Exact-history operations exposed to the search tool through a domain port.
#[async_trait]
pub trait HistorySearchStore: EpisodeStore {
    async fn history_snapshot_max_event_id(&self) -> anyhow::Result<i64>;
    async fn history_coverage(&self) -> anyhow::Result<HistoryCoverage>;
    async fn repair_history_projection(
        &self,
        max_batches: usize,
    ) -> anyhow::Result<ProjectionStats>;
    async fn search_history(
        &self,
        query: &str,
        scope: &HistoryScope,
        limit: usize,
        semantic_sessions: &HashSet<String>,
    ) -> anyhow::Result<Vec<HistoryMessage>>;
    async fn history_context(
        &self,
        event_id: i64,
        radius: usize,
        scope: &HistoryScope,
    ) -> anyhow::Result<Vec<HistoryMessage>>;
    async fn history_event_for_message_id(
        &self,
        message_id: &str,
        scope: &HistoryScope,
    ) -> anyhow::Result<Option<i64>>;
    async fn history_turn(
        &self,
        turn_id: &str,
        scope: &HistoryScope,
    ) -> anyhow::Result<Vec<HistoryMessage>>;
    async fn history_page(
        &self,
        anchor: i64,
        older: bool,
        scope: &HistoryScope,
        limit: usize,
    ) -> anyhow::Result<Vec<HistoryMessage>>;
    async fn history_task_bookends(
        &self,
        task_id: Option<&str>,
        session_id: &str,
        scope: &HistoryScope,
    ) -> anyhow::Result<TaskBookends>;
}

/// Layer-2 facts storage and retrieval (including privacy + channel provenance).
#[async_trait]
pub trait FactStore: Send + Sync {
    /// Reconcile an entity-aware personal-memory write atomically.
    async fn reconcile_personal_memory(
        &self,
        _write: &PersonalMemoryWrite,
        _source: &str,
        _source_excerpt: Option<&str>,
        _channel_id: Option<&str>,
        _privacy: FactPrivacy,
    ) -> anyhow::Result<PersonalMemoryWriteResult> {
        anyhow::bail!("structured personal memory is not supported by this store")
    }

    /// Compatibility projection of active canonical facts and relationships.
    async fn get_canonical_memory_facts(&self) -> anyhow::Result<Vec<Fact>> {
        Ok(Vec::new())
    }

    /// Record that these facts and episodes were shown to a model. Recall
    /// counts drive Core Profile salience and decay, so only content that
    /// actually reached a prompt or a tool result may count; fetching a
    /// candidate that was then cut by a budget is not a recall.
    async fn record_memory_recall(
        &self,
        _fact_ids: &[i64],
        _episode_ids: &[i64],
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Supersede active automated facts that restate another active fact in
    /// the same category, keeping the most authoritative copy. Returns how
    /// many rows were folded. Soft: superseded rows remain as history.
    async fn fold_restated_facts(&self) -> anyhow::Result<usize> {
        Ok(0)
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
    async fn get_facts(&self, category: Option<&str>) -> anyhow::Result<Vec<Fact>>;

    /// Get facts semantically relevant to a query, falling back to `get_facts` on error.
    async fn get_relevant_facts(&self, _query: &str, max: usize) -> anyhow::Result<Vec<Fact>> {
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
    ) -> anyhow::Result<Vec<Fact>> {
        self.get_relevant_facts(query, max).await
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
    async fn get_all_facts_with_provenance(&self) -> anyhow::Result<Vec<Fact>> {
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
    ) -> anyhow::Result<Vec<(Fact, f32)>> {
        Ok(vec![])
    }

    /// Persist a model-extracted semantic graph for an already-written fact.
    /// Implementations must validate the graph and retain claim provenance.
    async fn project_extracted_fact_graph(
        &self,
        _category: &str,
        _key: &str,
        _source_excerpt: &str,
        _graph: &ExtractedMemoryGraph,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Refresh canonical claim/vector projections for the active fact at this key.
    async fn refresh_fact_memory(&self, _category: &str, _key: &str) -> anyhow::Result<()> {
        Ok(())
    }

    async fn memory_health_report(&self) -> anyhow::Result<MemoryHealthReport> {
        Ok(MemoryHealthReport::default())
    }

    async fn repair_memory_projections(&self) -> anyhow::Result<MemoryHealthReport> {
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
        _initial_ids: &HashSet<i64>,
    ) -> anyhow::Result<Vec<Fact>> {
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
    ) -> anyhow::Result<Vec<Episode>> {
        Ok(vec![])
    }

    /// Get episodes for a specific channel context.
    async fn get_relevant_episodes_for_channel(
        &self,
        _query: &str,
        _limit: usize,
        _channel_id: Option<&str>,
    ) -> anyhow::Result<Vec<Episode>> {
        Ok(vec![])
    }

    /// Get episodes only from one canonical session (used by untrusted
    /// internal/sub-agent origins).
    async fn get_relevant_episodes_for_session(
        &self,
        _query: &str,
        _limit: usize,
        _session_id: &str,
    ) -> anyhow::Result<Vec<Episode>> {
        Ok(vec![])
    }
}

/// People persistence (social graph).
#[async_trait]
pub trait PeopleStore: Send + Sync {
    /// Create or update a person record. Returns the person ID.
    async fn upsert_person(&self, _person: &Person) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get a person by their database ID.
    async fn get_person(&self, _id: i64) -> anyhow::Result<Option<Person>> {
        Ok(None)
    }

    /// Look up a person by a platform-qualified sender ID (e.g., "slack:U123").
    async fn get_person_by_platform_id(
        &self,
        _platform_id: &str,
    ) -> anyhow::Result<Option<Person>> {
        Ok(None)
    }

    /// Find a person by name or alias (case-insensitive).
    async fn find_person_by_name(&self, _name: &str) -> anyhow::Result<Option<Person>> {
        Ok(None)
    }

    /// Get all people.
    async fn get_all_people(&self) -> anyhow::Result<Vec<Person>> {
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
    ) -> anyhow::Result<Vec<PersonFact>> {
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
    ) -> anyhow::Result<Vec<(Person, PersonFact)>> {
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
    ) -> anyhow::Result<Vec<Person>> {
        Ok(vec![])
    }
}
