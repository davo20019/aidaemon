//! The memory traits on the state store, answered by its [`SqliteMemory`].
//!
//! Callers still reach memory through `StateStore`; these forwards go once
//! they hold the memory stores directly.

use aidaemon_core::ChannelVisibility;
use aidaemon_memory::records::*;
use aidaemon_memory::stores::*;
use aidaemon_memory::{
    BehaviorPattern, ErrorSolution, LearningStore, Procedure, Result, UserProfile,
};
use async_trait::async_trait;
use std::collections::HashSet;

use super::SqliteStateStore;

#[async_trait]
impl FactStore for SqliteStateStore {
    async fn reconcile_personal_memory(
        &self,
        write: &PersonalMemoryWrite,
        source: &str,
        source_excerpt: Option<&str>,
        channel_id: Option<&str>,
        privacy: FactPrivacy,
    ) -> anyhow::Result<PersonalMemoryWriteResult> {
        FactStore::reconcile_personal_memory(
            &self.memory,
            write,
            source,
            source_excerpt,
            channel_id,
            privacy,
        )
        .await
    }
    async fn get_canonical_memory_facts(&self) -> anyhow::Result<Vec<Fact>> {
        FactStore::get_canonical_memory_facts(&self.memory).await
    }
    async fn record_memory_recall(
        &self,
        fact_ids: &[i64],
        episode_ids: &[i64],
    ) -> anyhow::Result<()> {
        FactStore::record_memory_recall(&self.memory, fact_ids, episode_ids).await
    }
    async fn fold_restated_facts(&self) -> anyhow::Result<usize> {
        FactStore::fold_restated_facts(&self.memory).await
    }
    async fn upsert_fact(
        &self,
        category: &str,
        key: &str,
        value: &str,
        source: &str,
        channel_id: Option<&str>,
        privacy: FactPrivacy,
    ) -> anyhow::Result<()> {
        FactStore::upsert_fact(
            &self.memory,
            category,
            key,
            value,
            source,
            channel_id,
            privacy,
        )
        .await
    }
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
    ) -> anyhow::Result<()> {
        FactStore::upsert_fact_with_provenance(
            &self.memory,
            category,
            key,
            value,
            source,
            channel_id,
            privacy,
            first_seen_at,
            source_excerpt,
        )
        .await
    }
    async fn get_facts(&self, category: Option<&str>) -> anyhow::Result<Vec<Fact>> {
        FactStore::get_facts(&self.memory, category).await
    }
    async fn get_relevant_facts(&self, query: &str, max: usize) -> anyhow::Result<Vec<Fact>> {
        FactStore::get_relevant_facts(&self.memory, query, max).await
    }
    async fn get_relevant_facts_for_channel(
        &self,
        query: &str,
        max: usize,
        channel_id: Option<&str>,
        visibility: ChannelVisibility,
        requester_is_owner: bool,
    ) -> anyhow::Result<Vec<Fact>> {
        FactStore::get_relevant_facts_for_channel(
            &self.memory,
            query,
            max,
            channel_id,
            visibility,
            requester_is_owner,
        )
        .await
    }
    async fn update_fact_privacy(&self, fact_id: i64, privacy: FactPrivacy) -> anyhow::Result<()> {
        FactStore::update_fact_privacy(&self.memory, fact_id, privacy).await
    }
    async fn delete_fact(&self, fact_id: i64) -> anyhow::Result<()> {
        FactStore::delete_fact(&self.memory, fact_id).await
    }
    async fn delete_fact_by_key(&self, category: &str, key: &str) -> anyhow::Result<bool> {
        FactStore::delete_fact_by_key(&self.memory, category, key).await
    }
    async fn get_all_facts_with_provenance(&self) -> anyhow::Result<Vec<Fact>> {
        FactStore::get_all_facts_with_provenance(&self.memory).await
    }
    async fn search_facts_semantic(
        &self,
        query: &str,
        max: usize,
    ) -> anyhow::Result<Vec<(Fact, f32)>> {
        FactStore::search_facts_semantic(&self.memory, query, max).await
    }
    async fn project_extracted_fact_graph(
        &self,
        category: &str,
        key: &str,
        source_excerpt: &str,
        graph: &ExtractedMemoryGraph,
    ) -> anyhow::Result<()> {
        FactStore::project_extracted_fact_graph(&self.memory, category, key, source_excerpt, graph)
            .await
    }
    async fn refresh_fact_memory(&self, category: &str, key: &str) -> anyhow::Result<()> {
        FactStore::refresh_fact_memory(&self.memory, category, key).await
    }
    async fn memory_health_report(&self) -> anyhow::Result<MemoryHealthReport> {
        FactStore::memory_health_report(&self.memory).await
    }
    async fn repair_memory_projections(&self) -> anyhow::Result<MemoryHealthReport> {
        FactStore::repair_memory_projections(&self.memory).await
    }
    async fn assemble_neighborhood(
        &self,
        entities: &[String],
        initial_ids: &HashSet<i64>,
    ) -> anyhow::Result<Vec<Fact>> {
        FactStore::assemble_neighborhood(&self.memory, entities, initial_ids).await
    }
}

#[async_trait]
impl EpisodeStore for SqliteStateStore {
    async fn project_episode_memory(&self, episode_id: i64) -> anyhow::Result<()> {
        EpisodeStore::project_episode_memory(&self.memory, episode_id).await
    }
    async fn get_relevant_episodes(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Episode>> {
        EpisodeStore::get_relevant_episodes(&self.memory, query, limit).await
    }
    async fn get_relevant_episodes_for_channel(
        &self,
        query: &str,
        limit: usize,
        channel_id: Option<&str>,
    ) -> anyhow::Result<Vec<Episode>> {
        EpisodeStore::get_relevant_episodes_for_channel(&self.memory, query, limit, channel_id)
            .await
    }
    async fn get_relevant_episodes_for_session(
        &self,
        query: &str,
        limit: usize,
        session_id: &str,
    ) -> anyhow::Result<Vec<Episode>> {
        EpisodeStore::get_relevant_episodes_for_session(&self.memory, query, limit, session_id)
            .await
    }
    async fn record_session_episode(
        &self,
        episode: &aidaemon_memory::episodes::SessionEpisode,
    ) -> anyhow::Result<i64> {
        EpisodeStore::record_session_episode(&self.memory, episode).await
    }
}

#[async_trait]
impl PeopleStore for SqliteStateStore {
    async fn upsert_person(&self, person: &Person) -> anyhow::Result<i64> {
        PeopleStore::upsert_person(&self.memory, person).await
    }
    async fn get_person(&self, id: i64) -> anyhow::Result<Option<Person>> {
        PeopleStore::get_person(&self.memory, id).await
    }
    async fn get_person_by_platform_id(&self, platform_id: &str) -> anyhow::Result<Option<Person>> {
        PeopleStore::get_person_by_platform_id(&self.memory, platform_id).await
    }
    async fn find_person_by_name(&self, name: &str) -> anyhow::Result<Option<Person>> {
        PeopleStore::find_person_by_name(&self.memory, name).await
    }
    async fn get_all_people(&self) -> anyhow::Result<Vec<Person>> {
        PeopleStore::get_all_people(&self.memory).await
    }
    async fn delete_person(&self, id: i64) -> anyhow::Result<()> {
        PeopleStore::delete_person(&self.memory, id).await
    }
    async fn link_platform_id(
        &self,
        person_id: i64,
        platform_id: &str,
        display_name: &str,
    ) -> anyhow::Result<()> {
        PeopleStore::link_platform_id(&self.memory, person_id, platform_id, display_name).await
    }
    async fn touch_person_interaction(&self, person_id: i64) -> anyhow::Result<()> {
        PeopleStore::touch_person_interaction(&self.memory, person_id).await
    }
    async fn upsert_person_fact(
        &self,
        person_id: i64,
        category: &str,
        key: &str,
        value: &str,
        source: &str,
        confidence: f32,
    ) -> anyhow::Result<()> {
        PeopleStore::upsert_person_fact(
            &self.memory,
            person_id,
            category,
            key,
            value,
            source,
            confidence,
        )
        .await
    }
    async fn get_person_facts(
        &self,
        person_id: i64,
        category: Option<&str>,
    ) -> anyhow::Result<Vec<PersonFact>> {
        PeopleStore::get_person_facts(&self.memory, person_id, category).await
    }
    async fn delete_person_fact(&self, fact_id: i64) -> anyhow::Result<()> {
        PeopleStore::delete_person_fact(&self.memory, fact_id).await
    }
    async fn confirm_person_fact(&self, fact_id: i64) -> anyhow::Result<()> {
        PeopleStore::confirm_person_fact(&self.memory, fact_id).await
    }
    async fn get_people_with_upcoming_dates(
        &self,
        within_days: i32,
    ) -> anyhow::Result<Vec<(Person, PersonFact)>> {
        PeopleStore::get_people_with_upcoming_dates(&self.memory, within_days).await
    }
    async fn prune_stale_person_facts(&self, retention_days: u32) -> anyhow::Result<u64> {
        PeopleStore::prune_stale_person_facts(&self.memory, retention_days).await
    }
    async fn get_people_needing_reconnect(
        &self,
        inactive_days: u32,
    ) -> anyhow::Result<Vec<Person>> {
        PeopleStore::get_people_needing_reconnect(&self.memory, inactive_days).await
    }
}

#[async_trait]
impl MemoryGraphStore for SqliteStateStore {
    async fn upsert_entity(&self, entity: GraphEntityUpsert) -> anyhow::Result<i64> {
        MemoryGraphStore::upsert_entity(&self.memory, entity).await
    }
    async fn upsert_edge(&self, edge: GraphEdgeUpsert) -> anyhow::Result<()> {
        MemoryGraphStore::upsert_edge(&self.memory, edge).await
    }
    async fn invalidate_claim(&self, claim_id: i64, valid_to: &str) -> anyhow::Result<()> {
        MemoryGraphStore::invalidate_claim(&self.memory, claim_id, valid_to).await
    }
    async fn neighbors(
        &self,
        start_entity_id: i64,
        max_depth: usize,
        scope: &GraphSearchScope,
    ) -> anyhow::Result<Vec<GraphNeighbor>> {
        MemoryGraphStore::neighbors(&self.memory, start_entity_id, max_depth, scope).await
    }
}

#[async_trait]
impl MemoryDecay for SqliteStateStore {
    async fn decay(
        &self,
        policy: &DecayPolicy,
        now: chrono::DateTime<chrono::Utc>,
    ) -> anyhow::Result<()> {
        MemoryDecay::decay(&self.memory, policy, now).await
    }
}

#[async_trait]
impl LearningStore for SqliteStateStore {
    async fn get_behavior_patterns(&self, min_confidence: f32) -> Result<Vec<BehaviorPattern>> {
        LearningStore::get_behavior_patterns(&self.memory, min_confidence).await
    }
    async fn record_behavior_pattern(
        &self,
        pattern_type: &str,
        description: &str,
        trigger_context: Option<&str>,
        action: Option<&str>,
        confidence_hint: f32,
        occurrence_delta: i32,
    ) -> Result<()> {
        LearningStore::record_behavior_pattern(
            &self.memory,
            pattern_type,
            description,
            trigger_context,
            action,
            confidence_hint,
            occurrence_delta,
        )
        .await
    }
    async fn get_relevant_procedures(&self, query: &str, limit: usize) -> Result<Vec<Procedure>> {
        LearningStore::get_relevant_procedures(&self.memory, query, limit).await
    }
    async fn get_relevant_error_solutions(
        &self,
        error: &str,
        limit: usize,
    ) -> Result<Vec<ErrorSolution>> {
        LearningStore::get_relevant_error_solutions(&self.memory, error, limit).await
    }
    async fn get_user_profile(&self) -> Result<Option<UserProfile>> {
        LearningStore::get_user_profile(&self.memory).await
    }
    async fn update_user_profile(&self, profile: &UserProfile) -> Result<()> {
        LearningStore::update_user_profile(&self.memory, profile).await
    }
    async fn increment_expertise(
        &self,
        domain: &str,
        success: bool,
        error: Option<&str>,
    ) -> Result<()> {
        LearningStore::increment_expertise(&self.memory, domain, success, error).await
    }
    async fn upsert_procedure(&self, procedure: &Procedure) -> Result<i64> {
        LearningStore::upsert_procedure(&self.memory, procedure).await
    }
    async fn update_procedure_outcome(
        &self,
        procedure_id: i64,
        success: bool,
        duration: Option<f32>,
    ) -> Result<()> {
        LearningStore::update_procedure_outcome(&self.memory, procedure_id, success, duration).await
    }
    async fn insert_error_solution(&self, solution: &ErrorSolution) -> Result<i64> {
        LearningStore::insert_error_solution(&self.memory, solution).await
    }
    async fn update_error_solution_outcome(&self, solution_id: i64, success: bool) -> Result<()> {
        LearningStore::update_error_solution_outcome(&self.memory, solution_id, success).await
    }
}
