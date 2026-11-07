//! Narrow store services that tools hold: settings, facts and episodes,
//! exact conversation history, people, and OAuth connections, plus the plain
//! data those traits carry. The daemon's SQLite store implements them; a tool
//! receives an `Arc<dyn ...>` of the slice it needs through its constructor.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::semantics::UserRole;

/// The conversation's audience, shared across the SDK.
pub use aidaemon_core::ChannelVisibility;

/// Privacy level for facts stored in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactPrivacy {
    /// Accessible everywhere (user's name, timezone, general preferences).
    Global,
    /// Accessible only in originating channel + DMs.
    Channel,
    /// DM-only, never shared in channels, never hinted.
    Private,
}

impl std::fmt::Display for FactPrivacy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FactPrivacy::Global => write!(f, "global"),
            FactPrivacy::Channel => write!(f, "channel"),
            FactPrivacy::Private => write!(f, "private"),
        }
    }
}

impl FactPrivacy {
    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "global" => Self::Global,
            "channel" => Self::Channel,
            "private" => Self::Private,
            _ => Self::Global,
        }
    }
}

/// A fact stored in Layer 2 memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id: i64,
    pub category: String,
    pub key: String,
    pub value: String,
    pub source: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub recall_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_recalled_at: Option<DateTime<Utc>>,
    /// Channel where this fact originated (e.g., "slack:C12345"). None for legacy/global facts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel_id: Option<String>,
    /// Privacy level controlling where this fact can be recalled.
    #[serde(default = "default_fact_privacy")]
    pub privacy: FactPrivacy,
    /// When the fact was originally stated or inferred in the event stream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_seen_at: Option<DateTime<Utc>>,
    /// The exact snippet of conversation that generated this fact.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_excerpt: Option<String>,
}

fn default_fact_privacy() -> FactPrivacy {
    FactPrivacy::Global
}

/// An entity proposed by the memory extraction model. `local_id` is scoped to
/// one extraction result and is used by relationships to avoid name ambiguity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedMemoryEntity {
    pub local_id: String,
    pub name: String,
    pub entity_type: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default = "default_extraction_confidence")]
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedMemoryRelationship {
    pub source_id: String,
    pub target_id: String,
    pub relation: String,
    #[serde(default = "default_extraction_confidence")]
    pub confidence: f32,
}

/// Optional semantic graph attached to a fact extraction. Fact persistence is
/// independent: invalid graph output is ignored rather than poisoning memory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractedMemoryGraph {
    #[serde(default)]
    pub entities: Vec<ExtractedMemoryEntity>,
    #[serde(default)]
    pub relationships: Vec<ExtractedMemoryRelationship>,
}

fn default_extraction_confidence() -> f32 {
    0.5
}

/// A canonical entity proposed by the personal-memory extractor.
///
/// `local_id` only links records inside one write. `is_reference` distinguishes
/// a mention of an already-known alias from a declaration of a new entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalEntityCandidate {
    pub local_id: String,
    pub entity_type: String,
    pub canonical_name: String,
    #[serde(default)]
    pub is_reference: bool,
    #[serde(default)]
    pub canonical_name_confirmed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalAliasCandidate {
    pub entity_local_id: String,
    pub value: String,
    pub alias_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalFactCandidate {
    pub subject_local_id: String,
    pub predicate: String,
    pub value: String,
    #[serde(default)]
    pub display_value: Option<String>,
    #[serde(default)]
    pub valid_from: Option<String>,
    #[serde(default)]
    pub valid_to: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalRelationshipCandidate {
    pub source_local_id: String,
    pub relationship_type: String,
    pub target_local_id: String,
    #[serde(default)]
    pub valid_from: Option<String>,
    #[serde(default)]
    pub valid_to: Option<String>,
}

/// One atomic, entity-aware personal-memory write plan.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersonalMemoryWrite {
    #[serde(default)]
    pub entities: Vec<PersonalEntityCandidate>,
    #[serde(default)]
    pub aliases: Vec<PersonalAliasCandidate>,
    #[serde(default)]
    pub facts: Vec<PersonalFactCandidate>,
    #[serde(default)]
    pub relationships: Vec<PersonalRelationshipCandidate>,
    /// True only when the owner directly supplied the information.
    #[serde(default)]
    pub direct_user_statement: bool,
    /// True when the owner explicitly corrected a previously stored value.
    #[serde(default)]
    pub correction: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersonalMemoryWriteResult {
    pub created_entities: usize,
    pub updated_entities: usize,
    pub created_aliases: usize,
    pub confirmed_aliases: usize,
    pub created_facts: usize,
    pub confirmed_facts: usize,
    pub superseded_facts: usize,
    pub disputed_facts: usize,
    pub created_relationships: usize,
    pub confirmed_relationships: usize,
    #[serde(default)]
    pub unresolved: Vec<String>,
}

impl PersonalMemoryWriteResult {
    pub fn concise_summary(&self) -> String {
        let mut parts = Vec::new();
        for (count, label) in [
            (self.created_entities, "entities created"),
            (self.updated_entities, "entities updated"),
            (self.created_aliases, "aliases created"),
            (self.confirmed_aliases, "aliases confirmed"),
            (self.created_facts, "facts created"),
            (self.confirmed_facts, "facts confirmed"),
            (self.superseded_facts, "facts superseded"),
            (self.disputed_facts, "facts disputed"),
            (self.created_relationships, "relationships created"),
            (self.confirmed_relationships, "relationships confirmed"),
        ] {
            if count > 0 {
                parts.push(format!("{count} {label}"));
            }
        }
        if !self.unresolved.is_empty() {
            parts.push(format!("{} unresolved", self.unresolved.len()));
        }
        if parts.is_empty() {
            "No changes (already current).".to_string()
        } else {
            parts.join(", ")
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryHealthReport {
    pub spans: i64,
    pub active_claims: i64,
    pub entities: i64,
    pub active_edges: i64,
    pub active_embeddings: i64,
    pub stale_embeddings: i64,
    pub facts_missing_claims: i64,
    pub episodes_missing_spans: i64,
    pub orphan_edges: i64,
    pub embedding_dimension_mismatches: i64,
}

/// An episode representing a session summary (episodic memory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: i64,
    pub session_id: String,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topics: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotional_tone: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    pub importance: f32,
    pub recall_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_recalled_at: Option<DateTime<Utc>>,
    pub message_count: i32,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    /// Channel where this episode occurred. None for legacy episodes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel_id: Option<String>,
}

/// A person in the owner's social circle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Person {
    pub id: i64,
    pub name: String,
    pub aliases: Vec<String>,
    pub relationship: Option<String>,
    pub platform_ids: HashMap<String, String>,
    pub notes: Option<String>,
    pub communication_style: Option<String>,
    pub language_preference: Option<String>,
    pub last_interaction_at: Option<DateTime<Utc>>,
    pub interaction_count: i64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A fact about a person (birthday, preference, interest, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonFact {
    pub id: i64,
    pub person_id: i64,
    pub category: String,
    pub key: String,
    pub value: String,
    pub source: String,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// An OAuth connection to an external service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthConnection {
    pub id: i64,
    pub service: String,
    pub auth_type: String,
    pub username: Option<String>,
    /// Stable remote account identifier proven through an authenticated
    /// identity endpoint. Mandate execution binds to this value rather than a
    /// mutable provider alias or display name.
    pub account_id: Option<String>,
    pub scopes: String,
    pub token_expires_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// A pending interactive OAuth flow awaiting browser callback completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingOAuthFlow {
    pub state: String,
    pub service: String,
    pub code_verifier: Option<String>,
    pub session_id: String,
    pub created_at: String,
}

/// A dynamically added skill (stored in database).

#[derive(Debug, Clone)]
pub struct HistoryScope {
    pub session_id: String,
    pub channel_id: Option<String>,
    pub visibility: ChannelVisibility,
    pub user_role: UserRole,
    pub trusted: bool,
    pub include_subagents: bool,
    pub session_filter: Option<String>,
    pub task_filter: Option<String>,
    pub snapshot_max_event_id: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HistoryMessage {
    pub event_id: i64,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    pub role: String,
    pub content: String,
    pub created_at: String,
    pub source_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical_rank: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskBookends {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objective: Option<HistoryMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_objective: Option<String>,
    pub objective_source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<HistoryMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_resolution: Option<String>,
    pub resolution_source: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ProjectionStats {
    pub projected: u64,
    pub orphans_removed: u64,
    pub fts_rebuilt: bool,
    pub episodes_repaired: u64,
    pub pending: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HistoryCoverage {
    pub canonical_messages: i64,
    pub indexed_messages: i64,
    pub pending_messages: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oldest_indexed_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub newest_indexed_at: Option<String>,
}

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

/// OAuth-connected external services persistence.
#[async_trait]
pub trait OAuthStore: Send + Sync {
    /// Save an OAuth connection. Returns the connection ID.
    async fn save_oauth_connection(&self, _conn: &OAuthConnection) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Save or replace a pending interactive OAuth flow.
    async fn save_pending_oauth_flow(&self, _flow: &PendingOAuthFlow) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get an OAuth connection by service name.
    async fn get_oauth_connection(
        &self,
        _service: &str,
    ) -> anyhow::Result<Option<OAuthConnection>> {
        Ok(None)
    }

    /// List all OAuth connections.
    async fn list_oauth_connections(&self) -> anyhow::Result<Vec<OAuthConnection>> {
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
    ) -> anyhow::Result<Option<PendingOAuthFlow>> {
        Ok(None)
    }

    /// List all pending OAuth flows.
    async fn list_pending_oauth_flows(&self) -> anyhow::Result<Vec<PendingOAuthFlow>> {
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

/// Focused persistence contract used by people management.
pub trait PeopleToolStore: PeopleStore + FactStore + SettingsStore {}

impl<T> PeopleToolStore for T where T: PeopleStore + FactStore + SettingsStore {}

/// OAuth flows only need OAuth connection and pending-flow persistence.
pub trait OAuthGatewayStore: OAuthStore {}

impl<T> OAuthGatewayStore for T where T: OAuthStore {}
