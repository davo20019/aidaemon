//! The records of the daemon-shaped memory contract ([`crate::stores`]):
//! facts with privacy and channel provenance, extraction results, episodes,
//! people, and exact conversation history.

use std::collections::HashMap;

use aidaemon_core::{ChannelVisibility, UserRole};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

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

    /// Whether moving a fact from `self` to `to` lets more chats see it.
    pub fn widens_to(self, to: Self) -> bool {
        fn reach(privacy: FactPrivacy) -> u8 {
            match privacy {
                FactPrivacy::Private => 0,
                FactPrivacy::Channel => 1,
                FactPrivacy::Global => 2,
            }
        }
        reach(to) > reach(self)
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
    /// The writer asks to forget this predicate rather than state a value.
    /// Only a direct owner statement retracts; `value` is ignored.
    #[serde(default)]
    pub retract: bool,
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

/// Where an episode stopped: the last event it covers (recent episodes) or
/// its end time (older ones, which recorded no event bound), as stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeBound {
    pub end_event_id: Option<i64>,
    pub end_time: Option<String>,
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

/// Who may see which conversation history in a search.
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

/// One completed tool call of a retained turn. Fresh tasks do not see these in
/// their transcript, so this is how they read what an earlier turn observed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryToolActivity {
    pub tool: String,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<serde_json::Value>,
    pub output: String,
    /// Characters of `output` left out to fit the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_omitted_chars: Option<usize>,
}

/// A recent turn of a session that ran tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecentToolTurn {
    pub turn_id: String,
    /// How many user turns back this one is; the newest prior turn is 1.
    pub turns_ago: usize,
    /// Tool name and how many calls of it completed, in first-use order.
    pub tools: Vec<(String, usize)>,
    /// Tool name and how many calls of it were refused before running, in
    /// first-use order.
    pub refused: Vec<(String, usize)>,
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
