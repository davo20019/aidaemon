use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::types::FactPrivacy;

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

/// User communication style preferences.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UserProfile {
    pub id: i64,
    pub verbosity_preference: String, // "brief", "medium", "detailed"
    pub explanation_depth: String,    // "minimal", "moderate", "thorough"
    pub tone_preference: String,      // "casual", "neutral", "formal"
    pub emoji_preference: String,     // "none", "minimal", "frequent"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typical_session_length: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_hours: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub common_workflows: Option<Vec<String>>,
    pub asks_before_acting: bool,
    pub prefers_explanations: bool,
    pub likes_suggestions: bool,
    pub updated_at: DateTime<Utc>,
}

/// A detected behavior pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorPattern {
    pub id: i64,
    pub pattern_type: String, // "sequence", "trigger", "habit"
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger_context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    pub confidence: f32,
    pub occurrence_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_seen_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

/// A learned procedure (procedural memory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Procedure {
    pub id: i64,
    pub name: String,
    pub trigger_pattern: String,
    pub steps: Vec<String>,
    pub success_count: i32,
    pub failure_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_duration_secs: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_used_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Expertise level in a domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expertise {
    pub id: i64,
    pub domain: String,
    pub tasks_attempted: i32,
    pub tasks_succeeded: i32,
    pub tasks_failed: i32,
    pub current_level: String, // "novice", "competent", "proficient", "expert"
    pub confidence_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub common_errors: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_task_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A learned error-solution pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSolution {
    pub id: i64,
    pub error_pattern: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    pub solution_summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solution_steps: Option<Vec<String>>,
    pub success_count: i32,
    pub failure_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_used_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}
