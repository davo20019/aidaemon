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
}

fn default_fact_privacy() -> FactPrivacy {
    FactPrivacy::Global
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

/// A goal being tracked over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: i64,
    pub description: String,
    pub status: String,   // "active", "completed", "abandoned"
    pub priority: String, // "low", "medium", "high"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress_notes: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_episode_id: Option<i64>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
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

