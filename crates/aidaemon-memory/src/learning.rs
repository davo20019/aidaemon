//! Learning memory: what the agent learns from doing work, beside facts and
//! episodes. Procedures (reusable step sequences), error solutions,
//! per-domain expertise, behavior patterns and the user's communication
//! profile, with [`LearningStore`] as the storage contract.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::Result;

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

/// Learning system: procedures, expertise, behavior patterns, and error solutions.
#[async_trait]
pub trait LearningStore: Send + Sync {
    /// Get behavior patterns above a confidence threshold.
    async fn get_behavior_patterns(&self, _min_confidence: f32) -> Result<Vec<BehaviorPattern>> {
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
    ) -> Result<()> {
        Ok(())
    }

    /// Get procedures relevant to a query.
    async fn get_relevant_procedures(&self, _query: &str, _limit: usize) -> Result<Vec<Procedure>> {
        Ok(vec![])
    }

    /// Get error solutions relevant to an error message.
    async fn get_relevant_error_solutions(
        &self,
        _error: &str,
        _limit: usize,
    ) -> Result<Vec<ErrorSolution>> {
        Ok(vec![])
    }

    /// Get the user profile.
    async fn get_user_profile(&self) -> Result<Option<UserProfile>> {
        Ok(None)
    }

    /// Increment expertise counters and update level for a domain.
    async fn increment_expertise(
        &self,
        _domain: &str,
        _success: bool,
        _error: Option<&str>,
    ) -> Result<()> {
        Ok(())
    }

    /// Insert or update a procedure.
    async fn upsert_procedure(&self, _procedure: &Procedure) -> Result<i64> {
        Ok(0)
    }

    /// Update procedure outcome after execution.
    #[allow(dead_code)] // Reserved for procedure feedback loop
    async fn update_procedure_outcome(
        &self,
        _procedure_id: i64,
        _success: bool,
        _duration: Option<f32>,
    ) -> Result<()> {
        Ok(())
    }

    /// Insert a new error-solution pair.
    async fn insert_error_solution(&self, _solution: &ErrorSolution) -> Result<i64> {
        Ok(0)
    }

    /// Update error solution outcome.
    #[allow(dead_code)] // Reserved for error solution feedback loop
    async fn update_error_solution_outcome(&self, _solution_id: i64, _success: bool) -> Result<()> {
        Ok(())
    }
}

/// Calculate expertise level based on success/failure counts.
///
/// Returns a tuple of (level_name, confidence_score).
/// - novice: <5 tasks or <70% success
/// - competent: >=5 tasks with >=70% success
/// - proficient: >=15 tasks with >=80% success
/// - expert: >=30 tasks with >=90% success
pub fn expertise_level(succeeded: i32, failed: i32) -> (&'static str, f32) {
    let total = succeeded + failed;
    if total == 0 {
        return ("novice", 0.0);
    }

    let success_rate = succeeded as f32 / total as f32;
    let volume_factor = (total as f32 / 50.0).min(1.0);
    let confidence = success_rate * volume_factor;

    let level = match (total, success_rate) {
        (t, r) if t >= 30 && r >= 0.9 => "expert",
        (t, r) if t >= 15 && r >= 0.8 => "proficient",
        (t, r) if t >= 5 && r >= 0.7 => "competent",
        _ => "novice",
    };

    (level, confidence)
}

/// Normalize explicit domain hints into canonical expertise domains.
fn normalize_domain(domain: &str) -> Option<&'static str> {
    match domain.trim().to_ascii_lowercase().as_str() {
        "rust" => Some("rust"),
        "python" => Some("python"),
        "javascript" | "typescript" | "js" | "ts" => Some("javascript"),
        "go" | "golang" => Some("go"),
        "docker" => Some("docker"),
        "kubernetes" | "k8s" => Some("kubernetes"),
        "infrastructure" | "devops" => Some("infrastructure"),
        "web-frontend" | "frontend" => Some("web-frontend"),
        "web-backend" | "backend" => Some("web-backend"),
        "databases" | "database" | "db" => Some("databases"),
        "git" => Some("git"),
        "system-admin" | "sysadmin" | "shell" => Some("system-admin"),
        "general" => Some("general"),
        _ => None,
    }
}

/// Detect domains from explicit classifier hints only.
///
/// Keyword-based message parsing is intentionally disabled to avoid false
/// positives; callers must provide explicit domain hints.
pub fn detect_domains(explicit_domains: &[String]) -> Vec<String> {
    let mut domains = Vec::new();
    for domain in explicit_domains {
        if let Some(normalized) = normalize_domain(domain) {
            let normalized = normalized.to_string();
            if !domains.contains(&normalized) {
                domains.push(normalized);
            }
        }
    }

    if domains.is_empty() {
        domains.push("general".to_string());
    }
    domains
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_novice_level() {
        let (level, _) = expertise_level(0, 0);
        assert_eq!(level, "novice");

        let (level, _) = expertise_level(2, 2);
        assert_eq!(level, "novice");
    }

    #[test]
    fn test_competent_level() {
        let (level, _) = expertise_level(7, 3); // 10 total, 70% success
        assert_eq!(level, "competent");
    }

    #[test]
    fn test_proficient_level() {
        let (level, _) = expertise_level(16, 4); // 20 total, 80% success
        assert_eq!(level, "proficient");
    }

    #[test]
    fn test_expert_level() {
        let (level, _) = expertise_level(45, 5); // 50 total, 90% success
        assert_eq!(level, "expert");
    }

    #[test]
    fn test_detect_domains() {
        let domains = detect_domains(&["Rust".to_string(), "typescript".to_string()]);
        assert!(domains.contains(&"rust".to_string()));
        assert!(domains.contains(&"javascript".to_string()));

        let domains = detect_domains(&["kubernetes".to_string(), "docker".to_string()]);
        assert!(domains.contains(&"docker".to_string()));
        assert!(domains.contains(&"kubernetes".to_string()));
    }

    #[test]
    fn test_detect_domains_defaults_general_when_no_explicit_hints() {
        let domains = detect_domains(&[]);
        assert_eq!(domains, vec!["general".to_string()]);
    }
}
