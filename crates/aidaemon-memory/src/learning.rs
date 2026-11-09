//! Learning memory: what the agent learns from doing work, beside facts and
//! episodes. Procedures (reusable step sequences), error solutions,
//! per-domain expertise, behavior patterns and the user's communication
//! profile, with [`LearningStore`] as the storage contract.

use std::sync::LazyLock;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::embed::Embedder;
use crate::error::{MemoryError, Result};

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

    /// Store the user profile.
    async fn update_user_profile(&self, _profile: &UserProfile) -> Result<()> {
        Ok(())
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

    /// Record a procedure seen in a finished task. Unlike
    /// [`upsert_procedure`](Self::upsert_procedure), a name already known
    /// keeps its stored trigger, steps and failure count and gains exactly
    /// one success, so a later sighting cannot rewrite a definition.
    async fn record_procedure_observation(&self, procedure: &Procedure) -> Result<()> {
        self.upsert_procedure(procedure).await.map(|_| ())
    }

    /// Record an error followed by a recovery. The pair is stored as given
    /// (no further normalization) and is embedded later, in the background.
    async fn record_error_solution_observation(&self, solution: &ErrorSolution) -> Result<()> {
        self.insert_error_solution(solution).await.map(|_| ())
    }

    /// Count a finished task toward a domain. Unlike
    /// [`increment_expertise`](Self::increment_expertise), this only moves
    /// the counters; the level and confidence stay as stored.
    async fn record_task_outcome(&self, domain: &str, success: bool) -> Result<()> {
        self.increment_expertise(domain, success, None).await
    }

    /// The trigger embeddings of the stored procedures, to spot duplicates.
    async fn procedure_trigger_embeddings(&self) -> Result<Vec<Vec<f32>>> {
        Ok(vec![])
    }

    /// Set the trigger embedding of the procedure with this name.
    async fn set_procedure_trigger_embedding(&self, _name: &str, _embedding: &[f32]) -> Result<()> {
        Ok(())
    }

    /// Up to `limit` procedures `(id, trigger)` with no current embedding.
    async fn procedures_missing_embedding(&self, _limit: usize) -> Result<Vec<(i64, String)>> {
        Ok(vec![])
    }

    /// Store a procedure's trigger embedding, computed over `trigger`.
    async fn save_procedure_embedding(
        &self,
        _id: i64,
        _trigger: &str,
        _embedding: Vec<f32>,
    ) -> Result<()> {
        Ok(())
    }

    /// Up to `limit` error solutions `(id, pattern)` with no current embedding.
    async fn error_solutions_missing_embedding(&self, _limit: usize) -> Result<Vec<(i64, String)>> {
        Ok(vec![])
    }

    /// Store an error solution's embedding, computed over `pattern`.
    async fn save_error_solution_embedding(
        &self,
        _id: i64,
        _pattern: &str,
        _embedding: Vec<f32>,
    ) -> Result<()> {
        Ok(())
    }
}

/// Embeddings computed per background pass, per kind of row.
const EMBEDDING_BACKFILL_BATCH: usize = 10;

/// Embed the procedures and error solutions that have no current embedding,
/// one batch of each. Returns whether there was anything to do.
pub async fn backfill_learning_embeddings<S>(store: &S, embedder: &dyn Embedder) -> Result<bool>
where
    S: LearningStore + ?Sized,
{
    let procedures = store
        .procedures_missing_embedding(EMBEDDING_BACKFILL_BATCH)
        .await?;
    if !procedures.is_empty() {
        info!(
            "Generating trigger embeddings for {} procedures",
            procedures.len()
        );
    }
    let mut did_work = !procedures.is_empty();
    for (id, trigger) in procedures {
        let trigger = trigger.trim();
        if trigger.is_empty() {
            continue;
        }
        match embedding_of(embedder, trigger).await {
            Ok(embedding) => {
                if let Err(error) = store.save_procedure_embedding(id, trigger, embedding).await {
                    warn!(procedure_id = id, %error, "Failed to project procedure embedding");
                }
            }
            Err(error) => {
                warn!(procedure_id = id, %error, "Failed to generate trigger embedding for procedure");
            }
        }
    }

    let solutions = store
        .error_solutions_missing_embedding(EMBEDDING_BACKFILL_BATCH)
        .await?;
    if !solutions.is_empty() {
        info!(
            "Generating embeddings for {} error solutions",
            solutions.len()
        );
    }
    did_work |= !solutions.is_empty();
    for (id, pattern) in solutions {
        let pattern = pattern.trim();
        if pattern.is_empty() {
            continue;
        }
        match embedding_of(embedder, pattern).await {
            Ok(embedding) => {
                if let Err(error) = store
                    .save_error_solution_embedding(id, pattern, embedding)
                    .await
                {
                    warn!(error_solution_id = id, %error, "Failed to project error-solution embedding");
                }
            }
            Err(error) => {
                warn!(error_solution_id = id, %error, "Failed to generate embedding for error solution");
            }
        }
    }
    Ok(did_work)
}

/// The embedding of one text.
pub(crate) async fn embedding_of(embedder: &dyn Embedder, text: &str) -> Result<Vec<f32>> {
    embedder
        .embed(&[text.to_string()])
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| MemoryError::Embedding("embedder returned no vector".to_string()))
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

/// Replace specific paths and URLs in recorded actions with `<path>` and
/// `<url>`, so a learned procedure applies beyond the task it came from.
pub fn generalize_procedure(actions: &[String]) -> Vec<String> {
    actions
        .iter()
        .map(|action| {
            let generalized = PATH.replace_all(action, "<path>");
            URL.replace_all(&generalized, "<url>").into_owned()
        })
        .collect()
}

/// The trigger a procedure is recalled by: the task's first sentence, cut
/// to 100 bytes at a character boundary.
pub fn extract_trigger_pattern(task_context: &str) -> String {
    let first_sentence = task_context
        .split('.')
        .next()
        .unwrap_or(task_context)
        .trim();
    truncate_bytes(first_sentence, 100).to_string()
}

/// A readable procedure name from its steps: the distinct tools it used, in
/// order (at most three), such as `terminal-edit_file`. Steps are recorded
/// as `tool(arguments)` (or `tool: ...`); the name never depends on the
/// wording of the request.
pub fn procedure_base_name(steps: &[String]) -> String {
    let mut tools: Vec<String> = Vec::new();
    for step in steps {
        let tool: String = step
            .split(['(', ':'])
            .next()
            .unwrap_or_default()
            .trim()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect::<String>()
            .to_ascii_lowercase();
        if !tool.is_empty() && !tools.contains(&tool) {
            tools.push(tool);
        }
        if tools.len() == 3 {
            break;
        }
    }
    if tools.is_empty() {
        "procedure".to_string()
    } else {
        tools.join("-")
    }
}

/// A stable, collision-resistant procedure name: the readable `base_name`
/// plus a short hash of the normalized steps, so different workflows under
/// a generic name ("deploy") do not overwrite each other.
pub fn procedure_keyed_name(base_name: &str, steps: &[String]) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    for step in steps {
        hasher.update(step.to_lowercase().as_bytes());
        hasher.update(b"\n");
    }
    let suffix: String = hasher
        .finalize()
        .iter()
        .take(4)
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("{base_name}-{suffix}")
}

/// A new procedure recording its first outcome. The store assigns the id.
pub fn new_procedure(
    name: String,
    trigger_pattern: String,
    steps: Vec<String>,
    success: bool,
) -> Procedure {
    let now = Utc::now();
    Procedure {
        id: 0,
        name,
        trigger_pattern,
        steps,
        success_count: i32::from(success),
        failure_count: i32::from(!success),
        avg_duration_secs: None,
        last_used_at: Some(now),
        created_at: now,
        updated_at: now,
    }
}

/// An error message reduced to a pattern that matches recurrences: line and
/// column numbers become `:<line>`, paths become `<path>`, cut to 200 bytes.
pub fn error_pattern(error: &str) -> String {
    let pattern = LINE_COLUMN.replace_all(error.trim(), ":<line>");
    let pattern = PATH.replace_all(&pattern, "<path>");
    truncate_bytes(&pattern, 200).to_string()
}

/// A one-line summary of the tools a solution used (`name(...)` actions).
pub fn summarize_solution(actions: &[String]) -> String {
    if actions.is_empty() {
        return "No specific actions recorded".to_string();
    }
    let tools: Vec<&str> = actions
        .iter()
        .filter_map(|action| action.split('(').next())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    format!("Used {} tool(s): {}", tools.len(), tools.join(", "))
}

/// A new error solution, recorded as having worked once. The store assigns
/// the id.
pub fn new_error_solution(
    error_pattern: String,
    domain: Option<String>,
    solution_summary: String,
    solution_steps: Option<Vec<String>>,
) -> ErrorSolution {
    let now = Utc::now();
    ErrorSolution {
        id: 0,
        error_pattern,
        domain,
        solution_summary,
        solution_steps,
        success_count: 1,
        failure_count: 0,
        last_used_at: Some(now),
        created_at: now,
    }
}

static PATH: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"/[\w/.-]+").unwrap());
static URL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"https?://[^\s)]+").unwrap());
static LINE_COLUMN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r":\d+:\d+").unwrap());

/// `text` cut to at most `max` bytes, on a character boundary.
fn truncate_bytes(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Communication verbosity from the average length of the user's
/// messages: `brief` up to 50 bytes, `medium` up to 200, `detailed` above.
/// `None` when there are no messages.
pub fn verbosity_from_messages(messages: &[&str]) -> Option<&'static str> {
    if messages.is_empty() {
        return None;
    }
    let average = messages.iter().map(|m| m.len() as f32).sum::<f32>() / messages.len() as f32;
    Some(if average > 200.0 {
        "detailed"
    } else if average > 50.0 {
        "medium"
    } else {
        "brief"
    })
}

/// Update the stored profile's verbosity from the owner's recent messages.
/// Tone and explanation preferences are not inferred from wording, so they
/// keep their stored values.
pub async fn update_communication_profile<S>(
    store: &S,
    owner_messages: &[&str],
) -> Result<Option<UserProfile>>
where
    S: LearningStore + ?Sized,
{
    let Some(verbosity) = verbosity_from_messages(owner_messages) else {
        return Ok(None);
    };
    let Some(mut profile) = store.get_user_profile().await? else {
        return Ok(None);
    };
    profile.verbosity_preference = verbosity.to_string();
    profile.updated_at = Utc::now();
    store.update_user_profile(&profile).await?;
    Ok(Some(profile))
}

/// Tool sequences seen often enough to be a habit: each pair "after A, B"
/// that occurs at least three times in `tools` (calls in order), with a
/// confidence from its share of all calls (capped at 0.9).
pub fn tool_sequence_patterns(tools: &[String]) -> Vec<(String, String, i32, f32)> {
    if tools.len() < 3 {
        return Vec::new();
    }
    let mut counts: std::collections::HashMap<(&str, &str), i32> = std::collections::HashMap::new();
    for pair in tools.windows(2) {
        *counts
            .entry((pair[0].as_str(), pair[1].as_str()))
            .or_insert(0) += 1;
    }
    let mut patterns: Vec<(String, String, i32, f32)> = counts
        .into_iter()
        .filter(|(_, count)| *count >= 3)
        .map(|((a, b), count)| {
            let confidence = (count as f32 / tools.len() as f32).min(0.9);
            (a.to_string(), b.to_string(), count, confidence)
        })
        .collect();
    patterns.sort_by(|x, y| {
        y.2.cmp(&x.2)
            .then_with(|| x.0.cmp(&y.0))
            .then_with(|| x.1.cmp(&y.1))
    });
    patterns
}

/// Record the habitual tool sequences in `tools` as behavior patterns.
pub async fn record_tool_sequence_patterns<S>(
    store: &S,
    tools: &[String],
) -> Result<Vec<BehaviorPattern>>
where
    S: LearningStore + ?Sized,
{
    let now = Utc::now();
    let mut recorded = Vec::new();
    for (tool_a, tool_b, count, confidence) in tool_sequence_patterns(tools) {
        let description = format!("After {tool_a} often use {tool_b}");
        store
            .record_behavior_pattern(
                "sequence",
                &description,
                Some(&tool_a),
                Some(&tool_b),
                confidence,
                count,
            )
            .await?;
        recorded.push(BehaviorPattern {
            id: 0,
            pattern_type: "sequence".to_string(),
            description,
            trigger_context: Some(tool_a),
            action: Some(tool_b),
            confidence,
            occurrence_count: count,
            last_seen_at: Some(now),
            created_at: now,
        });
    }
    Ok(recorded)
}

#[cfg(test)]
mod tests {

    #[test]
    fn verbosity_reads_message_length_only() {
        assert_eq!(verbosity_from_messages(&[]), None);
        assert_eq!(verbosity_from_messages(&["ok", "thanks"]), Some("brief"));
        let long = "x".repeat(300);
        assert_eq!(verbosity_from_messages(&[long.as_str()]), Some("detailed"));
    }

    #[test]
    fn habits_are_pairs_seen_three_times() {
        let calls: Vec<String> = ["read", "edit", "read", "edit", "read", "edit", "test"]
            .into_iter()
            .map(String::from)
            .collect();
        let patterns = tool_sequence_patterns(&calls);
        assert_eq!(patterns.len(), 1);
        assert_eq!(
            (
                patterns[0].0.as_str(),
                patterns[0].1.as_str(),
                patterns[0].2
            ),
            ("read", "edit", 3)
        );
        assert!(tool_sequence_patterns(&calls[..2]).is_empty());
    }

    #[test]
    fn error_patterns_drop_locations() {
        let pattern = error_pattern(
            "error[E0382]: borrow of moved value at /home/user/project/src/main.rs:42:10",
        );
        assert!(
            pattern.contains("E0382") && pattern.contains("<path>") && pattern.contains("<line>")
        );
        assert!(error_pattern(&"é".repeat(300)).len() <= 200);
    }

    #[test]
    fn procedure_names_come_from_the_tools_used() {
        let steps = vec![
            "terminal(cargo build)".to_string(),
            "edit_file(src/main.rs)".to_string(),
            "terminal(cargo test)".to_string(),
        ];
        assert_eq!(procedure_base_name(&steps), "terminal-edit_file");
        assert_eq!(
            procedure_base_name(&["terminal: cat <path>".to_string()]),
            "terminal"
        );
        assert_eq!(procedure_base_name(&[]), "procedure");
    }

    #[test]
    fn procedures_generalize_and_get_stable_names() {
        let actions = vec![
            "terminal(cargo)".to_string(),
            "web_fetch(/home/user/file.rs)".to_string(),
        ];
        assert!(generalize_procedure(&actions)[1].contains("<path>"));
        let a = procedure_keyed_name("deploy", &actions);
        assert_eq!(a, procedure_keyed_name("deploy", &actions));
        assert!(a.starts_with("deploy-"));
        let failed = new_procedure("n".into(), "t".into(), vec!["cargo test".into()], false);
        assert_eq!((failed.success_count, failed.failure_count), (0, 1));
        assert_eq!(
            extract_trigger_pattern("Run the tests. Then deploy."),
            "Run the tests"
        );
    }

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
