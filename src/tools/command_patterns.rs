//! Command pattern learning module.
//!
//! This module handles:
//! - Generalizing commands into reusable patterns
//! - Storing approval/denial history for patterns
//! - Matching new commands against learned patterns
//! - Calculating confidence scores for pattern-based risk adjustment

use regex::Regex;
use sqlx::SqlitePool;
use std::sync::OnceLock;
use chrono::Utc;
use tracing::{debug, info};

/// Row type for command pattern queries.
type PatternRow = (i64, String, String, i32, i32, Option<String>, Option<String>);

/// A learned command pattern with approval history.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CommandPattern {
    pub id: i64,
    pub pattern: String,
    pub original_example: String,
    pub approval_count: i32,
    pub denial_count: i32,
    pub last_approved_at: Option<String>,
    pub last_denied_at: Option<String>,
}

impl CommandPattern {
    /// Calculate confidence score (0.0 to 1.0) based on approval history.
    pub fn confidence(&self) -> f32 {
        let total = self.approval_count + self.denial_count;
        if total == 0 {
            return 0.0;
        }
        let approval_ratio = self.approval_count as f32 / total as f32;
        let volume_factor = (total as f32 / 10.0).min(1.0);
        approval_ratio * volume_factor
    }

    /// Check if this pattern is trustworthy enough to lower risk.
    pub fn is_trusted(&self) -> bool {
        self.approval_count >= 3 && self.confidence() >= 0.8
    }
}

// Pre-compiled regexes for pattern generalization
fn path_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"(?:^|[\s=])(/[\w./-]+|\.{1,2}/[\w./-]+)").unwrap())
}

fn url_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"https?://[^\s]+").unwrap())
}

fn number_arg_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"[0-9]{4,}").unwrap())
}

fn single_quoted_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"'[^']*'").unwrap())
}

fn double_quoted_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r#""[^"]*""#).unwrap())
}

/// Generalize a command into a pattern by replacing specific values with placeholders.
pub fn generalize_command(command: &str) -> String {
    let mut pattern = command.to_string();

    // Replace URLs first (before paths, since URLs contain paths)
    pattern = url_regex().replace_all(&pattern, "<url>").to_string();

    // Replace absolute and relative paths
    pattern = path_regex().replace_all(&pattern, " <path>").to_string();

    // Replace quoted strings (likely user-specific content)
    pattern = single_quoted_regex().replace_all(&pattern, "<string>").to_string();
    pattern = double_quoted_regex().replace_all(&pattern, "<string>").to_string();

    // Replace large numbers (PIDs, ports, etc.)
    pattern = number_arg_regex().replace_all(&pattern, "<num>").to_string();

    // Clean up multiple spaces
    let mut prev_space = false;
    pattern = pattern
        .chars()
        .filter(|c| {
            if *c == ' ' {
                if prev_space { false } else { prev_space = true; true }
            } else {
                prev_space = false;
                true
            }
        })
        .collect();

    pattern.trim().to_string()
}

/// Calculate similarity between a command and a pattern.
pub fn pattern_similarity(command: &str, pattern: &str) -> f32 {
    let generalized = generalize_command(command);

    if generalized == pattern {
        return 1.0;
    }

    let cmd_tokens: Vec<&str> = generalized.split_whitespace().collect();
    let pat_tokens: Vec<&str> = pattern.split_whitespace().collect();

    if cmd_tokens.is_empty() || pat_tokens.is_empty() {
        return 0.0;
    }

    if cmd_tokens[0] != pat_tokens[0] {
        return 0.0;
    }

    let mut matches = 1;
    let min_len = cmd_tokens.len().min(pat_tokens.len());

    for i in 1..min_len {
        if cmd_tokens[i] == pat_tokens[i] ||
           pat_tokens[i].starts_with('<') ||
           cmd_tokens[i].starts_with('<') {
            matches += 1;
        }
    }

    let max_len = cmd_tokens.len().max(pat_tokens.len());
    matches as f32 / max_len as f32
}

/// Store a command pattern after user approval.
pub async fn record_approval(pool: &SqlitePool, command: &str) -> anyhow::Result<()> {
    let pattern = generalize_command(command);
    let now = Utc::now().to_rfc3339();

    let result = sqlx::query(
        "UPDATE command_patterns SET approval_count = approval_count + 1, last_approved_at = ? WHERE pattern = ?"
    )
    .bind(&now)
    .bind(&pattern)
    .execute(pool)
    .await?;

    if result.rows_affected() == 0 {
        sqlx::query(
            "INSERT INTO command_patterns (pattern, original_example, approval_count, last_approved_at, created_at) VALUES (?, ?, 1, ?, ?)"
        )
        .bind(&pattern)
        .bind(command)
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await?;
        info!(pattern = %pattern, "Learned new command pattern from approval");
    } else {
        debug!(pattern = %pattern, "Updated existing pattern approval count");
    }

    Ok(())
}

/// Record a denied command to track patterns users reject.
pub async fn record_denial(pool: &SqlitePool, command: &str) -> anyhow::Result<()> {
    let pattern = generalize_command(command);
    let now = Utc::now().to_rfc3339();

    let result = sqlx::query(
        "UPDATE command_patterns SET denial_count = denial_count + 1, last_denied_at = ? WHERE pattern = ?"
    )
    .bind(&now)
    .bind(&pattern)
    .execute(pool)
    .await?;

    if result.rows_affected() == 0 {
        sqlx::query(
            "INSERT INTO command_patterns (pattern, original_example, approval_count, denial_count, last_denied_at, created_at) VALUES (?, ?, 0, 1, ?, ?)"
        )
        .bind(&pattern)
        .bind(command)
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await?;
        info!(pattern = %pattern, "Recorded denied command pattern");
    }

    Ok(())
}

/// Find the best matching pattern for a command.
pub async fn find_matching_pattern(pool: &SqlitePool, command: &str) -> anyhow::Result<Option<(CommandPattern, f32)>> {
    let generalized = generalize_command(command);

    let exact: Option<PatternRow> = sqlx::query_as(
        "SELECT id, pattern, original_example, approval_count, denial_count, last_approved_at, last_denied_at FROM command_patterns WHERE pattern = ?"
    )
    .bind(&generalized)
    .fetch_optional(pool)
    .await?;

    if let Some((id, pattern, original_example, approval_count, denial_count, last_approved_at, last_denied_at)) = exact {
        return Ok(Some((CommandPattern {
            id, pattern, original_example, approval_count, denial_count, last_approved_at, last_denied_at,
        }, 1.0)));
    }

    let base_cmd = command.split_whitespace().next().unwrap_or("");
    if base_cmd.is_empty() {
        return Ok(None);
    }

    let candidates: Vec<PatternRow> = sqlx::query_as(
        "SELECT id, pattern, original_example, approval_count, denial_count, last_approved_at, last_denied_at FROM command_patterns WHERE pattern LIKE ? ORDER BY approval_count DESC LIMIT 20"
    )
    .bind(format!("{}%", base_cmd))
    .fetch_all(pool)
    .await?;

    let mut best_match: Option<(CommandPattern, f32)> = None;

    for (id, pattern, original_example, approval_count, denial_count, last_approved_at, last_denied_at) in candidates {
        let similarity = pattern_similarity(command, &pattern);
        if similarity >= 0.7
            && (best_match.is_none() || similarity > best_match.as_ref().unwrap().1) {
                best_match = Some((CommandPattern {
                    id, pattern, original_example, approval_count, denial_count, last_approved_at, last_denied_at,
                }, similarity));
            }
    }

    Ok(best_match)
}

/// Statistics about learned command patterns.
#[derive(Debug)]
#[allow(dead_code)]
pub struct PatternStats {
    pub total_patterns: usize,
    pub trusted_patterns: usize,
    pub top_patterns: Vec<(String, i32)>,
}

/// Get statistics about learned patterns for display/prompt.
#[allow(dead_code)]
pub async fn get_pattern_stats(pool: &SqlitePool) -> anyhow::Result<PatternStats> {
    let total: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM command_patterns")
        .fetch_one(pool)
        .await?;

    let trusted: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM command_patterns WHERE approval_count >= 3")
        .fetch_one(pool)
        .await?;

    let top_patterns: Vec<(String, i32)> = sqlx::query_as(
        "SELECT pattern, approval_count FROM command_patterns WHERE approval_count >= 3 ORDER BY approval_count DESC LIMIT 10"
    )
    .fetch_all(pool)
    .await?;

    Ok(PatternStats {
        total_patterns: total.0 as usize,
        trusted_patterns: trusted.0 as usize,
        top_patterns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generalize_command() {
        assert_eq!(generalize_command("cargo test"), "cargo test");
        assert!(generalize_command("curl https://api.example.com").contains("<url>"));
        assert!(generalize_command("kill 12345").contains("<num>"));
    }

    #[test]
    fn test_pattern_similarity() {
        assert_eq!(pattern_similarity("cargo test", "cargo test"), 1.0);
        assert_eq!(pattern_similarity("cargo test", "npm test"), 0.0);
    }

    #[test]
    fn test_confidence_calculation() {
        let pattern = CommandPattern {
            id: 1,
            pattern: "cargo test".to_string(),
            original_example: "cargo test".to_string(),
            approval_count: 10,
            denial_count: 0,
            last_approved_at: None,
            last_denied_at: None,
        };
        assert_eq!(pattern.confidence(), 1.0);
        assert!(pattern.is_trusted());
    }
}
