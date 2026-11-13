//! Per-task `computer_use` session telemetry for post-mortems and tuning.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::Mutex;

/// Mutation budget after a successful reservation (or at failure time).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MutationBudget {
    pub used: u32,
    pub max: u32,
}

impl MutationBudget {
    pub fn remaining(&self) -> u32 {
        self.max.saturating_sub(self.used)
    }

    pub fn at_limit(&self) -> bool {
        self.used >= self.max
    }

    pub fn near_limit(&self) -> bool {
        self.max > 0 && self.used * 5 >= self.max * 4
    }
}

#[derive(Debug, Clone, Default)]
pub struct ElementTarget {
    pub index: Option<u32>,
    pub title: Option<String>,
    pub role: Option<String>,
}

impl ElementTarget {
    pub fn label(&self) -> Option<String> {
        match (&self.title, &self.role) {
            (Some(title), Some(role)) if !title.is_empty() => Some(format!("{role} \"{title}\"")),
            (Some(title), _) if !title.is_empty() => Some(title.clone()),
            (_, Some(role)) if !role.is_empty() => Some(role.clone()),
            _ => self.index.map(|i| format!("index {i}")),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct SessionStats {
    observations: u32,
    mutations_ok: u32,
    mutations_failed: u32,
    mutations_budget_max: u32,
    last_mutations_used: u32,
    action_counts: HashMap<String, u32>,
    apps: HashSet<String>,
    /// Recent click targets as `index:"label"` for repeat-click analysis.
    click_targets: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub observations: u32,
    pub mutations_ok: u32,
    pub mutations_failed: u32,
    pub mutations_budget_max: u32,
    pub last_mutations_used: u32,
    pub total_actions: u32,
    pub apps: Vec<String>,
    pub action_counts: HashMap<String, u32>,
    pub click_targets: Vec<String>,
}

#[derive(Clone)]
pub struct SessionTelemetry {
    sessions: Arc<Mutex<HashMap<String, SessionStats>>>,
}

impl Default for SessionTelemetry {
    fn default() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActionRecord<'a> {
    pub task_id: &'a str,
    pub action: &'a str,
    pub app: &'a str,
    pub is_mutation: bool,
    pub success: bool,
    pub budget: Option<MutationBudget>,
    pub target: Option<&'a ElementTarget>,
}

impl SessionTelemetry {
    pub async fn record_action(&self, record: &ActionRecord<'_>) {
        if record.task_id.is_empty() || record.task_id == "default" {
            return;
        }
        let mut sessions = self.sessions.lock().await;
        let stats = sessions.entry(record.task_id.to_string()).or_default();
        *stats
            .action_counts
            .entry(record.action.to_string())
            .or_insert(0) += 1;
        if !record.app.is_empty() {
            stats.apps.insert(record.app.to_string());
        }
        if record.is_mutation {
            if record.success {
                stats.mutations_ok += 1;
            } else {
                stats.mutations_failed += 1;
            }
        } else {
            stats.observations += 1;
        }
        if let Some(budget) = record.budget {
            stats.mutations_budget_max = budget.max;
            stats.last_mutations_used = budget.used;
        }
        if record.action == "click" {
            if let Some(target) = record.target {
                let label = target.label().unwrap_or_else(|| "?".to_string());
                let index = target.index.unwrap_or(0);
                let entry = format!("{index}:{label}");
                stats.click_targets.push(entry);
                const MAX_CLICK_HISTORY: usize = 30;
                if stats.click_targets.len() > MAX_CLICK_HISTORY {
                    let drain = stats.click_targets.len() - MAX_CLICK_HISTORY;
                    stats.click_targets.drain(0..drain);
                }
            }
        }
    }

    pub async fn finish_task(&self, task_id: &str) -> Option<SessionSummary> {
        let mut sessions = self.sessions.lock().await;
        let stats = sessions.remove(task_id)?;
        let total_actions = stats
            .action_counts
            .values()
            .copied()
            .sum::<u32>()
            .saturating_add(0);
        let mut apps: Vec<String> = stats.apps.into_iter().collect();
        apps.sort();
        Some(SessionSummary {
            observations: stats.observations,
            mutations_ok: stats.mutations_ok,
            mutations_failed: stats.mutations_failed,
            mutations_budget_max: stats.mutations_budget_max,
            last_mutations_used: stats.last_mutations_used,
            total_actions,
            apps,
            action_counts: stats.action_counts,
            click_targets: stats.click_targets,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ActionLog<'a> {
    pub task_id: &'a str,
    pub action: &'a str,
    pub app: &'a str,
    pub generation: Option<u64>,
    pub target: Option<&'a ElementTarget>,
    pub click_method: Option<&'a str>,
    pub duration_ms: u64,
    pub success: bool,
    pub error: Option<&'a str>,
    pub screenshot_bytes: usize,
    pub screenshot_path: Option<&'a str>,
    pub truncated: bool,
    pub budget: Option<MutationBudget>,
    pub is_mutation: bool,
}

pub fn log_action(log: &ActionLog<'_>) {
    let element_index = log.target.and_then(|t| t.index);
    let element_title = log.target.and_then(|t| t.title.as_deref());
    let element_role = log.target.and_then(|t| t.role.as_deref());
    let mutations_used = log.budget.map(|b| b.used);
    let mutations_max = log.budget.map(|b| b.max);
    let mutations_remaining = log.budget.map(|b| b.remaining());

    if log.success {
        tracing::info!(
            target: "computer_use",
            task_id = log.task_id,
            action = log.action,
            app = log.app,
            generation = log.generation,
            element_index,
            element_title,
            element_role,
            click_method = log.click_method,
            duration_ms = log.duration_ms,
            outcome = "ok",
            screenshot_bytes = log.screenshot_bytes,
            screenshot_path = log.screenshot_path,
            truncated = log.truncated,
            mutations_used,
            mutations_max,
            mutations_remaining,
            is_mutation = log.is_mutation,
            "computer_use action"
        );
        if let Some(budget) = log.budget {
            if budget.near_limit() && !budget.at_limit() {
                tracing::warn!(
                    target: "computer_use",
                    task_id = log.task_id,
                    mutations_used = budget.used,
                    mutations_max = budget.max,
                    mutations_remaining = budget.remaining(),
                    "computer_use mutation budget nearly exhausted"
                );
            }
        }
    } else {
        tracing::warn!(
            target: "computer_use",
            task_id = log.task_id,
            action = log.action,
            app = log.app,
            generation = log.generation,
            element_index,
            element_title,
            element_role,
            click_method = log.click_method,
            duration_ms = log.duration_ms,
            outcome = "error",
            error = log.error.unwrap_or("unknown"),
            mutations_used,
            mutations_max,
            mutations_remaining,
            is_mutation = log.is_mutation,
            "computer_use action failed"
        );
    }
}

pub fn log_session_end(task_id: &str, session_id: &str, summary: &SessionSummary) {
    let click_target_summary = if summary.click_targets.is_empty() {
        String::new()
    } else {
        summary.click_targets.join(" | ")
    };
    tracing::info!(
        target: "computer_use",
        task_id,
        session_id,
        observations = summary.observations,
        mutations_ok = summary.mutations_ok,
        mutations_failed = summary.mutations_failed,
        mutations_budget_max = summary.mutations_budget_max,
        last_mutations_used = summary.last_mutations_used,
        total_actions = summary.total_actions,
        apps = ?summary.apps,
        action_counts = ?summary.action_counts,
        click_targets = click_target_summary,
        "computer_use session end"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn session_summary_aggregates_clicks_and_budget() {
        let telemetry = SessionTelemetry::default();
        let target = ElementTarget {
            index: Some(2),
            title: Some("+".to_string()),
            role: Some("AXButton".to_string()),
        };
        telemetry
            .record_action(&ActionRecord {
                task_id: "task-1",
                action: "click",
                app: "Calculator",
                is_mutation: true,
                success: true,
                budget: Some(MutationBudget { used: 1, max: 15 }),
                target: Some(&target),
            })
            .await;
        telemetry
            .record_action(&ActionRecord {
                task_id: "task-1",
                action: "get_app_state",
                app: "Calculator",
                is_mutation: false,
                success: true,
                budget: None,
                target: None,
            })
            .await;
        let summary = telemetry.finish_task("task-1").await.expect("summary");
        assert_eq!(summary.mutations_ok, 1);
        assert_eq!(summary.observations, 1);
        assert_eq!(summary.last_mutations_used, 1);
        assert_eq!(summary.click_targets, vec!["2:AXButton \"+\"".to_string()]);
    }

    #[test]
    fn mutation_budget_near_limit() {
        let budget = MutationBudget { used: 12, max: 15 };
        assert!(budget.near_limit());
        assert!(!budget.at_limit());
    }
}
