//! Session context compilation from events.
//!
//! This module provides the SessionContext struct and compiler that
//! transforms raw events into a structured context for system prompt injection.

use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use serde::Serialize;

use super::{
    Event, EventStore, EventType, TaskStatus,
    ErrorData, TaskEndData, TaskStartData, ThinkingStartData, ToolCallData, ToolResultData,
    SubAgentSpawnData,
};
use crate::utils::truncate_str;

/// Compiled session context for system prompt injection.
///
/// This provides the agent with awareness of its current and recent activity,
/// enabling it to answer questions like "what are you doing?" and "what was the error?"
#[derive(Debug, Clone, Serialize, Default)]
pub struct SessionContext {
    /// Currently running task (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_task: Option<CurrentTask>,

    /// Last completed/ended task
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_completed_task: Option<CompletedTask>,

    /// Most recent error (if any in the time window)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<RecentError>,

    /// Recent tool calls (limited to last N)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub recent_tools: Vec<RecentTool>,

    /// Current thinking iteration (if task is running)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_iteration: Option<u32>,

    /// Active sub-agents (spawned but not completed)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub active_sub_agents: Vec<ActiveSubAgent>,

    /// Total events in the time window
    pub event_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CurrentTask {
    pub task_id: String,
    pub description: String,
    pub started_at: DateTime<Utc>,
    pub elapsed_secs: u64,
    pub iterations: u32,
    pub tool_calls: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletedTask {
    pub task_id: String,
    pub description: String,
    pub status: TaskStatus,
    pub duration_secs: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub completed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecentError {
    pub message: String,
    pub error_type: String,
    pub occurred_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    pub recovered: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecentTool {
    pub name: String,
    pub summary: String,
    pub success: bool,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActiveSubAgent {
    pub child_session_id: String,
    pub mission: String,
    pub depth: u32,
    pub started_at: DateTime<Utc>,
}

impl SessionContext {
    /// Check if there's any meaningful context to include
    pub fn is_empty(&self) -> bool {
        self.current_task.is_none()
            && self.last_completed_task.is_none()
            && self.last_error.is_none()
            && self.recent_tools.is_empty()
            && self.active_sub_agents.is_empty()
    }

    /// Format the context for system prompt injection
    pub fn format_for_prompt(&self) -> String {
        if self.is_empty() {
            return String::new();
        }

        let mut lines = vec!["## Current Session Activity".to_string()];

        // Current task
        if let Some(task) = &self.current_task {
            lines.push(format!(
                "- **Active task:** \"{}\" (running {}s, iteration {}, {} tool calls)",
                truncate_str(&task.description, 60),
                task.elapsed_secs,
                task.iterations,
                task.tool_calls
            ));
        }

        // Last completed task
        if let Some(task) = &self.last_completed_task {
            let status_str = match task.status {
                TaskStatus::Completed => "completed",
                TaskStatus::Cancelled => "CANCELLED",
                TaskStatus::Failed => "FAILED",
            };
            let error_suffix = task
                .error
                .as_ref()
                .map(|e| format!(" - Error: {}", truncate_str(e, 50)))
                .unwrap_or_default();
            lines.push(format!(
                "- **Last task:** \"{}\" - {} ({}s){}",
                truncate_str(&task.description, 50),
                status_str,
                task.duration_secs,
                error_suffix
            ));
        }

        // Last error
        if let Some(error) = &self.last_error {
            let tool_suffix = error
                .tool_name
                .as_ref()
                .map(|t| format!(" in {}", t))
                .unwrap_or_default();
            let recovered_suffix = if error.recovered { " (recovered)" } else { "" };
            lines.push(format!(
                "- **Recent error:** {}{}{}",
                truncate_str(&error.message, 60),
                tool_suffix,
                recovered_suffix
            ));
        }

        // Recent tools (summarized)
        if !self.recent_tools.is_empty() {
            let tool_summary: Vec<String> = self
                .recent_tools
                .iter()
                .take(5)
                .map(|t| {
                    let status = if t.success { "ok" } else { "err" };
                    format!("{}({})", t.name, status)
                })
                .collect();
            lines.push(format!("- **Recent tools:** {}", tool_summary.join(", ")));
        }

        // Active sub-agents
        if !self.active_sub_agents.is_empty() {
            let sub_agent_summary: Vec<String> = self
                .active_sub_agents
                .iter()
                .map(|sa| format!("depth {}: \"{}\"", sa.depth, truncate_str(&sa.mission, 30)))
                .collect();
            lines.push(format!("- **Sub-agents:** {}", sub_agent_summary.join("; ")));
        }

        lines.push(String::new());
        lines.push(
            "IMPORTANT: This session activity is reference context only. \
             When the user sends a new message, respond to THAT message. \
             Do NOT continue or resume previous tasks unless the user explicitly asks \
             (e.g., \"continue\", \"keep going\", \"resume\"). \
             If the user asks about your recent activity (\"what did you do?\", \
             \"why didn't you respond?\"), explain using this context — do not take new actions."
                .to_string(),
        );

        lines.join("\n")
    }
}

/// Compiles SessionContext from raw events
pub struct SessionContextCompiler {
    store: Arc<EventStore>,
}

impl SessionContextCompiler {
    pub fn new(store: Arc<EventStore>) -> Self {
        Self { store }
    }

    /// Compile session context from events within a time window
    pub async fn compile(
        &self,
        session_id: &str,
        window: Duration,
    ) -> anyhow::Result<SessionContext> {
        let since = Utc::now() - window;
        let events = self.store.query_events(session_id, since).await?;

        Ok(self.compile_from_events(&events))
    }

    /// Compile session context from a pre-fetched list of events
    pub fn compile_from_events(&self, events: &[Event]) -> SessionContext {
        let mut context = SessionContext {
            event_count: events.len(),
            ..Default::default()
        };

        if events.is_empty() {
            return context;
        }

        // Track task states
        let mut task_starts: std::collections::HashMap<String, (Event, TaskStartData)> =
            std::collections::HashMap::new();
        let mut task_ends: std::collections::HashMap<String, (Event, TaskEndData)> =
            std::collections::HashMap::new();
        let mut task_iterations: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
        let mut task_tool_calls: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();

        // Track sub-agent states
        let mut sub_agent_starts: std::collections::HashMap<String, (Event, SubAgentSpawnData)> =
            std::collections::HashMap::new();
        let mut completed_sub_agents: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // Recent tools
        let mut recent_tools: Vec<RecentTool> = Vec::new();

        // Last error
        let mut last_error: Option<(Event, ErrorData)> = None;

        // Process events
        for event in events {
            match event.event_type {
                EventType::TaskStart => {
                    if let Ok(data) = event.parse_data::<TaskStartData>() {
                        task_starts.insert(data.task_id.clone(), (event.clone(), data));
                    }
                }
                EventType::TaskEnd => {
                    if let Ok(data) = event.parse_data::<TaskEndData>() {
                        task_ends.insert(data.task_id.clone(), (event.clone(), data));
                    }
                }
                EventType::ThinkingStart => {
                    if let Ok(data) = event.parse_data::<ThinkingStartData>() {
                        *task_iterations.entry(data.task_id.clone()).or_insert(0) = data.iteration;
                    }
                }
                EventType::ToolCall => {
                    if let Ok(data) = event.parse_data::<ToolCallData>() {
                        if let Some(task_id) = &data.task_id {
                            *task_tool_calls.entry(task_id.clone()).or_insert(0) += 1;
                        }
                    }
                }
                EventType::ToolResult => {
                    if let Ok(data) = event.parse_data::<ToolResultData>() {
                        recent_tools.push(RecentTool {
                            name: data.name,
                            summary: truncate_str(&data.result, 50).to_string(),
                            success: data.success,
                            timestamp: event.created_at,
                        });
                    }
                }
                EventType::Error => {
                    if let Ok(data) = event.parse_data::<ErrorData>() {
                        last_error = Some((event.clone(), data));
                    }
                }
                EventType::SubAgentSpawn => {
                    if let Ok(data) = event.parse_data::<SubAgentSpawnData>() {
                        sub_agent_starts
                            .insert(data.child_session_id.clone(), (event.clone(), data));
                    }
                }
                EventType::SubAgentComplete => {
                    if let Ok(data) = event.parse_data::<super::SubAgentCompleteData>() {
                        completed_sub_agents.insert(data.child_session_id);
                    }
                }
                _ => {}
            }
        }

        // Find current task (started but not ended)
        let now = Utc::now();
        for (task_id, (start_event, start_data)) in &task_starts {
            if !task_ends.contains_key(task_id) {
                let elapsed = (now - start_event.created_at).num_seconds().max(0) as u64;
                context.current_task = Some(CurrentTask {
                    task_id: task_id.clone(),
                    description: start_data.description.clone(),
                    started_at: start_event.created_at,
                    elapsed_secs: elapsed,
                    iterations: task_iterations.get(task_id).copied().unwrap_or(0),
                    tool_calls: task_tool_calls.get(task_id).copied().unwrap_or(0),
                });
                context.current_iteration = task_iterations.get(task_id).copied();
                break; // Only one current task
            }
        }

        // Find last completed task
        if let Some((_task_id, (end_event, end_data))) = task_ends
            .iter()
            .max_by_key(|(_, (e, _))| e.created_at)
        {
            let description = task_starts
                .get(&end_data.task_id)
                .map(|(_, d)| d.description.clone())
                .unwrap_or_else(|| "Unknown task".to_string());

            context.last_completed_task = Some(CompletedTask {
                task_id: end_data.task_id.clone(),
                description,
                status: end_data.status,
                duration_secs: end_data.duration_secs,
                error: end_data.error.clone(),
                completed_at: end_event.created_at,
            });
        }

        // Set last error
        if let Some((error_event, error_data)) = last_error {
            context.last_error = Some(RecentError {
                message: error_data.message,
                error_type: format!("{:?}", error_data.error_type),
                occurred_at: error_event.created_at,
                task_context: error_data.context,
                tool_name: error_data.tool_name,
                recovered: error_data.recovered,
            });
        }

        // Set recent tools (last 10, most recent first)
        recent_tools.reverse();
        recent_tools.truncate(10);
        context.recent_tools = recent_tools;

        // Find active sub-agents
        for (child_session_id, (spawn_event, spawn_data)) in sub_agent_starts {
            if !completed_sub_agents.contains(&child_session_id) {
                context.active_sub_agents.push(ActiveSubAgent {
                    child_session_id,
                    mission: spawn_data.mission,
                    depth: spawn_data.depth,
                    started_at: spawn_event.created_at,
                });
            }
        }

        context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_context() {
        let ctx = SessionContext::default();
        assert!(ctx.is_empty());
        assert_eq!(ctx.format_for_prompt(), "");
    }

    #[test]
    fn test_context_formatting() {
        let ctx = SessionContext {
            current_task: Some(CurrentTask {
                task_id: "task_1".to_string(),
                description: "Add blog posts".to_string(),
                started_at: Utc::now(),
                elapsed_secs: 120,
                iterations: 5,
                tool_calls: 8,
            }),
            last_error: Some(RecentError {
                message: "Command failed".to_string(),
                error_type: "ToolError".to_string(),
                occurred_at: Utc::now(),
                task_context: None,
                tool_name: Some("terminal".to_string()),
                recovered: false,
            }),
            recent_tools: vec![
                RecentTool {
                    name: "terminal".to_string(),
                    summary: "git status".to_string(),
                    success: true,
                    timestamp: Utc::now(),
                },
            ],
            ..Default::default()
        };

        let formatted = ctx.format_for_prompt();
        assert!(formatted.contains("Active task"));
        assert!(formatted.contains("Add blog posts"));
        assert!(formatted.contains("Recent error"));
        assert!(formatted.contains("terminal"));
    }
}
