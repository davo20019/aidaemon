//! Event consolidation system.
//!
//! Consolidation extracts learnings from raw events and stores them
//! in long-term memory (procedures, error-solutions, expertise, episodes).
//! After consolidation, events can be pruned to manage storage.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{Duration, Utc};
use sqlx::Row;
use tracing::{info, warn};

use super::{ErrorData, TaskEndData, TaskStartData, ToolCallData, ToolResultData};
use super::{Event, EventStore, EventType, TaskStatus};
use crate::memory::binary::encode_embedding;
use crate::memory::embeddings::EmbeddingService;
use crate::plans::{PlanStore, StepStatus};
use crate::traits::{Episode, ErrorSolution, ModelProvider, Procedure};

/// Statistics from a consolidation run
#[derive(Debug, Default)]
pub struct ConsolidationResult {
    pub events_processed: usize,
    pub events_consolidated: usize,
    pub procedures_created: usize,
    pub error_solutions_created: usize,
    pub expertise_updated: usize,
    pub episodes_created: usize,
}

impl ConsolidationResult {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn add(&mut self, other: &ConsolidationResult) {
        self.events_processed += other.events_processed;
        self.events_consolidated += other.events_consolidated;
        self.procedures_created += other.procedures_created;
        self.error_solutions_created += other.error_solutions_created;
        self.expertise_updated += other.expertise_updated;
        self.episodes_created += other.episodes_created;
    }
}

/// Statistics from daily consolidation
#[derive(Debug, Default)]
pub struct DailyConsolidationStats {
    pub sessions_processed: usize,
    pub total_result: ConsolidationResult,
    pub failures: usize,
}

/// Extracts learnings from events and stores in long-term memory
pub struct Consolidator {
    event_store: Arc<EventStore>,
    plan_store: Arc<PlanStore>,
    pool: sqlx::SqlitePool,
    provider: Option<Arc<dyn ModelProvider>>,
    fast_model: String,
    embedding_service: Option<Arc<EmbeddingService>>,
}

impl Consolidator {
    pub fn new(
        event_store: Arc<EventStore>,
        plan_store: Arc<PlanStore>,
        pool: sqlx::SqlitePool,
        provider: Option<Arc<dyn ModelProvider>>,
        fast_model: String,
        embedding_service: Option<Arc<EmbeddingService>>,
    ) -> Self {
        Self {
            event_store,
            plan_store,
            pool,
            provider,
            fast_model,
            embedding_service,
        }
    }

    /// Consolidate events for a session into long-term memory
    pub async fn consolidate_session(
        &self,
        session_id: &str,
    ) -> anyhow::Result<ConsolidationResult> {
        // Get unconsolidated events
        let events = self.event_store.query_unconsolidated(session_id).await?;

        if events.is_empty() {
            return Ok(ConsolidationResult::empty());
        }

        let mut result = ConsolidationResult {
            events_processed: events.len(),
            ..Default::default()
        };

        // 1. Extract and save procedures from successful task sequences
        let procedures = self.extract_procedures(&events).await;
        for proc in procedures {
            if let Err(e) = self.save_procedure(&proc).await {
                warn!("Failed to save procedure: {}", e);
            } else {
                self.generate_trigger_embedding(&proc).await;
                result.procedures_created += 1;
            }
        }

        // 1b. Extract procedures from completed plans (higher quality than event-based)
        let since = Utc::now() - Duration::hours(24);
        if let Ok(completed_plans) = self.plan_store.get_completed_since(session_id, since).await {
            for plan in completed_plans {
                let steps: Vec<String> = plan
                    .steps
                    .iter()
                    .filter(|s| s.status == StepStatus::Completed)
                    .map(|s| s.description.clone())
                    .collect();
                if steps.len() >= 2 {
                    // Deduplication check for plan-based procedures
                    if self.is_duplicate_procedure(&plan.description).await {
                        continue;
                    }

                    let proc = Procedure {
                        id: 0,
                        name: format!("plan_{}", &plan.id[..8.min(plan.id.len())]),
                        trigger_pattern: plan.description.clone(),
                        steps: steps.clone(),
                        success_count: 1,
                        failure_count: 0,
                        avg_duration_secs: None,
                        last_used_at: Some(Utc::now()),
                        created_at: Utc::now(),
                        updated_at: Utc::now(),
                    };

                    // Try LLM enhancement for plan-based procedures too
                    let tool_sequence_str = proc
                        .steps
                        .iter()
                        .enumerate()
                        .map(|(i, s)| format!("{}. {}", i + 1, s))
                        .collect::<Vec<_>>()
                        .join("\n");

                    let final_proc = match self
                        .enhance_procedure_with_llm(&proc, &plan.description, &tool_sequence_str)
                        .await
                    {
                        Some(enhanced) => enhanced,
                        None => proc,
                    };

                    if let Err(e) = self.save_procedure(&final_proc).await {
                        warn!("Failed to save plan-based procedure: {}", e);
                    } else {
                        self.generate_trigger_embedding(&final_proc).await;
                        result.procedures_created += 1;
                    }
                }
            }
        }

        // 2. Extract and save error-solution pairs
        let error_solutions = self.extract_error_solutions(&events).await;
        for es in error_solutions {
            if let Err(e) = self.save_error_solution(&es).await {
                warn!("Failed to save error solution: {}", e);
            } else {
                result.error_solutions_created += 1;
            }
        }

        // 3. Update expertise from task outcomes
        let expertise_updates = self.extract_expertise_updates(&events).await;
        for (domain, success) in expertise_updates {
            if let Err(e) = self.update_expertise(&domain, success).await {
                warn!("Failed to update expertise: {}", e);
            } else {
                result.expertise_updated += 1;
            }
        }

        // 4. Mark events as consolidated
        // Note: Episode creation is handled exclusively by MemoryManager (LLM-quality episodes)
        let event_ids: Vec<i64> = events.iter().map(|e| e.id).collect();
        self.event_store.mark_consolidated(&event_ids).await?;
        result.events_consolidated = event_ids.len();

        info!(
            session_id,
            events = result.events_consolidated,
            procedures = result.procedures_created,
            error_solutions = result.error_solutions_created,
            "Session consolidation complete"
        );

        Ok(result)
    }

    /// Trigger: End of session/task
    pub async fn on_session_end(&self, session_id: &str) {
        if let Err(e) = self.consolidate_session(session_id).await {
            warn!("Consolidation failed for session {}: {}", session_id, e);
        }
    }

    /// Trigger: Daily cron job - consolidate all sessions with pending events
    pub async fn daily_consolidation(&self) -> anyhow::Result<DailyConsolidationStats> {
        let sessions = self
            .event_store
            .get_sessions_needing_consolidation()
            .await?;

        let mut stats = DailyConsolidationStats {
            sessions_processed: sessions.len(),
            ..Default::default()
        };

        for session_id in sessions {
            match self.consolidate_session(&session_id).await {
                Ok(result) => stats.total_result.add(&result),
                Err(e) => {
                    warn!("Consolidation failed for {}: {}", session_id, e);
                    stats.failures += 1;
                }
            }
        }

        info!(
            sessions = stats.sessions_processed,
            events = stats.total_result.events_consolidated,
            procedures = stats.total_result.procedures_created,
            failures = stats.failures,
            "Daily consolidation complete"
        );

        Ok(stats)
    }

    // =========================================================================
    // Extraction Methods
    // =========================================================================

    /// Extract procedures from successful task sequences
    pub async fn extract_procedures(&self, events: &[Event]) -> Vec<Procedure> {
        let mut procedures = Vec::new();

        // Group events by task
        let mut task_events: HashMap<String, Vec<&Event>> = HashMap::new();
        for event in events {
            if let Some(task_id) = &event.task_id {
                task_events.entry(task_id.clone()).or_default().push(event);
            }
        }

        // Find successful tasks with multiple tool calls
        for (task_id, task_evts) in task_events {
            // Check if task completed successfully
            let task_end = task_evts
                .iter()
                .find(|e| e.event_type == EventType::TaskEnd);
            let task_start = task_evts
                .iter()
                .find(|e| e.event_type == EventType::TaskStart);

            let (task_end, task_start) = match (task_end, task_start) {
                (Some(end), Some(start)) => (end, start),
                _ => continue,
            };

            let end_data: TaskEndData = match task_end.parse_data() {
                Ok(d) => d,
                Err(_) => continue,
            };

            // Only extract from successful tasks
            if end_data.status != TaskStatus::Completed {
                continue;
            }

            // Need at least 2 tool calls to make a procedure
            let tool_calls: Vec<&Event> = task_evts
                .iter()
                .filter(|e| e.event_type == EventType::ToolCall)
                .copied()
                .collect();

            if tool_calls.len() < 2 {
                continue;
            }

            // Extract task description as trigger pattern
            let start_data: TaskStartData = match task_start.parse_data() {
                Ok(d) => d,
                Err(_) => continue,
            };

            // Build steps from tool calls
            let mut steps = Vec::new();
            for tc_event in &tool_calls {
                if let Ok(tc_data) = tc_event.parse_data::<ToolCallData>() {
                    let step = format!(
                        "{}: {}",
                        tc_data.name,
                        tc_data.summary.unwrap_or_else(|| "...".to_string())
                    );
                    steps.push(step);
                }
            }

            if steps.len() >= 2 {
                // Deduplication: check if a semantically similar procedure already exists
                if self.is_duplicate_procedure(&start_data.description).await {
                    continue;
                }

                let proc = Procedure {
                    id: 0,
                    name: format!("auto_{}", &task_id[..8.min(task_id.len())]),
                    trigger_pattern: start_data.description.clone(),
                    steps: steps.clone(),
                    success_count: 1,
                    failure_count: 0,
                    avg_duration_secs: Some(end_data.duration_secs as f32),
                    last_used_at: Some(Utc::now()),
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                };

                // Try LLM enhancement
                let tool_sequence_str = proc
                    .steps
                    .iter()
                    .enumerate()
                    .map(|(i, s)| format!("{}. {}", i + 1, s))
                    .collect::<Vec<_>>()
                    .join("\n");

                let final_proc = match self
                    .enhance_procedure_with_llm(&proc, &start_data.description, &tool_sequence_str)
                    .await
                {
                    Some(enhanced) => enhanced,
                    None => proc,
                };

                procedures.push(final_proc);
            }
        }

        procedures
    }

    /// Extract error-solution pairs from recovered errors
    pub async fn extract_error_solutions(&self, events: &[Event]) -> Vec<ErrorSolution> {
        let mut solutions = Vec::new();

        // Find Error events followed by successful ToolResult
        let mut i = 0;
        while i < events.len() {
            let event = &events[i];

            if event.event_type == EventType::Error {
                if let Ok(error_data) = event.parse_data::<ErrorData>() {
                    // Look for recovery in subsequent events
                    let mut recovery_tool: Option<String> = None;
                    let mut recovery_summary: Option<String> = None;

                    for next_event in events.iter().take(events.len().min(i + 10)).skip(i + 1) {
                        if next_event.event_type == EventType::ToolResult {
                            if let Ok(result_data) = next_event.parse_data::<ToolResultData>() {
                                if result_data.success {
                                    recovery_tool = Some(result_data.name);
                                    recovery_summary = Some(truncate(&result_data.result, 200));
                                    break;
                                }
                            }
                        }
                    }

                    // If we found a recovery, create an error-solution pair
                    if let (Some(tool), Some(summary)) = (recovery_tool, recovery_summary) {
                        let solution = ErrorSolution {
                            id: 0,
                            error_pattern: error_data.message.clone(),
                            domain: error_data.tool_name.clone(),
                            solution_summary: format!("Used {} to resolve", tool),
                            solution_steps: Some(vec![summary]),
                            success_count: 1,
                            failure_count: 0,
                            last_used_at: Some(Utc::now()),
                            created_at: Utc::now(),
                        };
                        solutions.push(solution);
                    }
                }
            }
            i += 1;
        }

        solutions
    }

    /// Extract expertise updates from task outcomes
    pub async fn extract_expertise_updates(&self, events: &[Event]) -> Vec<(String, bool)> {
        let mut updates = Vec::new();

        for event in events {
            if event.event_type == EventType::TaskEnd {
                if let Ok(data) = event.parse_data::<TaskEndData>() {
                    // Infer domain from tool usage
                    let domain = self.infer_domain_from_task(&data.task_id, events);
                    if let Some(domain) = domain {
                        let success = data.status == TaskStatus::Completed;
                        updates.push((domain, success));
                    }
                }
            }
        }

        updates
    }

    /// Infer the domain from tools used in a task
    fn infer_domain_from_task(&self, task_id: &str, events: &[Event]) -> Option<String> {
        let tool_names: Vec<String> = events
            .iter()
            .filter(|e| e.task_id.as_ref() == Some(&task_id.to_string()))
            .filter(|e| e.event_type == EventType::ToolCall)
            .filter_map(|e| e.tool_name.clone())
            .collect();

        if tool_names.is_empty() {
            return None;
        }

        // Simple domain inference based on tool usage
        if tool_names.iter().any(|t| t == "terminal") {
            if tool_names.iter().any(|t| t.contains("git")) {
                return Some("git".to_string());
            }
            return Some("shell".to_string());
        }
        if tool_names
            .iter()
            .any(|t| t == "web_search" || t == "web_fetch")
        {
            return Some("research".to_string());
        }
        if tool_names.iter().any(|t| t == "browser") {
            return Some("browser".to_string());
        }

        Some("general".to_string())
    }

    /// Create an episode summary for the session
    async fn create_episode_summary(&self, session_id: &str, events: &[Event]) -> Option<Episode> {
        if events.is_empty() {
            return None;
        }

        // Get time range
        let start_time = events.first()?.created_at;
        let end_time = events.last()?.created_at;

        // Count key events
        let user_messages = events
            .iter()
            .filter(|e| e.event_type == EventType::UserMessage)
            .count();
        let task_count = events
            .iter()
            .filter(|e| e.event_type == EventType::TaskStart)
            .count();
        let error_count = events
            .iter()
            .filter(|e| e.event_type == EventType::Error)
            .count();

        // Determine outcome
        let last_task_end = events
            .iter()
            .rev()
            .find(|e| e.event_type == EventType::TaskEnd);

        let outcome = if let Some(end_event) = last_task_end {
            if let Ok(data) = end_event.parse_data::<TaskEndData>() {
                match data.status {
                    TaskStatus::Completed => "successful",
                    TaskStatus::Cancelled => "cancelled",
                    TaskStatus::Failed => "failed",
                }
            } else {
                "unknown"
            }
        } else {
            "incomplete"
        };

        // Collect topics from task descriptions
        let topics: Vec<String> = events
            .iter()
            .filter(|e| e.event_type == EventType::TaskStart)
            .filter_map(|e| e.parse_data::<TaskStartData>().ok())
            .map(|d| truncate(&d.description, 50))
            .collect();

        // Create summary
        let summary = format!(
            "Session with {} messages, {} tasks ({} errors). Outcome: {}",
            user_messages, task_count, error_count, outcome
        );

        Some(Episode {
            id: 0,
            session_id: session_id.to_string(),
            summary,
            topics: if topics.is_empty() {
                None
            } else {
                Some(topics)
            },
            emotional_tone: None,
            outcome: Some(outcome.to_string()),
            importance: calculate_importance(user_messages, task_count, error_count),
            recall_count: 0,
            last_recalled_at: None,
            message_count: user_messages as i32,
            start_time,
            end_time,
            created_at: Utc::now(),
            channel_id: None,
        })
    }

    // =========================================================================
    // LLM Enhancement Methods
    // =========================================================================

    /// Check if a semantically similar procedure already exists.
    async fn is_duplicate_procedure(&self, description: &str) -> bool {
        let Some(ref emb_service) = self.embedding_service else {
            return false;
        };
        let new_emb = match emb_service.embed(description.to_string()).await {
            Ok(emb) => emb,
            Err(_) => return false,
        };

        let rows = match sqlx::query(
            "SELECT name, trigger_embedding FROM procedures WHERE trigger_embedding IS NOT NULL",
        )
        .fetch_all(&self.pool)
        .await
        {
            Ok(rows) => rows,
            Err(_) => return false,
        };

        for row in &rows {
            let blob: Option<Vec<u8>> = row.get("trigger_embedding");
            if let Some(blob) = blob {
                if let Ok(existing_emb) = crate::memory::binary::decode_embedding(&blob) {
                    if crate::memory::math::cosine_similarity(&new_emb, &existing_emb) > 0.85 {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Use the FAST LLM model to generate a clean procedure definition.
    /// Returns None if LLM fails, returns skip=true, or is unavailable.
    async fn enhance_procedure_with_llm(
        &self,
        proc: &Procedure,
        task_description: &str,
        tool_sequence: &str,
    ) -> Option<Procedure> {
        let provider = self.provider.as_ref()?;

        let step_count = proc.steps.len();
        let duration_str = proc
            .avg_duration_secs
            .map(|d| format!("{:.0}", d))
            .unwrap_or_else(|| "unknown".to_string());

        let system_prompt = "You generate clean procedure definitions from successful task sequences.\nRespond with ONLY valid JSON.";

        let user_prompt = format!(
            "Given a task that completed successfully, generate a clean procedure definition.\n\n\
             Task: {task_description}\n\
             Tools used:\n{tool_sequence}\n\
             Duration: {duration_str}s\n\n\
             Respond with JSON:\n\
             {{\n\
               \"name\": \"descriptive-kebab-case-name\",\n\
               \"trigger_pattern\": \"general description of when to use this\",\n\
               \"steps\": [\"Step 1: ...\", \"Step 2: ...\"],\n\
               \"skip\": false\n\
             }}\n\n\
             Rules:\n\
             - name: 2-4 word kebab-case describing PURPOSE (not tools used)\n\
             - trigger_pattern: General description, not the exact task\n\
             - steps: Actionable instructions including tool names and key parameters\n\
             - skip: true if too context-specific to be reusable\n\
             - Step count must match original ({step_count} steps)"
        );

        let llm_messages = vec![
            serde_json::json!({"role": "system", "content": system_prompt}),
            serde_json::json!({"role": "user", "content": user_prompt}),
        ];

        let response = match provider.chat(&self.fast_model, &llm_messages, &[]).await {
            Ok(r) => r,
            Err(e) => {
                warn!("LLM enhancement failed for procedure: {}", e);
                return None;
            }
        };

        let text = response.content?;
        let trimmed = text.trim();
        let json_str = if let Some(start) = trimmed.find('{') {
            if let Some(end) = trimmed.rfind('}') {
                &trimmed[start..=end]
            } else {
                trimmed
            }
        } else {
            trimmed
        };

        let parsed: serde_json::Value = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(e) => {
                warn!("Failed to parse LLM procedure enhancement: {}", e);
                return None;
            }
        };

        // Check if LLM says skip
        if parsed
            .get("skip")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            return None;
        }

        let name = parsed
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let trigger_pattern = parsed
            .get("trigger_pattern")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if name.is_empty() || trigger_pattern.is_empty() {
            return None;
        }

        // Extract steps from LLM
        let llm_steps: Vec<String> = parsed
            .get("steps")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        // Step count validation: if LLM steps differ by more than 1, keep original steps
        // but use LLM's name and trigger_pattern
        let final_steps = if llm_steps.is_empty()
            || (llm_steps.len() as isize - proc.steps.len() as isize).unsigned_abs() > 1
        {
            proc.steps.clone()
        } else {
            llm_steps
        };

        Some(Procedure {
            id: proc.id,
            name,
            trigger_pattern,
            steps: final_steps,
            success_count: proc.success_count,
            failure_count: proc.failure_count,
            avg_duration_secs: proc.avg_duration_secs,
            last_used_at: proc.last_used_at,
            created_at: proc.created_at,
            updated_at: proc.updated_at,
        })
    }

    /// Generate and store a trigger embedding for a procedure.
    async fn generate_trigger_embedding(&self, proc: &Procedure) {
        let Some(ref embedding_service) = self.embedding_service else {
            return;
        };
        if let Ok(embedding) = embedding_service.embed(proc.trigger_pattern.clone()).await {
            let blob = encode_embedding(&embedding);
            let _ = sqlx::query("UPDATE procedures SET trigger_embedding = ? WHERE name = ?")
                .bind(blob)
                .bind(&proc.name)
                .execute(&self.pool)
                .await;
        }
    }

    // =========================================================================
    // Storage Methods
    // =========================================================================

    pub async fn save_procedure(&self, proc: &Procedure) -> anyhow::Result<()> {
        let steps_json = serde_json::to_string(&proc.steps)?;
        let now = Utc::now().to_rfc3339();

        // Try to update existing or insert new
        sqlx::query(
            r#"
            INSERT INTO procedures (name, trigger_pattern, steps, success_count, failure_count,
                                    avg_duration_secs, last_used_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                success_count = success_count + 1,
                last_used_at = excluded.last_used_at,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(&proc.name)
        .bind(&proc.trigger_pattern)
        .bind(&steps_json)
        .bind(proc.success_count)
        .bind(proc.failure_count)
        .bind(proc.avg_duration_secs)
        .bind(&now)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn save_error_solution(&self, es: &ErrorSolution) -> anyhow::Result<()> {
        let steps_json = es
            .solution_steps
            .as_ref()
            .map(|s| serde_json::to_string(s).unwrap_or_default());
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            r#"
            INSERT INTO error_solutions (error_pattern, domain, solution_summary, solution_steps,
                                         success_count, failure_count, last_used_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&es.error_pattern)
        .bind(&es.domain)
        .bind(&es.solution_summary)
        .bind(&steps_json)
        .bind(es.success_count)
        .bind(es.failure_count)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn update_expertise(&self, domain: &str, success: bool) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();

        // Try to update existing
        let result = if success {
            sqlx::query(
                r#"
                UPDATE expertise
                SET tasks_attempted = tasks_attempted + 1,
                    tasks_succeeded = tasks_succeeded + 1,
                    last_task_at = ?,
                    updated_at = ?
                WHERE domain = ?
                "#,
            )
            .bind(&now)
            .bind(&now)
            .bind(domain)
            .execute(&self.pool)
            .await?
        } else {
            sqlx::query(
                r#"
                UPDATE expertise
                SET tasks_attempted = tasks_attempted + 1,
                    tasks_failed = tasks_failed + 1,
                    last_task_at = ?,
                    updated_at = ?
                WHERE domain = ?
                "#,
            )
            .bind(&now)
            .bind(&now)
            .bind(domain)
            .execute(&self.pool)
            .await?
        };

        // If no rows updated, insert new
        if result.rows_affected() == 0 {
            sqlx::query(
                r#"
                INSERT INTO expertise (domain, tasks_attempted, tasks_succeeded, tasks_failed,
                                       current_level, confidence_score, last_task_at, created_at, updated_at)
                VALUES (?, 1, ?, ?, 'novice', 0.1, ?, ?, ?)
                "#
            )
            .bind(domain)
            .bind(if success { 1 } else { 0 })
            .bind(if success { 0 } else { 1 })
            .bind(&now)
            .bind(&now)
            .bind(&now)
            .execute(&self.pool)
            .await?;
        }

        Ok(())
    }

    async fn save_episode(&self, episode: &Episode) -> anyhow::Result<()> {
        let topics_json = episode
            .topics
            .as_ref()
            .map(|t| serde_json::to_string(t).unwrap_or_default());

        sqlx::query(
            r#"
            INSERT INTO episodes (session_id, summary, topics, emotional_tone, outcome,
                                  importance, recall_count, message_count, start_time, end_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#
        )
        .bind(&episode.session_id)
        .bind(&episode.summary)
        .bind(&topics_json)
        .bind(&episode.emotional_tone)
        .bind(&episode.outcome)
        .bind(episode.importance)
        .bind(episode.recall_count)
        .bind(episode.message_count)
        .bind(episode.start_time.to_rfc3339())
        .bind(episode.end_time.to_rfc3339())
        .bind(episode.created_at.to_rfc3339())
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

/// Pruner for cleaning up old consolidated events
pub struct Pruner {
    event_store: Arc<EventStore>,
    consolidator: Arc<Consolidator>,
    retention_days: u32,
}

impl Pruner {
    pub fn new(
        event_store: Arc<EventStore>,
        consolidator: Arc<Consolidator>,
        retention_days: u32,
    ) -> Self {
        Self {
            event_store,
            consolidator,
            retention_days,
        }
    }

    /// Prune old events (consolidate first if needed, then delete)
    pub async fn prune(&self) -> anyhow::Result<PruneStats> {
        let cutoff = Utc::now() - Duration::days(self.retention_days as i64);

        // First, consolidate any old unconsolidated events
        let old_sessions = self
            .event_store
            .get_sessions_with_old_unconsolidated_events(cutoff)
            .await?;

        let mut consolidation_errors = 0;
        for session_id in old_sessions {
            if let Err(e) = self.consolidator.consolidate_session(&session_id).await {
                warn!(
                    "Failed to consolidate before prune for {}: {}",
                    session_id, e
                );
                consolidation_errors += 1;
            }
        }

        // Now prune consolidated events older than cutoff
        let deleted = self.event_store.delete_old_consolidated(cutoff).await?;

        info!(
            deleted,
            retention_days = self.retention_days,
            "Event pruning complete"
        );

        Ok(PruneStats {
            deleted,
            consolidation_errors,
        })
    }
}

#[derive(Debug, Default)]
pub struct PruneStats {
    pub deleted: u64,
    pub consolidation_errors: usize,
}

// Helper functions

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

fn calculate_importance(messages: usize, tasks: usize, errors: usize) -> f32 {
    // Simple importance heuristic
    let base = 0.3;
    let message_factor = (messages as f32 * 0.05).min(0.3);
    let task_factor = (tasks as f32 * 0.1).min(0.3);
    let error_factor = (errors as f32 * 0.05).min(0.1);

    (base + message_factor + task_factor + error_factor).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── truncate tests ──

    #[test]
    fn test_truncate_short_string() {
        let result = truncate("hello", 10);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_truncate_long_string() {
        let result = truncate("hello world, this is a long string", 10);
        // max_len=10, saturating_sub(3)=7, so first 7 chars + "..."
        assert_eq!(result, "hello w...");
        assert_eq!(result.len(), 10);
    }

    // ── calculate_importance tests ──

    #[test]
    fn test_calculate_importance_base() {
        let result = calculate_importance(0, 0, 0);
        assert!((result - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_calculate_importance_capped() {
        // Very high values: messages=100 (factor 5.0 capped to 0.3),
        // tasks=100 (factor 10.0 capped to 0.3), errors=100 (factor 5.0 capped to 0.1)
        // total = 0.3 + 0.3 + 0.3 + 0.1 = 1.0, capped at 1.0
        let result = calculate_importance(100, 100, 100);
        assert!((result - 1.0).abs() < f32::EPSILON);
    }
}
