//! Event consolidation system.
//!
//! Consolidation extracts learnings from raw events and stores them
//! in long-term memory (procedures, error-solutions, expertise, behavior
//! patterns, episodes).
//! After consolidation, events can be pruned to manage storage.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{Duration, Utc};
use sqlx::Row;
use tracing::{info, warn};

use super::{
    DecisionPointData, ErrorData, TaskEndData, TaskStartData, ToolCallData, ToolResultData,
};
use super::{Event, EventStore, EventType, TaskOutcome};
use crate::llm_runtime::SharedLlmRuntime;
use crate::memory::binary::encode_embedding;
use crate::memory::embeddings::EmbeddingService;
use crate::plans::{PlanStore, StepStatus};
use crate::traits::{ErrorSolution, Procedure};

/// Statistics from a consolidation run
#[derive(Debug, Default)]
pub struct ConsolidationResult {
    pub events_processed: usize,
    pub events_consolidated: usize,
    pub procedures_created: usize,
    pub error_solutions_created: usize,
    pub expertise_updated: usize,
    pub behavior_patterns_recorded: usize,
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
        self.behavior_patterns_recorded += other.behavior_patterns_recorded;
        self.episodes_created += other.episodes_created;
    }
}

#[derive(Debug, Clone)]
struct FailurePatternObservation {
    trigger_context: String,
    action: String,
    description: String,
    confidence: f32,
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
    llm_runtime: Option<SharedLlmRuntime>,
    embedding_service: Option<Arc<EmbeddingService>>,
    state: Option<Arc<dyn crate::traits::StateStore>>,
    memory: Option<Arc<dyn aidaemon_memory::stores::MemoryStores>>,
    learning_evidence_gate_enforce: bool,
}

impl Consolidator {
    pub fn new(
        event_store: Arc<EventStore>,
        plan_store: Arc<PlanStore>,
        pool: sqlx::SqlitePool,
        llm_runtime: Option<SharedLlmRuntime>,
        embedding_service: Option<Arc<EmbeddingService>>,
    ) -> Self {
        Self {
            event_store,
            plan_store,
            pool,
            llm_runtime,
            embedding_service,
            state: None,
            memory: None,
            learning_evidence_gate_enforce: false,
        }
    }

    /// Set the state store for token usage tracking.
    pub fn with_state(mut self, state: Arc<dyn crate::traits::StateStore>) -> Self {
        self.state = Some(state);
        self
    }

    /// Set the memory the consolidator's lessons are written to.
    pub fn with_memory(mut self, memory: Arc<dyn aidaemon_memory::stores::MemoryStores>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_learning_evidence_gate(mut self, enforce: bool) -> Self {
        self.learning_evidence_gate_enforce = enforce;
        self
    }

    /// Mark a session's events as handled without learning from them, which
    /// also makes them eligible for retention.
    pub async fn retire_session(&self, session_id: &str) -> anyhow::Result<ConsolidationResult> {
        let event_ids: Vec<i64> = sqlx::query_scalar(
            "SELECT id FROM events
             WHERE session_id = ? AND consolidated_at IS NULL",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;
        if event_ids.is_empty() {
            return Ok(ConsolidationResult::empty());
        }
        self.event_store.mark_consolidated(&event_ids).await?;
        info!(
            session_id,
            events = event_ids.len(),
            "Marked events consolidated without learning"
        );
        Ok(ConsolidationResult {
            events_processed: event_ids.len(),
            events_consolidated: event_ids.len(),
            ..Default::default()
        })
    }
    /// Consolidate events for a session into long-term memory
    pub async fn consolidate_session(
        &self,
        session_id: &str,
    ) -> anyhow::Result<ConsolidationResult> {
        // Resolve isolation before deserializing any event payload. This also
        // makes malformed legacy JSON fail closed: the provenance resolver
        // classifies the session as isolated, and we can retire its rows by ID
        // without asking EventStore to parse the unreadable payload.
        let memory_isolated = super::is_memory_isolated_worker_session(session_id)
            || super::session_has_mandate_execution_provenance(&self.pool, session_id).await?;
        if memory_isolated {
            return self.retire_session(session_id).await;
        }

        // Get unconsolidated events, then remove every task whose durable
        // capability policy suppresses automatic memory. Canonical events stay
        // available for history/audit; they are merely retired from learning.
        let all_events = self.event_store.query_unconsolidated(session_id).await?;

        let suppressed_task_ids: std::collections::HashSet<String> = sqlx::query_scalar(
            "SELECT DISTINCT task_id FROM events
             WHERE session_id = ? AND event_type = 'memory_policy_compiled'
               AND task_id IS NOT NULL
               AND json_extract(data, '$.access') = 'suppressed'",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .collect();
        let allowed_task_ids: std::collections::HashSet<String> = sqlx::query_scalar(
            "SELECT DISTINCT task_id FROM events
             WHERE session_id = ? AND event_type = 'memory_policy_compiled'
               AND task_id IS NOT NULL
               AND json_extract(data, '$.access') = 'allowed'",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .collect();
        let (suppressed_events, policy_eligible): (Vec<_>, Vec<_>) =
            all_events.into_iter().partition(|event| {
                event
                    .task_id
                    .as_ref()
                    .is_some_and(|task_id| suppressed_task_ids.contains(task_id))
            });
        let events = policy_eligible
            .into_iter()
            .filter(|event| {
                event
                    .task_id
                    .as_ref()
                    .is_none_or(|task_id| allowed_task_ids.contains(task_id))
            })
            .collect::<Vec<_>>();
        if !suppressed_events.is_empty() {
            let ids = suppressed_events
                .iter()
                .map(|event| event.id)
                .collect::<Vec<_>>();
            self.event_store.mark_consolidated(&ids).await?;
        }

        if events.is_empty() {
            return Ok(ConsolidationResult {
                events_processed: suppressed_events.len(),
                events_consolidated: suppressed_events.len(),
                ..Default::default()
            });
        }

        let mut result = ConsolidationResult {
            events_processed: events.len() + suppressed_events.len(),
            events_consolidated: suppressed_events.len(),
            ..Default::default()
        };

        // 1. Extract and save procedures from successful task sequences
        let procedures = self.extract_procedures(&events).await;
        for proc in procedures {
            if self.learning_evidence_gate_enforce
                && !aidaemon_memory::trajectory::meets_evidence_gate(&proc)
            {
                info!(
                    trigger = %proc.trigger_pattern,
                    steps = proc.steps.len(),
                    "Skipping procedure without sufficient evidence"
                );
                continue;
            }
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
                if plan
                    .task_id
                    .as_ref()
                    .is_some_and(|task_id| suppressed_task_ids.contains(task_id))
                {
                    continue;
                }
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
                    let Some(proc) = aidaemon_memory::trajectory::procedure_from_plan(
                        &plan.id,
                        &plan.description,
                        steps,
                    ) else {
                        continue;
                    };
                    let final_proc = self.enhanced(proc, &plan.description).await;
                    if self.learning_evidence_gate_enforce
                        && !aidaemon_memory::trajectory::meets_evidence_gate(&final_proc)
                    {
                        info!(
                            trigger = %final_proc.trigger_pattern,
                            "Skipping plan-based procedure without evidence"
                        );
                        continue;
                    }

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

        // 4. Mine warning/error decision points into durable failure patterns
        result.behavior_patterns_recorded =
            self.extract_failure_patterns_from_decisions(&events).await;

        // 4.5. Extract and stamp fact provenance
        self.extract_and_stamp_fact_provenance(&events).await;

        // 5. Mark events as consolidated
        // Note: Episode creation is handled exclusively by MemoryManager (LLM-quality episodes)
        let event_ids: Vec<i64> = events.iter().map(|e| e.id).collect();
        self.event_store.mark_consolidated(&event_ids).await?;
        result.events_consolidated = result.events_consolidated.saturating_add(event_ids.len());

        info!(
            session_id,
            events = result.events_consolidated,
            procedures = result.procedures_created,
            error_solutions = result.error_solutions_created,
            behavior_patterns = result.behavior_patterns_recorded,
            "Session consolidation complete"
        );

        Ok(result)
    }

    async fn extract_and_stamp_fact_provenance(&self, events: &[Event]) {
        let mut last_user_message = None;
        let mut last_msg_time = None;

        for event in events {
            if event.event_type == crate::events::EventType::UserMessage {
                if let Ok(data) = event.parse_data::<super::UserMessageData>() {
                    if data
                        .annotations
                        .contains(&crate::traits::MessageAnnotation::InternalContinuation)
                    {
                        continue;
                    }
                    last_user_message = Some(data.content);
                    last_msg_time = Some(event.created_at);
                }
            } else if event.event_type == crate::events::EventType::ToolCall
                && event.tool_name.as_deref() == Some("remember_fact")
            {
                if let Ok(data) = event.parse_data::<super::ToolCallData>() {
                    if let Ok(args) = serde_json::from_value::<serde_json::Value>(data.arguments) {
                        let mut entries = Vec::new();

                        if let Some(facts_arr) = args.get("facts").and_then(|v| v.as_array()) {
                            for f in facts_arr {
                                if let (Some(c), Some(k), Some(v)) = (
                                    f.get("category").and_then(|val| val.as_str()),
                                    f.get("key").and_then(|val| val.as_str()),
                                    f.get("value").and_then(|val| val.as_str()),
                                ) {
                                    entries.push((c.to_string(), k.to_string(), v.to_string()));
                                }
                            }
                        } else if let (Some(c), Some(k), Some(v)) = (
                            args.get("category").and_then(|val| val.as_str()),
                            args.get("key").and_then(|val| val.as_str()),
                            args.get("value").and_then(|val| val.as_str()),
                        ) {
                            entries.push((c.to_string(), k.to_string(), v.to_string()));
                        }

                        for (category, key, value) in entries {
                            let mut source = "derived";
                            let mut source_excerpt = None;
                            let mut first_seen_at = None;

                            if let Some(msg) = &last_user_message {
                                let lower_msg = msg.to_lowercase();
                                let lower_val = value.to_lowercase();
                                if lower_val.len() > 3 && lower_msg.contains(&lower_val) {
                                    source = "user_stated";
                                }
                                source_excerpt = Some(crate::utils::truncate_str(msg, 200));
                                first_seen_at = last_msg_time;
                            }

                            let first_seen_str = first_seen_at.map(|dt| dt.to_rfc3339());
                            let _ = sqlx::query(
                                "UPDATE facts
                                 SET source = ?, first_seen_at = ?, source_excerpt = ?
                                 WHERE category = ? AND key = ? AND superseded_at IS NULL",
                            )
                            .bind(source)
                            .bind(&first_seen_str)
                            .bind(&source_excerpt)
                            .bind(&category)
                            .bind(&key)
                            .execute(&self.pool)
                            .await;
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // Extraction Methods
    // =========================================================================

    /// Procedures from successful task sequences (the SDK memory crate's
    /// extraction over this session's trace), deduplicated against stored
    /// triggers and enhanced by the model when available.
    pub async fn extract_procedures(&self, events: &[Event]) -> Vec<Procedure> {
        let trace = trace_events(events);
        let mut procedures = Vec::new();
        for candidate in aidaemon_memory::trajectory::procedure_candidates(&trace) {
            if self.is_duplicate_procedure(&candidate.description).await {
                continue;
            }
            let procedure = aidaemon_memory::trajectory::procedure_from_task(&candidate);
            procedures.push(self.enhanced(procedure, &candidate.description).await);
        }
        procedures
    }

    /// Error-solution pairs from recovered errors.
    pub async fn extract_error_solutions(&self, events: &[Event]) -> Vec<ErrorSolution> {
        aidaemon_memory::trajectory::error_solutions(&trace_events(events))
    }

    /// Expertise observations from task outcomes.
    pub async fn extract_expertise_updates(&self, events: &[Event]) -> Vec<(String, bool)> {
        aidaemon_memory::trajectory::expertise_updates(&trace_events(events))
    }

    /// Mine operational warning/error decision points into durable failure patterns.
    async fn extract_failure_patterns_from_decisions(&self, events: &[Event]) -> usize {
        let Some(memory) = self.memory.as_ref() else {
            return 0;
        };

        let mut task_events: HashMap<String, Vec<&Event>> = HashMap::new();
        for event in events {
            if let Some(task_id) = &event.task_id {
                task_events.entry(task_id.clone()).or_default().push(event);
            }
        }

        let mut recorded = 0usize;

        for task_evts in task_events.into_values() {
            let task_outcome = task_evts
                .iter()
                .rev()
                .find(|event| event.event_type == EventType::TaskEnd)
                .and_then(|event| event.parse_data::<TaskEndData>().ok())
                .map(|data| data.effective_outcome());
            let mut seen_for_task: HashSet<(String, String)> = HashSet::new();

            for (idx, event) in task_evts.iter().enumerate() {
                if event.event_type != EventType::DecisionPoint {
                    continue;
                }

                let Ok(decision) = event.parse_data::<DecisionPointData>() else {
                    continue;
                };
                if !decision.severity.is_warning_or_higher() {
                    continue;
                }

                let recovery_tool = next_successful_tool_after(&task_evts, idx);
                let Some(pattern) = build_failure_pattern_observation(
                    &decision,
                    task_outcome,
                    recovery_tool.as_deref(),
                ) else {
                    continue;
                };

                let dedup_key = (pattern.trigger_context.clone(), pattern.action.clone());
                if !seen_for_task.insert(dedup_key) {
                    continue;
                }

                match memory
                    .record_behavior_pattern(
                        "failure",
                        &pattern.description,
                        Some(&pattern.trigger_context),
                        Some(&pattern.action),
                        pattern.confidence,
                        1,
                    )
                    .await
                {
                    Ok(()) => recorded += 1,
                    Err(e) => warn!(
                        trigger_context = %pattern.trigger_context,
                        action = %pattern.action,
                        error = %e,
                        "Failed to record failure pattern from decision point"
                    ),
                }
            }
        }

        recorded
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

        let existing: Vec<Vec<f32>> = rows
            .iter()
            .filter_map(|row| row.get::<Option<Vec<u8>>, _>("trigger_embedding"))
            .filter_map(|blob| crate::memory::binary::decode_embedding(&blob).ok())
            .collect();
        aidaemon_memory::trajectory::duplicates_existing_trigger(&new_emb, &existing)
    }

    /// `procedure` rewritten by the model into a reusable one (the SDK
    /// memory crate's [`ProcedureEnhancer`]), or unchanged when there is no
    /// model or it declines.
    ///
    /// [`ProcedureEnhancer`]: aidaemon_memory::trajectory::ProcedureEnhancer
    async fn enhanced(&self, procedure: Procedure, task_description: &str) -> Procedure {
        let Some(runtime) = self.llm_runtime.as_ref() else {
            return procedure;
        };
        let snapshot = runtime.snapshot();
        let model = snapshot.primary_model();
        let provider: Arc<dyn aidaemon_core::Provider> = match &self.state {
            Some(state) => Arc::new(crate::agent::AccountedProvider {
                inner: snapshot.provider(),
                accounting: crate::agent::ModelCallAccounting {
                    state: state.clone(),
                    event_store: self.event_store.clone(),
                    purpose: "consolidation",
                },
            }),
            None => snapshot.provider(),
        };
        aidaemon_memory::trajectory::ProcedureEnhancer::new(provider, model)
            .enhance(&procedure, task_description)
            .await
            .unwrap_or(procedure)
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
        let domain = es.domain.clone().unwrap_or_default();
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            r#"
            INSERT INTO error_solutions (error_pattern, domain, solution_summary, solution_steps,
                                         success_count, failure_count, last_used_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(error_pattern, domain, solution_summary) DO UPDATE SET
                solution_steps = COALESCE(excluded.solution_steps, error_solutions.solution_steps),
                success_count = error_solutions.success_count + excluded.success_count,
                failure_count = error_solutions.failure_count + excluded.failure_count,
                last_used_at = excluded.last_used_at
            "#,
        )
        .bind(&es.error_pattern)
        .bind(&domain)
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
}

/// Consolidates old event sessions so the configured retention manager can
/// later remove eligible rows. It does not own retention policy.
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

    /// Consolidate old events. Deletion is exclusively owned by
    /// `RetentionManager`, including the `messages_days = 0` permanent mode.
    pub async fn prune(&self) -> anyhow::Result<PruneStats> {
        self.pass(true).await
    }

    /// Hand old events to retention without learning from them, for when
    /// another memory system does the learning.
    pub async fn retire_old_events(&self) -> anyhow::Result<PruneStats> {
        self.pass(false).await
    }

    async fn pass(&self, learn: bool) -> anyhow::Result<PruneStats> {
        let cutoff = Utc::now() - Duration::days(self.retention_days as i64);

        // First, consolidate any old unconsolidated events
        let old_sessions = self
            .event_store
            .get_sessions_with_old_unconsolidated_events(cutoff)
            .await?;

        let mut consolidation_errors = 0;
        for session_id in old_sessions {
            let handled = if learn {
                self.consolidator.consolidate_session(&session_id).await
            } else {
                self.consolidator.retire_session(&session_id).await
            };
            if let Err(e) = handled {
                warn!(
                    "Failed to consolidate before prune for {}: {}",
                    session_id, e
                );
                consolidation_errors += 1;
            }
        }

        info!(
            deleted = 0,
            retention_days = self.retention_days,
            "Event consolidation pass complete; retention manager owns deletion"
        );

        Ok(PruneStats {
            deleted: 0,
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
        let target = max_len.saturating_sub(3);
        let safe_end = s
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| i <= target)
            .last()
            .unwrap_or(0);
        format!("{}...", &s[..safe_end])
    }
}

fn next_successful_tool_after(task_events: &[&Event], start_idx: usize) -> Option<String> {
    task_events.iter().skip(start_idx + 1).find_map(|event| {
        if event.event_type != EventType::ToolResult {
            return None;
        }
        let data = event.parse_data::<ToolResultData>().ok()?;
        data.succeeded().then_some(data.name)
    })
}

fn build_failure_pattern_observation(
    decision: &DecisionPointData,
    task_outcome: Option<TaskOutcome>,
    recovery_tool: Option<&str>,
) -> Option<FailurePatternObservation> {
    let code = decision
        .code
        .as_deref()
        .unwrap_or(decision.decision_type.as_str());
    let metadata = &decision.metadata;
    let recovery_suffix = recovery_tool
        .map(|tool| format!(" Previous successful recoveries switched to {}.", tool))
        .unwrap_or_default();
    let with_confidence = |base: f32| failure_pattern_confidence(base, task_outcome, recovery_tool);

    match code {
        "repetitive_call_detection" => {
            let tool = metadata
                .get("tool")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown_tool");
            let count = metadata.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
            let action_kind = metadata
                .get("action")
                .and_then(|v| v.as_str())
                .unwrap_or("redirect");
            let action = if action_kind == "hard_stop" {
                "stop_retrying_identical_calls"
            } else {
                "pivot_to_alternate_tool_or_strategy"
            };
            let description = if count > 0 {
                format!(
                    "After {} repeated calls to {}, stop retrying the same operation and pivot to a different tool or summarize the blocker.{}",
                    count, tool, recovery_suffix
                )
            } else {
                format!(
                    "Repeated calls to {} tend to become a dead-end loop; pivot to a different tool or summarize the blocker sooner.{}",
                    tool, recovery_suffix
                )
            };
            Some(FailurePatternObservation {
                trigger_context: format!("{}:{}", code, tool),
                action: action.to_string(),
                description,
                confidence: with_confidence(if action_kind == "hard_stop" { 0.74 } else { 0.62 }),
            })
        }
        "consecutive_same_tool_detection" => {
            let tool = metadata
                .get("tool")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown_tool");
            let consecutive_count = metadata
                .get("consecutive_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let unique_args = metadata
                .get("unique_args")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let is_diverse = metadata
                .get("is_diverse")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let description = if is_diverse {
                format!(
                    "Even varied {} calls can become a loop after {} consecutive uses; switch tools or summarize progress before continuing.{}",
                    tool, consecutive_count, recovery_suffix
                )
            } else {
                format!(
                    "Long {} streaks with only {} distinct argument sets usually indicate a loop; change tools or adjust strategy before continuing.{}",
                    tool, unique_args, recovery_suffix
                )
            };
            Some(FailurePatternObservation {
                trigger_context: format!("{}:{}", code, tool),
                action: "switch_tools_before_long_streaks".to_string(),
                description,
                confidence: with_confidence(if is_diverse { 0.54 } else { 0.68 }),
            })
        }
        "alternating_pattern_detection" => {
            let mut tools: Vec<String> = metadata
                .get("tools")
                .and_then(|v| v.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            if tools.is_empty() {
                tools.push("unknown_pair".to_string());
            }
            tools.sort();
            let tool_pair = tools.join(" <-> ");
            Some(FailurePatternObservation {
                trigger_context: format!("{}:{}", code, tools.join("+")),
                action: "break_alternating_tool_loops".to_string(),
                description: format!(
                    "Alternating between {} with low call diversity usually indicates a loop; commit to one path or choose a third tool instead of bouncing between them.{}",
                    tool_pair, recovery_suffix
                ),
                confidence: with_confidence(0.72),
            })
        }
        "tool_budget_block" => {
            let tool = metadata
                .get("tool")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown_tool");
            let semantic_failures = metadata
                .get("prior_semantic_failures")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let prior_calls = metadata
                .get("prior_calls")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let description = if semantic_failures > 0 {
                format!(
                    "Once {} has repeated the same semantic failure {} times, stop retrying it and switch strategy earlier.{}",
                    tool, semantic_failures, recovery_suffix
                )
            } else {
                format!(
                    "After {} repeated {} calls without enough progress, stop retrying the blocked tool and switch strategy earlier.{}",
                    prior_calls, tool, recovery_suffix
                )
            };
            Some(FailurePatternObservation {
                trigger_context: format!("{}:{}", code, tool),
                action: "stop_retrying_blocked_tool".to_string(),
                description,
                confidence: with_confidence(if semantic_failures > 0 { 0.74 } else { 0.68 }),
            })
        }
        "missing_pre_execution_evidence" => {
            let tool = metadata
                .get("tool")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown_tool");
            let evidence_kind = metadata
                .get("required_evidence_kind")
                .and_then(|v| v.as_str())
                .unwrap_or("direct evidence");
            let target = metadata
                .get("target")
                .and_then(|v| v.as_str())
                .unwrap_or("current target");
            Some(FailurePatternObservation {
                trigger_context: format!("{}:{}:{}", code, tool, evidence_kind),
                action: "gather_direct_evidence_before_mutation".to_string(),
                description: format!(
                    "Before using {}, gather {} for {} instead of mutating from memory or inference alone.{}",
                    tool, evidence_kind, target, recovery_suffix
                ),
                confidence: with_confidence(0.82),
            })
        }
        "plan_rejected" => Some(FailurePatternObservation {
            trigger_context: code.to_string(),
            action: "replan_first_risky_step_before_execution".to_string(),
            description: format!(
                "When the structured first-step plan is rejected, revise the target or first action before executing again.{}",
                recovery_suffix
            ),
            confidence: with_confidence(0.74),
        }),
        "critique_rejected" => Some(FailurePatternObservation {
            trigger_context: code.to_string(),
            action: "address_critique_before_retry".to_string(),
            description: format!(
                "If the critique pass rejects the first risky action, fix the missing evidence or target assumptions before retrying.{}",
                recovery_suffix
            ),
            confidence: with_confidence(0.76),
        }),
        "target_scope_violation" => {
            let tool = metadata
                .get("tool")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown_tool");
            Some(FailurePatternObservation {
                trigger_context: format!("{}:{}", code, tool),
                action: "confirm_target_scope_before_mutation".to_string(),
                description: format!(
                    "If {} attempts to act outside the compiled target scope, stop and confirm the correct repo/file target before retrying.{}",
                    tool, recovery_suffix
                ),
                confidence: with_confidence(0.86),
            })
        }
        "tool_contract_violation" => {
            let tool = metadata
                .get("tool")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown_tool");
            Some(FailurePatternObservation {
                trigger_context: format!("{}:{}", code, tool),
                action: "fix_tool_arguments_before_retry".to_string(),
                description: format!(
                    "Deterministic argument-contract failures on {} should be corrected locally before retrying the step.{}",
                    tool, recovery_suffix
                ),
                confidence: with_confidence(0.72),
            })
        }
        "contradictory_file_evidence" => Some(FailurePatternObservation {
            trigger_context: code.to_string(),
            action: "recheck_conflicting_state_before_completion".to_string(),
            description: format!(
                "When file evidence contradicts itself, re-read the target before claiming completion instead of trusting stale state.{}",
                recovery_suffix
            ),
            confidence: with_confidence(0.80),
        }),
        "verification_pending" | "verification_unavailable_in_phase" => {
            Some(FailurePatternObservation {
                trigger_context: code.to_string(),
                action: "run_verification_before_claiming_success".to_string(),
                description: format!(
                    "Do not claim success while verification is pending; either run the check or surface a precise partial result.{}",
                    recovery_suffix
                ),
                confidence: with_confidence(0.82),
            })
        }
        "validation_budget_exhausted" | "execution_budget_exhausted" => {
            Some(FailurePatternObservation {
                trigger_context: code.to_string(),
                action: "reduce_scope_when_budget_exhausts".to_string(),
                description: format!(
                    "When execution or validation budget is exhausted, reduce scope or checkpoint progress instead of pushing through unverified completion.{}",
                    recovery_suffix
                ),
                confidence: with_confidence(0.70),
            })
        }
        "pre_tool_deferral_stall" => Some(FailurePatternObservation {
            trigger_context: code.to_string(),
            action: "ask_clarifying_question_or_take_next_tool_step".to_string(),
            description: format!(
                "Repeatedly deferring before taking any tool step leads to failure; ask one concrete clarification question or take the next tool action earlier.{}",
                recovery_suffix
            ),
            confidence: with_confidence(0.72),
        }),
        "stall" => {
            let stall_mode = metadata
                .get("stall_mode")
                .and_then(|v| v.as_str())
                .unwrap_or("generic");
            Some(FailurePatternObservation {
                trigger_context: format!("{}:{}", code, stall_mode),
                action: "summarize_progress_or_change_strategy".to_string(),
                description: format!(
                    "When iterations stop producing new evidence, stop looping, summarize what is known, and change strategy instead of continuing the same pattern.{}",
                    recovery_suffix
                ),
                confidence: with_confidence(0.76),
            })
        }
        "hard_iteration_cap" => Some(FailurePatternObservation {
            trigger_context: code.to_string(),
            action: "finish_before_hard_iteration_cap".to_string(),
            description: format!(
                "As the hard iteration cap approaches, stop repeating work and return the best partial answer or next concrete step earlier.{}",
                recovery_suffix
            ),
            confidence: with_confidence(0.60),
        }),
        "task_timeout" => Some(FailurePatternObservation {
            trigger_context: code.to_string(),
            action: "checkpoint_and_reduce_scope_before_timeout".to_string(),
            description: format!(
                "Long-running tasks that approach the timeout should checkpoint progress and reduce scope earlier instead of continuing the same loop.{}",
                recovery_suffix
            ),
            confidence: with_confidence(0.64),
        }),
        "task_token_budget" => Some(FailurePatternObservation {
            trigger_context: code.to_string(),
            action: "reduce_token_spend_and_checkpoint".to_string(),
            description: format!(
                "When task token spend stays high, reduce scope, checkpoint findings, or stop earlier instead of consuming the full task budget.{}",
                recovery_suffix
            ),
            confidence: with_confidence(0.68),
        }),
        "scheduled_run_budget" => Some(FailurePatternObservation {
            trigger_context: code.to_string(),
            action: "checkpoint_scheduled_run_before_budget_exhaustion".to_string(),
            description: format!(
                "Scheduled runs should bail out once progress slows; checkpoint the current state instead of burning the full per-run token budget.{}",
                recovery_suffix
            ),
            confidence: with_confidence(0.70),
        }),
        "goal_daily_token_budget" => Some(FailurePatternObservation {
            trigger_context: code.to_string(),
            action: "checkpoint_goal_work_before_daily_budget_exhaustion".to_string(),
            description: format!(
                "For recurring goal work, avoid spending the full daily budget in one pass; checkpoint progress and defer the rest earlier.{}",
                recovery_suffix
            ),
            confidence: with_confidence(0.70),
        }),
        "daily_token_budget" => Some(FailurePatternObservation {
            trigger_context: code.to_string(),
            action: "checkpoint_before_daily_budget_exhaustion".to_string(),
            description: format!(
                "When the session is nearing the daily token budget, checkpoint useful progress and stop earlier instead of exhausting the remaining budget.{}",
                recovery_suffix
            ),
            confidence: with_confidence(0.70),
        }),
        "task_token_budget_warning" => Some(FailurePatternObservation {
            trigger_context: code.to_string(),
            action: "reduce_token_spend_before_budget_warning".to_string(),
            description: format!(
                "High token-spend phases should trigger earlier summarization or scope reduction before the task budget is exhausted.{}",
                recovery_suffix
            ),
            confidence: with_confidence(0.45),
        }),
        "soft_iteration_warning" => Some(FailurePatternObservation {
            trigger_context: code.to_string(),
            action: "adjust_strategy_before_soft_iteration_warning".to_string(),
            description: format!(
                "When the soft iteration warning appears, summarize progress or change strategy instead of continuing the same loop.{}",
                recovery_suffix
            ),
            confidence: with_confidence(0.45),
        }),
        _ => None,
    }
}

fn failure_pattern_confidence(
    base: f32,
    task_outcome: Option<TaskOutcome>,
    recovery_tool: Option<&str>,
) -> f32 {
    let outcome_bonus = match task_outcome {
        Some(TaskOutcome::Failed) => 0.08,
        Some(TaskOutcome::Partial) => 0.04,
        Some(TaskOutcome::Succeeded) => 0.0,
        None => 0.0,
    };
    let recovery_bonus = if recovery_tool.is_some() { 0.03 } else { 0.0 };
    (base + outcome_bonus + recovery_bonus).clamp(0.1, 0.96)
}

/// This session's events as the SDK memory crate's task trace. Error
/// messages lose the daemon's own scaffolding lines (untrusted-data
/// envelopes, truncation notices), which would otherwise make unrelated
/// errors share a pattern.
fn trace_events(events: &[Event]) -> Vec<aidaemon_memory::trajectory::TraceEvent> {
    use aidaemon_memory::trajectory::{TaskEnding, TraceEvent};
    events
        .iter()
        .filter_map(|event| match event.event_type {
            EventType::TaskStart => {
                let data = event.parse_data::<TaskStartData>().ok()?;
                Some(TraceEvent::TaskStarted {
                    task_id: event.task_id.clone().unwrap_or(data.task_id),
                    description: data.description,
                })
            }
            EventType::ToolCall => {
                let data = event.parse_data::<ToolCallData>().ok()?;
                Some(TraceEvent::ToolCalled {
                    task_id: event.task_id.clone(),
                    name: data.name,
                    summary: data.summary,
                })
            }
            EventType::ToolResult => {
                let data = event.parse_data::<ToolResultData>().ok()?;
                Some(TraceEvent::ToolReturned {
                    task_id: event.task_id.clone(),
                    succeeded: data.succeeded(),
                    name: data.name,
                    result: data.result,
                })
            }
            EventType::Error => {
                let data = event.parse_data::<ErrorData>().ok()?;
                let message = data
                    .message
                    .lines()
                    .filter(|line| !crate::utils::is_internal_scaffolding_line(line))
                    .collect::<Vec<_>>()
                    .join("\n");
                Some(TraceEvent::Failed {
                    task_id: event.task_id.clone(),
                    message,
                    tool: data.tool_name,
                })
            }
            EventType::TaskEnd => {
                let data = event.parse_data::<TaskEndData>().ok()?;
                let task_id = event
                    .task_id
                    .clone()
                    .or_else(|| (!data.task_id.is_empty()).then(|| data.task_id.clone()))?;
                Some(TraceEvent::TaskEnded {
                    task_id,
                    outcome: match data.effective_outcome() {
                        TaskOutcome::Succeeded => TaskEnding::Succeeded,
                        TaskOutcome::Partial => TaskEnding::Partial,
                        TaskOutcome::Failed => TaskEnding::Failed,
                    },
                    duration_secs: data.duration_secs as f64,
                })
            }
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderKind;
    use crate::events::{DiagnosticSeverity, Event, EventType, TaskStatus};
    use crate::llm_runtime::SharedLlmRuntime;
    use crate::memory::embeddings::EmbeddingService;
    use crate::plans::PlanStore;
    use crate::state::SqliteStateStore;
    use crate::testing::MockProvider;
    use crate::traits::StateStore;
    use serde_json::json;
    use std::sync::Arc;

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

    async fn setup_consolidator_test() -> (
        Arc<SqliteStateStore>,
        Arc<EventStore>,
        Consolidator,
        tempfile::NamedTempFile,
    ) {
        let db_file = tempfile::NamedTempFile::new().expect("temp db file");
        let db_path = db_file.path().to_str().expect("db path");
        let embedding_service = Arc::new(EmbeddingService::new().expect("embedding service"));
        let state = Arc::new(
            SqliteStateStore::new(db_path, 100, None, embedding_service)
                .await
                .expect("state store"),
        );
        let event_store = Arc::new(EventStore::new(state.pool()).await.expect("event store"));
        let plan_store = Arc::new(PlanStore::new(state.pool()).await.expect("plan store"));
        let consolidator =
            Consolidator::new(event_store.clone(), plan_store, state.pool(), None, None)
                .with_state(state.clone() as Arc<dyn StateStore>)
                .with_memory(state.memory_stores());
        (state, event_store, consolidator, db_file)
    }

    async fn append_old_procedure_trace(
        event_store: &EventStore,
        session_id: &str,
        task_id: &str,
        execution_origin: Option<&str>,
    ) {
        let base = Utc::now() - Duration::days(9);
        let mut user_data = json!({
            "content": "Run the repeatable release workflow.",
            "message_id": format!("message-{task_id}"),
            "has_attachments": false,
            "user_role": "owner"
        });
        if let Some(origin) = execution_origin {
            user_data["execution_origin"] = json!(origin);
        }

        let events = vec![
            {
                user_data["task_id"] = json!(task_id);
                (EventType::UserMessage, user_data)
            },
            (
                EventType::TaskStart,
                json!({
                    "task_id": task_id,
                    "description": format!("release workflow for {task_id}")
                }),
            ),
            (
                EventType::MemoryPolicyCompiled,
                json!({
                    "task_id": task_id,
                    "access": "allowed",
                    "reason_code": "synthetic_test_policy",
                    "retrieval_suppressed": false,
                    "persistence_suppressed": false
                }),
            ),
            (
                EventType::ToolCall,
                json!({
                    "tool_call_id": format!("call-1-{task_id}"),
                    "name": "terminal",
                    "arguments": {"command": "cargo test"},
                    "summary": "run tests",
                    "task_id": task_id
                }),
            ),
            (
                EventType::ToolCall,
                json!({
                    "tool_call_id": format!("call-2-{task_id}"),
                    "name": "terminal",
                    "arguments": {"command": "cargo build --release"},
                    "summary": "build release",
                    "task_id": task_id
                }),
            ),
            (
                EventType::TaskEnd,
                serde_json::to_value(TaskEndData {
                    task_id: task_id.to_string(),
                    status: TaskStatus::Completed,
                    outcome: Some(TaskOutcome::Succeeded),
                    duration_secs: 30,
                    iterations: 2,
                    tool_calls_count: 2,
                    error: None,
                    summary: Some("release verified".to_string()),
                    efficiency: None,
                    turn_id: None,
                    completion_proof: None,
                    harness_eval: None,
                })
                .expect("serialize task end"),
            ),
        ];

        for (offset, (event_type, data)) in events.into_iter().enumerate() {
            let mut event = Event::new(session_id, event_type, data);
            event.created_at = base + Duration::seconds(offset as i64);
            event_store.append(event).await.expect("append old event");
        }
    }

    /// With another memory system learning, old events must still reach
    /// retention, which only deletes what consolidation has marked.
    #[tokio::test]
    async fn retiring_old_events_marks_them_without_learning() {
        let db_file = tempfile::NamedTempFile::new().expect("temp db file");
        let db_path = db_file.path().to_str().expect("db path");
        let embedding_service = Arc::new(EmbeddingService::new().expect("embedding service"));
        let state = Arc::new(
            SqliteStateStore::new(db_path, 100, None, embedding_service)
                .await
                .expect("state store"),
        );
        let event_store = Arc::new(EventStore::new(state.pool()).await.expect("event store"));
        let plan_store = Arc::new(PlanStore::new(state.pool()).await.expect("plan store"));
        let provider = Arc::new(MockProvider::with_responses(vec![]));
        let runtime = SharedLlmRuntime::new(
            provider.clone(),
            None,
            ProviderKind::OpenaiCompatible,
            "mock".to_string(),
        );
        let consolidator = Arc::new(
            Consolidator::new(
                event_store.clone(),
                plan_store,
                state.pool(),
                Some(runtime),
                None,
            )
            .with_state(state.clone() as Arc<dyn StateStore>)
            .with_memory(state.memory_stores()),
        );
        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at)
             VALUES ('ordinary-owner-session', 'user_message', '{\"content\":\"hello\"}',
                     '2025-01-01T00:00:00+00:00')",
        )
        .execute(&state.pool())
        .await
        .expect("old event");

        let stats = Pruner::new(event_store, consolidator, 7)
            .retire_old_events()
            .await
            .expect("retire pass");

        assert_eq!(stats.consolidation_errors, 0);
        assert_eq!(provider.call_count().await, 0, "nothing is learned");
        let unmarked: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM events WHERE consolidated_at IS NULL")
                .fetch_one(&state.pool())
                .await
                .expect("count");
        assert_eq!(unmarked, 0);
    }

    #[tokio::test]
    async fn pruner_marks_mandate_and_internal_sessions_without_learning() {
        let db_file = tempfile::NamedTempFile::new().expect("temp db file");
        let db_path = db_file.path().to_str().expect("db path");
        let embedding_service = Arc::new(EmbeddingService::new().expect("embedding service"));
        let state = Arc::new(
            SqliteStateStore::new(db_path, 100, None, embedding_service)
                .await
                .expect("state store"),
        );
        let event_store = Arc::new(EventStore::new(state.pool()).await.expect("event store"));
        let plan_store = Arc::new(PlanStore::new(state.pool()).await.expect("plan store"));
        let provider = Arc::new(MockProvider::with_responses(vec![
            MockProvider::text_response(
                r#"{
                    "name":"verified-release-workflow",
                    "trigger_pattern":"when preparing a verified release",
                    "steps":["run tests","build release"],
                    "skip":false
                }"#,
            ),
        ]));
        let runtime = SharedLlmRuntime::new(
            provider.clone(),
            None,
            ProviderKind::OpenaiCompatible,
            "mock".to_string(),
        );
        let consolidator = Arc::new(
            Consolidator::new(
                event_store.clone(),
                plan_store,
                state.pool(),
                Some(runtime),
                None,
            )
            .with_state(state.clone() as Arc<dyn StateStore>)
            .with_memory(state.memory_stores()),
        );

        // Deliberately use an owner-looking session ID for the mandate: the
        // explicit event provenance, not a prefix or role, must isolate it.
        append_old_procedure_trace(
            &event_store,
            "owner-looking-mandate-session",
            "mandate-task",
            Some("mandate"),
        )
        .await;
        append_old_procedure_trace(
            &event_store,
            "specialist:executor:isolated",
            "specialist-task",
            None,
        )
        .await;
        append_old_procedure_trace(
            &event_store,
            "ordinary-owner-session",
            "ordinary-task",
            None,
        )
        .await;
        append_old_procedure_trace(
            &event_store,
            "malformed-origin-session",
            "malformed-task",
            None,
        )
        .await;
        sqlx::query(
            "UPDATE events SET data = '{not-json'
             WHERE session_id = 'malformed-origin-session'
               AND event_type = 'user_message'",
        )
        .execute(&state.pool())
        .await
        .expect("corrupt legacy user-message payload");

        let stats = Pruner::new(event_store.clone(), consolidator, 7)
            .prune()
            .await
            .expect("prune consolidation pass");
        assert_eq!(stats.consolidation_errors, 0);
        assert_eq!(provider.call_count().await, 1);

        let procedures: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM procedures")
            .fetch_one(&state.pool())
            .await
            .expect("count procedures");
        assert_eq!(procedures, 1, "only the ordinary owner session may learn");

        let learned_trigger: String =
            sqlx::query_scalar("SELECT trigger_pattern FROM procedures LIMIT 1")
                .fetch_one(&state.pool())
                .await
                .expect("ordinary procedure");
        assert_eq!(learned_trigger, "when preparing a verified release");

        for session_id in [
            "owner-looking-mandate-session",
            "specialist:executor:isolated",
            "malformed-origin-session",
            "ordinary-owner-session",
        ] {
            assert!(
                event_store
                    .query_unconsolidated(session_id)
                    .await
                    .expect("query unconsolidated")
                    .is_empty(),
                "{session_id} should not remain a perpetual prune candidate"
            );
        }
    }

    #[tokio::test]
    async fn test_consolidation_records_failure_patterns_from_warning_decisions() {
        let (state, event_store, consolidator, _db_file) = setup_consolidator_test().await;

        let task_id = "task-failure-pattern";
        let mut task_start = Event::new(
            "session-failure-pattern",
            EventType::TaskStart,
            json!({
                "task_id": task_id,
                "description": "debug repetitive terminal loop"
            }),
        );
        task_start.created_at = Utc::now();
        event_store
            .append(task_start)
            .await
            .expect("append task start");

        event_store
            .append(Event::new(
                "session-failure-pattern",
                EventType::MemoryPolicyCompiled,
                json!({
                    "task_id": task_id,
                    "access": "allowed",
                    "reason_code": "synthetic_test_policy",
                    "retrieval_suppressed": false,
                    "persistence_suppressed": false
                }),
            ))
            .await
            .expect("append memory policy");

        let mut decision = Event::new(
            "session-failure-pattern",
            EventType::DecisionPoint,
            serde_json::to_value(DecisionPointData {
                decision_type: crate::events::DecisionType::RepetitiveCallDetection,
                task_id: task_id.to_string(),
                iteration: 4,
                severity: DiagnosticSeverity::Warning,
                code: Some("repetitive_call_detection".to_string()),
                metadata: json!({
                    "tool": "terminal",
                    "count": 4,
                    "action": "hard_stop"
                }),
                summary: "Repetitive tool call hard-stopped for terminal (count=4)".to_string(),
            })
            .expect("serialize decision"),
        );
        decision.created_at = Utc::now();
        event_store.append(decision).await.expect("append decision");

        let mut task_end = Event::new(
            "session-failure-pattern",
            EventType::TaskEnd,
            serde_json::to_value(TaskEndData {
                task_id: task_id.to_string(),
                status: TaskStatus::Failed,
                outcome: Some(crate::events::TaskOutcome::Failed),
                duration_secs: 3,
                iterations: 4,
                tool_calls_count: 4,
                error: Some("Repetitive tool calls".to_string()),
                summary: Some("Agent stopped due to repetitive terminal loop".to_string()),
                efficiency: None,
                turn_id: None,
                completion_proof: None,
                harness_eval: None,
            })
            .expect("serialize task end"),
        );
        task_end.created_at = Utc::now();
        event_store.append(task_end).await.expect("append task end");

        let result = consolidator
            .consolidate_session("session-failure-pattern")
            .await
            .expect("consolidate session");
        assert_eq!(result.behavior_patterns_recorded, 1);

        let patterns = state
            .get_behavior_patterns(0.0)
            .await
            .expect("get behavior patterns");
        let pattern = patterns
            .iter()
            .find(|pattern| {
                pattern.trigger_context.as_deref() == Some("repetitive_call_detection:terminal")
            })
            .expect("repetitive failure pattern");
        assert_eq!(pattern.pattern_type, "failure");
        assert_eq!(
            pattern.action.as_deref(),
            Some("stop_retrying_identical_calls")
        );
        assert!(
            pattern
                .description
                .contains("stop retrying the same operation"),
            "pattern description was: {}",
            pattern.description
        );
        assert!(pattern.confidence >= 0.8);
    }

    #[tokio::test]
    async fn test_consolidation_records_evidence_gate_failure_patterns() {
        let (state, event_store, consolidator, _db_file) = setup_consolidator_test().await;

        let task_id = "task-evidence-pattern";
        let mut task_start = Event::new(
            "session-evidence-pattern",
            EventType::TaskStart,
            json!({
                "task_id": task_id,
                "description": "edit src/main.rs"
            }),
        );
        task_start.created_at = Utc::now();
        event_store
            .append(task_start)
            .await
            .expect("append task start");

        event_store
            .append(Event::new(
                "session-evidence-pattern",
                EventType::MemoryPolicyCompiled,
                json!({
                    "task_id": task_id,
                    "access": "allowed",
                    "reason_code": "synthetic_test_policy",
                    "retrieval_suppressed": false,
                    "persistence_suppressed": false
                }),
            ))
            .await
            .expect("append memory policy");

        let mut decision = Event::new(
            "session-evidence-pattern",
            EventType::DecisionPoint,
            serde_json::to_value(DecisionPointData {
                decision_type: crate::events::DecisionType::EvidenceGate,
                task_id: task_id.to_string(),
                iteration: 1,
                severity: DiagnosticSeverity::Warning,
                code: Some("missing_pre_execution_evidence".to_string()),
                metadata: json!({
                    "condition": "missing_pre_execution_evidence",
                    "tool": "edit_file",
                    "required_evidence_kind": "file_read",
                    "target": "src/main.rs"
                }),
                summary: "Blocked edit_file until required evidence is gathered".to_string(),
            })
            .expect("serialize decision"),
        );
        decision.created_at = Utc::now();
        event_store.append(decision).await.expect("append decision");

        let mut task_end = Event::new(
            "session-evidence-pattern",
            EventType::TaskEnd,
            serde_json::to_value(TaskEndData {
                task_id: task_id.to_string(),
                status: TaskStatus::Failed,
                outcome: Some(crate::events::TaskOutcome::Failed),
                duration_secs: 2,
                iterations: 1,
                tool_calls_count: 0,
                error: Some("missing evidence".to_string()),
                summary: None,
                efficiency: None,
                turn_id: None,
                completion_proof: None,
                harness_eval: None,
            })
            .expect("serialize task end"),
        );
        task_end.created_at = Utc::now();
        event_store.append(task_end).await.expect("append task end");

        let result = consolidator
            .consolidate_session("session-evidence-pattern")
            .await
            .expect("consolidate session");
        assert_eq!(result.behavior_patterns_recorded, 1);

        let patterns = state
            .get_behavior_patterns(0.0)
            .await
            .expect("get behavior patterns");
        let pattern = patterns
            .iter()
            .find(|pattern| {
                pattern.trigger_context.as_deref()
                    == Some("missing_pre_execution_evidence:edit_file:file_read")
            })
            .expect("evidence gate failure pattern");
        assert_eq!(pattern.pattern_type, "failure");
        assert_eq!(
            pattern.action.as_deref(),
            Some("gather_direct_evidence_before_mutation")
        );
        assert!(pattern.description.contains("gather file_read"));
    }
}
