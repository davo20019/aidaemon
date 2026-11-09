//! The previous consolidation path, kept only so the parity test can run it
//! beside the new one. Copied unchanged except for the method name, the
//! trace's new `arguments` field, and `memory` no longer being optional.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{Duration, Utc};
use sqlx::Row;
use tracing::{info, warn};

use super::super::{
    DecisionPointData, ErrorData, Event, EventType, TaskEndData, TaskOutcome, TaskStartData,
    ToolCallData, ToolResultData,
};
use super::{ConsolidationResult, Consolidator};
use crate::plans::StepStatus;
use crate::traits::{ErrorSolution, Procedure};
use aidaemon_memory::encode_embedding;

#[derive(Debug, Clone)]
struct FailurePatternObservation {
    trigger_context: String,
    action: String,
    description: String,
    confidence: f32,
}

impl Consolidator {
    /// Consolidate events for a session into long-term memory
    pub(super) async fn consolidate_session_legacy(
        &self,
        session_id: &str,
    ) -> anyhow::Result<ConsolidationResult> {
        // Resolve isolation before deserializing any event payload. This also
        // makes malformed legacy JSON fail closed: the provenance resolver
        // classifies the session as isolated, and we can retire its rows by ID
        // without asking EventStore to parse the unreadable payload.
        let memory_isolated = super::super::is_memory_isolated_worker_session(session_id)
            || super::super::session_has_mandate_execution_provenance(&self.pool, session_id)
                .await?;
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
                if let Ok(data) = event.parse_data::<super::super::UserMessageData>() {
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
                if let Ok(data) = event.parse_data::<ToolCallData>() {
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
        let memory = &self.memory;

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
            .filter_map(|blob| aidaemon_memory::decode_embedding_with_dim(&blob, 384).ok())
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
                    arguments: None,
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
