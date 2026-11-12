//! Event consolidation: the daemon's side of learning from finished work.
//!
//! The daemon decides which of a session's events may teach anything
//! (isolated sessions and suppressed tasks may not), writes the rest as the
//! memory crate's task trace, and hands it to the crate's learner. After
//! that the events are marked, so retention can remove them.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use aidaemon_memory::stores::MemoryStores;
use aidaemon_memory::trajectory::{
    CompletedPlan, DecisionDetails, DecisionPoint, ProcedureEnhancer, TaskEnding, TaskLearner,
    TraceEvent,
};
use chrono::{Duration, Utc};
use tracing::{info, warn};

use super::{
    DecisionPointData, ErrorData, TaskEndData, TaskStartData, ToolCallData, ToolResultData,
    UserMessageData,
};
use super::{Event, EventStore, EventType, TaskOutcome};
use crate::llm_runtime::gateway::SideCallGateway;
use crate::llm_runtime::side_calls::SideCall;
use crate::llm_runtime::SharedLlmRuntime;
use crate::memory::embeddings::EmbeddingService;
use crate::plans::{PlanStore, StepStatus};
use crate::traits::MessageAnnotation;

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
    memory: Arc<dyn MemoryStores>,
    llm_runtime: Option<SharedLlmRuntime>,
    embedding_service: Option<Arc<EmbeddingService>>,
    /// Runs the procedure enhancer's model calls.
    side_calls: SideCallGateway,
    learning_evidence_gate_enforce: bool,
}

impl Consolidator {
    pub fn new(
        event_store: Arc<EventStore>,
        plan_store: Arc<PlanStore>,
        pool: sqlx::SqlitePool,
        memory: Arc<dyn MemoryStores>,
        llm_runtime: Option<SharedLlmRuntime>,
        side_calls: SideCallGateway,
        embedding_service: Option<Arc<EmbeddingService>>,
    ) -> Self {
        Self {
            event_store,
            plan_store,
            pool,
            memory,
            llm_runtime,
            embedding_service,
            side_calls,
            learning_evidence_gate_enforce: false,
        }
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

        let plans = self.completed_plans(session_id, &suppressed_task_ids).await;
        let report = self.learner().learn(&trace_events(&events), &plans).await;

        // Note: Episode creation is handled exclusively by the memory jobs (LLM-quality episodes)
        let event_ids: Vec<i64> = events.iter().map(|e| e.id).collect();
        self.event_store.mark_consolidated(&event_ids).await?;

        let result = ConsolidationResult {
            events_processed: events.len() + suppressed_events.len(),
            events_consolidated: suppressed_events.len() + event_ids.len(),
            procedures_created: report.procedures,
            error_solutions_created: report.error_solutions,
            expertise_updated: report.expertise_updates,
            behavior_patterns_recorded: report.failure_patterns,
            ..Default::default()
        };
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

    /// The memory crate's learner, with the daemon's embeddings and, when
    /// it has a model, a procedure enhancer that calls through the gateway.
    fn learner(&self) -> TaskLearner {
        let mut learner = TaskLearner::new(self.memory.clone())
            .with_evidence_gate(self.learning_evidence_gate_enforce);
        if let Some(embeddings) = &self.embedding_service {
            learner = learner.with_embedder(embeddings.clone());
        }
        if let Some(runtime) = &self.llm_runtime {
            // The side-call gateway picks the model; this value is not sent.
            let model = runtime.snapshot().primary_model();
            let provider = self.side_calls.provider(SideCall::ProcedureEnhancement);
            learner = learner.with_enhancer(ProcedureEnhancer::new(provider, model));
        }
        learner
    }

    /// The session's plans finished in the last day, without suppressed
    /// tasks' ones. A lookup failure means no plans, not a failed pass.
    async fn completed_plans(
        &self,
        session_id: &str,
        suppressed_task_ids: &HashSet<String>,
    ) -> Vec<CompletedPlan> {
        let since = Utc::now() - Duration::hours(24);
        let Ok(plans) = self.plan_store.get_completed_since(session_id, since).await else {
            return Vec::new();
        };
        plans
            .into_iter()
            .filter(|plan| {
                !plan
                    .task_id
                    .as_ref()
                    .is_some_and(|task_id| suppressed_task_ids.contains(task_id))
            })
            .map(|plan| CompletedPlan {
                completed_steps: plan
                    .steps
                    .iter()
                    .filter(|step| step.status == StepStatus::Completed)
                    .map(|step| step.description.clone())
                    .collect(),
                id: plan.id,
                description: plan.description,
            })
            .collect()
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

fn task_ending(outcome: TaskOutcome) -> TaskEnding {
    match outcome {
        TaskOutcome::Succeeded => TaskEnding::Succeeded,
        TaskOutcome::Partial => TaskEnding::Partial,
        TaskOutcome::Failed => TaskEnding::Failed,
    }
}

/// How each task's last end event says it ended; `None` when that event
/// cannot be read. Only events indexed under the task count.
fn last_task_endings(events: &[Event]) -> HashMap<&str, Option<TaskEnding>> {
    let mut endings = HashMap::new();
    for event in events {
        if event.event_type != EventType::TaskEnd {
            continue;
        }
        if let Some(task_id) = &event.task_id {
            let ending = event
                .parse_data::<TaskEndData>()
                .ok()
                .map(|data| task_ending(data.effective_outcome()));
            endings.insert(task_id.as_str(), ending);
        }
    }
    endings
}

/// This session's events as the SDK memory crate's task trace. Error
/// messages lose the daemon's own scaffolding lines (untrusted-data
/// envelopes, truncation notices), which would otherwise make unrelated
/// errors share a pattern.
fn trace_events(events: &[Event]) -> Vec<TraceEvent> {
    let endings = last_task_endings(events);
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
                // A row missing its indexed tool name is an old partial
                // record: its name counts, its arguments are not offered.
                let whole = event.tool_name.as_deref() == Some(data.name.as_str());
                Some(TraceEvent::ToolCalled {
                    task_id: event.task_id.clone(),
                    arguments: whole.then_some(data.arguments),
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
                    outcome: task_ending(data.effective_outcome()),
                    duration_secs: data.duration_secs as f64,
                })
            }
            EventType::UserMessage => {
                let data = event.parse_data::<UserMessageData>().ok()?;
                (!data
                    .annotations
                    .contains(&MessageAnnotation::InternalContinuation))
                .then_some(TraceEvent::UserMessage {
                    content: data.content,
                    at: event.created_at,
                })
            }
            EventType::DecisionPoint => {
                let task_id = event.task_id.clone()?;
                let data = event.parse_data::<DecisionPointData>().ok()?;
                let task_ending = endings.get(task_id.as_str()).copied().flatten();
                Some(TraceEvent::DecisionPoint(DecisionPoint {
                    task_id,
                    code: data
                        .code
                        .unwrap_or_else(|| data.decision_type.as_str().to_string()),
                    warning: data.severity.is_warning_or_higher(),
                    details: DecisionDetails::from_metadata(&data.metadata),
                    task_ending,
                }))
            }
            _ => None,
        })
        .collect()
}

#[cfg(test)]
#[path = "consolidation_golden_tests.rs"]
mod golden_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderKind;
    use crate::events::EventStoreForTests as _;
    use crate::events::{DiagnosticSeverity, Event, EventType, TaskStatus};
    use crate::llm_runtime::SharedLlmRuntime;
    use crate::memory::embeddings::EmbeddingService;
    use crate::plans::PlanStore;
    use crate::state::SqliteStateStore;
    use crate::testing::MockProvider;
    use serde_json::json;
    use std::sync::Arc;

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
        let consolidator = Consolidator::new(
            event_store.clone(),
            plan_store,
            state.pool(),
            state.memory_stores(),
            None,
            unused_side_calls(&state, &event_store),
            None,
        );
        (state, event_store, consolidator, db_file)
    }

    /// A gateway for a consolidator that has no model to call.
    fn unused_side_calls(
        state: &Arc<SqliteStateStore>,
        event_store: &Arc<EventStore>,
    ) -> SideCallGateway {
        let runtime = SharedLlmRuntime::new(
            Arc::new(MockProvider::with_responses(vec![])),
            None,
            ProviderKind::OpenaiCompatible,
            "mock".to_string(),
        );
        SideCallGateway::new(runtime, state.clone(), event_store.clone())
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
        let consolidator = Arc::new(Consolidator::new(
            event_store.clone(),
            plan_store,
            state.pool(),
            state.memory_stores(),
            Some(runtime.clone()),
            SideCallGateway::new(runtime, state.clone(), event_store.clone()),
            None,
        ));
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
        let consolidator = Arc::new(Consolidator::new(
            event_store.clone(),
            plan_store,
            state.pool(),
            state.memory_stores(),
            Some(runtime.clone()),
            SideCallGateway::new(runtime, state.clone(), event_store.clone()),
            None,
        ));

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

    /// "Don't remember this": consolidation later passes over an opted-out
    /// turn's events without learning, and still learns from a normal turn.
    #[tokio::test]
    async fn consolidation_skips_a_turn_opted_out_of_learning() {
        let (state, event_store, consolidator, _db_file) = setup_consolidator_test().await;
        append_old_procedure_trace(&event_store, "opted-out-session", "private-task", None).await;
        append_old_procedure_trace(&event_store, "normal-session", "normal-task", None).await;
        crate::agent::memory_plugin::mark_turn_opted_out(
            &crate::events::EventEmitter::new(event_store.clone(), "opted-out-session"),
            "private-task",
            "message-private-task",
        )
        .await
        .expect("mark opted out");

        let skipped = consolidator
            .consolidate_session("opted-out-session")
            .await
            .expect("consolidate opted-out session");
        assert_eq!(skipped.procedures_created, 0);
        assert!(event_store
            .query_unconsolidated("opted-out-session")
            .await
            .unwrap()
            .is_empty());
        let procedures: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM procedures")
            .fetch_one(&state.pool())
            .await
            .unwrap();
        assert_eq!(procedures, 0, "nothing learned from the opted-out turn");

        let learned = consolidator
            .consolidate_session("normal-session")
            .await
            .expect("consolidate normal session");
        assert_eq!(learned.procedures_created, 1, "a normal turn still learns");
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
